"""S7: run any system over the golden set; automated metrics + LLM-judge
rubric; bootstrap CIs; writes results/<system>/{metrics.json,predictions.csv,
judged.csv}. Cached LLM calls make re-runs free."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent import draft as draft_mod
from hiver_agent import judge as judge_mod
from hiver_agent import metrics as M
from hiver_agent.agent import Agent
from hiver_agent.baselines import SimpleAgent, TrivialAgent
from hiver_agent.config import CFG, ROOT

SYSTEMS = ["agent", "trivial", "simple", "noretr"]


def get_system(name: str):
    corpus = pd.read_parquet(ROOT / "data/processed/corpus_labeled.parquet")
    if name == "agent":
        return Agent(use_retrieval=True)
    if name == "noretr":
        return Agent(use_retrieval=False)
    if name == "trivial":
        t = TrivialAgent()
        t.fit(corpus.intent.tolist())
        return t
    if name == "simple":
        s = SimpleAgent()
        s.fit(corpus.text.tolist(), corpus.intent.tolist(), corpus.brand_reply.tolist())
        return s
    raise SystemExit(f"unknown system {name}")


def judge_all(preds: pd.DataFrame, index) -> list[dict]:
    playbooks = draft_mod.load_playbooks()

    def one(row):
        # holdout: never judge grounding against the tweet's own historical thread
        exemplars = (index.retrieve_one(row.text, 4, {row.root_tweet_id})
                     if index is not None else [])
        playbook = playbooks.get(row.intent, {}).get("playbook", "")
        return judge_mod.judge(row.text, row.reply, row.intent, playbook, exemplars)

    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(one, preds.itertuples()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    args = ap.parse_args()
    golden = pd.read_csv(ROOT / "data/golden.csv")
    system = get_system(args.system)

    # self-thread holdout: the tweet's own historical thread is banned from its
    # retrieval at eval time (it would otherwise match verbatim, similarity 1.0)
    with ThreadPoolExecutor(8) as ex:
        outs = list(ex.map(
            lambda i: system.handle(golden.text[i],
                                    exclude_ids={int(golden.root_tweet_id[i])}),
            range(len(golden))))
    preds = golden[["root_tweet_id", "text", "intent", "escalate"]].copy()
    preds["intent_pred"] = [o["intent"] for o in outs]
    preds["confidence"] = [o.get("confidence", "") for o in outs]
    preds["reply"] = [o["reply"] for o in outs]
    preds["escalate_pred"] = [o["escalate"] for o in outs]
    preds["escalate_reason"] = [o.get("escalate_reason", "") for o in outs]
    preds["grounding_thread_ids"] = [str(o.get("grounding_thread_ids", "")) for o in outs]
    preds["reply_violations"] = [str(o.get("reply_violations", "")) for o in outs]

    y_i, y_p = preds.intent.tolist(), preds.intent_pred.tolist()
    y_e = preds.escalate.astype(bool).tolist()
    y_ep = preds.escalate_pred.astype(bool).tolist()

    # leakage regression guard: a tweet's own thread must never be its own
    # grounding (100% self-match made the first headline run a lie — see
    # REPORT §6.1). Fail the run loudly instead of shipping a inflated number.
    leaks = preds.apply(
        lambda r: str(int(r.root_tweet_id)) in str(r.grounding_thread_ids)
        and not r.grounding_thread_ids.startswith("[]"), axis=1)
    if leaks.any():
        raise SystemExit(
            f"LEAKAGE GUARD: {int(leaks.sum())} golden tweets retrieved their own "
            "thread as grounding — the self-thread holdout broke. Refusing to report.")

    im = M.intent_metrics(y_i, y_p)
    em = M.escalation_metrics(y_e, y_ep)

    n = CFG["bootstrap"]["n_resamples"]
    rng = np.random.default_rng(CFG["bootstrap"]["seed"])
    takes = [rng.integers(0, len(preds), len(preds)) for _ in range(n)]
    im["accuracy_ci95"] = M.bootstrap_ci([int(a == b) for a, b in zip(y_i, y_p)], n=n)
    im["macro_f1_ci95"] = [float(x) for x in np.percentile(
        [M.intent_metrics(np.take(y_i, t).tolist(), np.take(y_p, t).tolist())["macro_f1"]
         for t in takes], [2.5, 97.5])]
    em["auto_handle_safety_ci95"] = [float(x) for x in np.percentile(
        [M.escalation_metrics(np.take(y_e, t).tolist(), np.take(y_ep, t).tolist())["auto_handle_safety"]
         for t in takes], [2.5, 97.5])]

    index = system.index if hasattr(system, "index") else None
    judged = judge_all(preds, index)
    for k in ["groundedness", "actionability", "tone", "safety"]:
        preds[f"judge_{k}"] = [j[k] for j in judged]
    preds["judge_pass"] = [j["pass"] for j in judged]
    preds["judge_rationale"] = [j.get("rationale", "") for j in judged]

    m = {
        "system": args.system,
        "n": len(preds),
        "golden_verified": bool(golden.verified.astype(bool).all()),
        "intent": im,
        "escalation": em,
        "reply_constraint_violation_rate": float(
            preds.reply_violations.str.strip().str.len().gt(2).mean()),
        "judge": {"mean_" + k: float(preds[f"judge_{k}"].mean())
                  for k in ["groundedness", "actionability", "tone", "safety"]}
        | {"pass_rate": float(preds.judge_pass.mean())},
    }
    out = ROOT / "results" / args.system
    out.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(m, indent=2, default=float))
    (out / "confusion.csv").write_text(pd.DataFrame(M.confusion(y_i, y_p)).to_csv())
    print(json.dumps(m, indent=2, default=float))

    if args.system == "agent":  # human judge-validation sheet
        js = preds.sample(n=min(CFG["judge"]["human_validation_sample"], len(preds)), random_state=5)
        js = js[["root_tweet_id", "text", "reply"]]
        for k in ["groundedness", "actionability", "tone", "safety", "pass"]:
            js[f"human_{k}"] = ""
        (ROOT / "labeling").mkdir(exist_ok=True)
        js.to_csv(ROOT / "labeling/judge_sheet.csv", index=False)
        print("wrote labeling/judge_sheet.csv — HUMAN GATE: score 1-5, then scripts/judge_agreement.py")


if __name__ == "__main__":
    main()
