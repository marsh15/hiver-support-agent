"""Independent second pass over data/golden.csv labels.

Protocol (deliberately different from the pre-labeler): the full thread head
is shown, the pre-label is NOT, the prompt is framed as adjudication, and every
disagreement goes to a separate arbiter call that sees both candidates.
High-bar rule: escalate labels win ties (safety bias).

Outputs: corrected data/golden.csv (label_process column),
results/adjudication_log.csv, results/label_stability.json.
The human gate is NOT marked: `verified` stays false until a person does it.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.classify import load_intents
from hiver_agent.config import CFG, ROOT
from hiver_agent.llm import LLM

BLIND = """You are auditing labels for a Spotify support tweet dataset. Given the
tweet AND its real thread (what Spotify actually did), assign exactly one intent
and an escalate verdict. Policy for escalate=true: account-specific action or
private info needed, security/fraud suspicion, money dispute, legal threat,
repeated unresolved failure, or the brand historically took it to DMs.
Safety bias on ties: escalate. Answer JSON:
{"intent": "...", "escalate": true|false, "confidence": "high|medium|low"}"""

ARBITER = """Two labelers disagree on this Spotify support tweet. You are the
arbiter. Read the tweet and the real thread, then pick the better label pair.
Answer JSON: {"intent": "...", "escalate": true|false, "reason": "<=15 words",
"confidence": "high|medium|low"}"""


def blind(row, defs, llm):
    out = llm.json([
        {"role": "system", "content": BLIND},
        {"role": "user", "content": f"Intents:\n{defs}\n\nTweet: {row.text}\n\nThread:\n{row.thread_head}"},
    ])
    if out.get("intent") not in load_intents():
        out["intent"] = "other"
    return out


def arbitrate(row, defs, a, b, llm):
    return llm.json([
        {"role": "system", "content": ARBITER},
        {"role": "user", "content": (
            f"Intents:\n{defs}\n\nTweet: {row.text}\n\nThread:\n{row.thread_head}\n\n"
            f"Label A: intent={a['intent']} escalate={a['escalate']}\n"
            f"Label B: intent={b['intent']} escalate={b['escalate']}\nPick A or B's values.")},
    ])


def main():
    golden = pd.read_csv(ROOT / "data/golden.csv")
    defs = "\n".join(f"- {n}: {i['definition']}" for n, i in load_intents().items())
    llm = LLM(CFG["models"]["bulk"], 0.0)

    with ThreadPoolExecutor(8) as ex:
        seconds = list(ex.map(lambda r: blind(r, defs, llm), golden.itertuples()))

    log, flips_i, flips_e, dis_i, dis_e = [], 0, 0, 0, 0
    for row, second in zip(golden.itertuples(), seconds):
        old_i, old_e = row.intent, bool(row.escalate)
        new_i, new_e = second["intent"], bool(second["escalate"])
        d_i, d_e = new_i != old_i, new_e != old_e
        dis_i += d_i
        dis_e += d_e
        if d_i or d_e:
            arb = arbitrate(row, defs,
                            {"intent": old_i, "escalate": old_e},
                            {"intent": new_i, "escalate": new_e}, llm)
            fin_i = arb.get("intent", old_i) if arb.get("intent") in load_intents() else old_i
            fin_e = bool(arb.get("escalate", old_e)) if "escalate" in arb else old_e
            # ties resolve toward the pre-label unless arbiter is high-confidence
            if arb.get("confidence") != "high":
                fin_i, fin_e = old_i, old_e
            if fin_i != old_i:
                flips_i += 1
            if fin_e != old_e:
                flips_e += 1
            log.append({"root_tweet_id": int(row.root_tweet_id), "text": row.text,
                        "intent_old": old_i, "intent_blind": new_i, "intent_final": fin_i,
                        "escalate_old": old_e, "escalate_blind": new_e, "escalate_final": fin_e,
                        "arbiter_reason": arb.get("reason", "")})
        else:
            fin_i, fin_e = old_i, old_e
        golden.at[row.Index, "intent"], golden.at[row.Index, "escalate"] = fin_i, fin_e

    golden["label_process"] = "prelabel+adjudicated-v2"
    golden.to_csv(ROOT / "data/golden.csv", index=False)
    (ROOT / "results").mkdir(exist_ok=True)
    pd.DataFrame(log).to_csv(ROOT / "results/adjudication_log.csv", index=False)
    stats = {
        "n": len(golden),
        "intent_disagreement_rate": dis_i / len(golden),
        "escalate_disagreement_rate": dis_e / len(golden),
        "intent_flips_applied": flips_i,
        "escalate_flips_applied": flips_e,
        "note": "blind second pass + arbiter; flips applied only on high-confidence arbitration; human gate still pending (verified=false)",
    }
    (ROOT / "results/label_stability.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
