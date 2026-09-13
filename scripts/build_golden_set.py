"""S3: build the golden evaluation set. Stratified 200 from the labeled pool
with LLM pre-labels; escalation pre-label = documented heuristic (DM-redirect,
always-escalate intent, anger/legal keywords). The human verifies/corrects
`intent` and `escalate` columns in data/golden.csv (labeling/ has the reading
sheet + blind relabel file for self-agreement)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.classify import load_intents
from hiver_agent.config import CFG, ROOT
from hiver_agent.escalate import ANGER_RE, load_intents_escalating


def heuristic_escalate(row) -> tuple[bool, str]:
    if row.intent in load_intents_escalating():
        return True, f"heuristic: intent '{row.intent}' always escalates"
    if ANGER_RE.search(row.text):
        return True, "heuristic: anger/legal keyword"
    if row.dm_redirect:
        return True, "heuristic: brand historically moved this to DM"
    return False, "heuristic: historically answered publicly"


def main():
    pool = pd.read_parquet(ROOT / "data/processed/corpus_labeled.parquet")
    size, min_per = CFG["golden"]["size"], CFG["golden"]["min_per_intent"]
    rng = np.random.default_rng(13)

    idx_by_intent = {i: pool.index[pool.intent == i].tolist() for i in load_intents()}
    chosen = set()
    for i, ids in idx_by_intent.items():  # guarantee representation
        take = min(min_per, len(ids))
        chosen.update(rng.choice(ids, take, replace=False).tolist() if len(ids) > take else ids)
    rest = pool.index.difference(sorted(chosen)).tolist()
    need = size - len(chosen)
    if need > 0 and rest:
        chosen.update(rng.choice(rest, min(need, len(rest)), replace=False).tolist())

    g = pool.loc[sorted(chosen)].copy()
    esc = g.apply(heuristic_escalate, axis=1)
    g["pre_escalate"], g["pre_escalate_reason"] = [e[0] for e in esc], [e[1] for e in esc]
    g = g.rename(columns={"intent": "pre_intent", "confidence": "pre_confidence"})
    g["intent"] = g.pre_intent          # final labels start as pre-labels;
    g["escalate"] = g.pre_escalate      # human corrects these two columns
    g["verified"] = False
    cols = ["root_tweet_id", "text", "thread_head", "brand_reply", "dm_redirect",
            "pre_intent", "pre_confidence", "pre_escalate", "pre_escalate_reason",
            "intent", "escalate", "verified"]
    g = g[cols]

    (ROOT / "labeling").mkdir(exist_ok=True)
    g.to_csv(ROOT / "data/golden.csv", index=False)

    # blind self-agreement file: same rows, shuffled, labels blank
    relabel = g[["root_tweet_id", "text", "thread_head"]].sample(frac=1.0, random_state=99)
    relabel["intent"] = ""
    relabel["escalate"] = ""
    relabel.to_csv(ROOT / "labeling/relabel_30.csv", index=False)  # human fills first 30
    relabel.head(30).to_csv(ROOT / "labeling/relabel_30.csv", index=False)

    sheet = (ROOT / "labeling/golden_sheet.html")
    rows = "\n".join(
        f"<tr><td>{r.root_tweet_id}</td><td><pre>{r.text}</pre><details><summary>thread</summary>"
        f"<pre>{r.thread_head}</pre></details></td><td>{r.pre_intent} ({r.pre_confidence})</td>"
        f"<td>{r.pre_escalate}</td></tr>"
        for r in g.itertuples()
    )
    sheet.write_text(
        "<html><body><h1>Golden set verification sheet</h1>"
        "<p>Read each tweet + thread, then correct <code>intent</code> and <code>escalate</code> "
        "in data/golden.csv. Set verified=true when done.</p>"
        "<table border=1 cellpadding=4><tr><th>id</th><th>tweet</th><th>pre-intent</th>"
        f"<th>pre-escalate</th></tr>{rows}</table></body></html>"
    )
    print(f"golden set: {len(g)} rows ({g.intent.value_counts().to_dict()})")
    print(f"escalate pre-labels: {int(g.escalate.sum())}/{len(g)}")
    print("HUMAN GATE: verify data/golden.csv against labeling/golden_sheet.html")


if __name__ == "__main__":
    main()
