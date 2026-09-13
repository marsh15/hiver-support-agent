"""Labeler self-agreement: compare labeling/relabel_30.csv (blind re-labels)
against data/golden.csv; reports raw agreement + Cohen's kappa."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent import metrics as M
from hiver_agent.config import ROOT


def main():
    relabel = pd.read_csv(ROOT / "labeling/relabel_30.csv")
    golden = pd.read_csv(ROOT / "data/golden.csv").set_index("root_tweet_id")
    relabel = relabel.dropna(subset=["intent"])
    relabel["escalate"] = relabel["escalate"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes"))

    truth = golden.loc[relabel.root_tweet_id]
    out = {
        "n_relabelled": int(len(relabel)),
        "intent_agreement": float((relabel.intent.values == truth.intent.values).mean()),
        "intent_kappa": M.kappa(truth.intent.tolist(), relabel.intent.tolist()),
        "escalate_agreement": float(
            (relabel.escalate.values == truth.escalate.astype(bool).values).mean()),
        "escalate_kappa": M.kappa(truth.escalate.astype(bool).tolist(), relabel.escalate.tolist()),
    }
    print(out)
    (ROOT / "results/self_agreement.json").write_text(
        __import__("json").dumps(out, indent=2))


if __name__ == "__main__":
    main()
