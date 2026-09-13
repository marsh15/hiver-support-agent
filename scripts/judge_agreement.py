"""Judge–human agreement: compare labeling/judge_sheet.csv (human rubric
scores) against results/agent/judged scores on the same rows."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent import metrics as M
from hiver_agent.config import ROOT

DIMS = ["groundedness", "actionability", "tone", "safety"]


def main():
    sheet = pd.read_csv(ROOT / "labeling/judge_sheet.csv")
    sheet = sheet.dropna(subset=[f"human_{d}" for d in DIMS])
    for d in DIMS:
        sheet[f"human_{d}"] = sheet[f"human_{d}"].astype(int)
    preds = pd.read_csv(ROOT / "results/agent/predictions.csv").set_index("root_tweet_id")
    preds = preds.loc[sheet.root_tweet_id]

    out = {"n_scored": int(len(sheet))}
    for d in DIMS:
        h = sheet[f"human_{d}"].tolist()
        j = preds[f"judge_{d}"].tolist()
        out[d] = {
            "exact_agreement": float(np_mean([a == b for a, b in zip(h, j)])),
            "within_one": M.within_one(h, j),
            "kappa": M.kappa(h, j),
            "mean_bias_judge_minus_human": float(np_mean([b - a for a, b in zip(h, j)])),
        }
    hpass = sheet["human_pass"].astype(str).str.lower().isin(["true", "1", "yes"])
    out["pass_fail"] = {
        "agreement": float((hpass == preds.judge_pass).mean()),
        "kappa": M.kappa(hpass.tolist(), preds.judge_pass.tolist()),
    }
    print(json.dumps(out, indent=2))
    (ROOT / "results/judge_agreement.json").write_text(json.dumps(out, indent=2))


def np_mean(xs):
    import numpy as np

    return sum(xs) / len(xs)


if __name__ == "__main__":
    main()
