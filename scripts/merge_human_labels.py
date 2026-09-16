"""Merge the three human-labeled downloads (from ~/Downloads by default) into
the repo. Usage: make merge-labels        (or DL=/path make merge-labels)
Fails loudly if any file is missing rows, so gaps can't slip in silently."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.config import ROOT

DL = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads").expanduser()


def load(name: str, min_rows: int) -> pd.DataFrame:
    p = DL / name
    if not p.exists():
        raise SystemExit(f"missing: {p} — download it from its editor first")
    df = pd.read_csv(p)
    if len(df) < min_rows:
        raise SystemExit(f"{name}: {len(df)} rows, expected >= {min_rows} — "
                         "finish labeling in the editor before downloading")
    return df


def main():
    # 1) golden corrections -> data/golden.csv
    cor = load("golden_corrected.csv", 200)
    golden = pd.read_csv(ROOT / "data/golden.csv")
    cor = cor.rename(columns={"id": "root_tweet_id"}).set_index("root_tweet_id")
    n_flip_i = n_flip_e = 0
    for tid, row in cor.iterrows():
        if tid not in golden.root_tweet_id.values:
            raise SystemExit(f"golden_corrected.csv has unknown tweet id {tid}")
        i = golden.index[golden.root_tweet_id == tid][0]
        if pd.notna(row["intent"]) and str(row["intent"]) != str(golden.at[i, "intent"]):
            n_flip_i += 1
        if pd.notna(row["escalate"]):
            new_e = str(row["escalate"]).strip().lower() in ("true", "1", "yes")
            if new_e != bool(golden.at[i, "escalate"]):
                n_flip_e += 1
        if pd.notna(row["intent"]):
            golden.at[i, "intent"] = row["intent"]
        if pd.notna(row["escalate"]):
            golden.at[i, "escalate"] = str(row["escalate"]).strip().lower() in ("true", "1", "yes")
        golden.at[i, "verified"] = True
    golden["label_process"] = "prelabel+adjudicated+human-verified"
    golden.to_csv(ROOT / "data/golden.csv", index=False)

    # 2) blind relabels -> labeling/relabel_30.csv
    rel = load("relabel_30_filled.csv", 30).rename(columns={"id": "root_tweet_id"})
    base = pd.read_csv(ROOT / "labeling/relabel_30.csv").drop(
        columns=["intent", "escalate"], errors="ignore")
    rel["escalate"] = rel["escalate"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    merged = base.merge(rel[["root_tweet_id", "intent", "escalate"]],
                        on="root_tweet_id", how="left")
    if merged.intent.isna().any():
        raise SystemExit("relabel_30_filled.csv is missing some of the 30 tweets")
    merged.to_csv(ROOT / "labeling/relabel_30.csv", index=False)

    # 3) judge scores -> labeling/judge_sheet.csv
    js = load("judge_sheet_filled.csv", 50).rename(columns={"id": "root_tweet_id"})
    sheet = pd.read_csv(ROOT / "labeling/judge_sheet.csv").drop(
        columns=["human_groundedness", "human_actionability", "human_tone",
                 "human_safety", "human_pass"], errors="ignore")
    js["human_pass"] = js["human_pass"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes"))
    merged_js = sheet.merge(js[["root_tweet_id", "human_groundedness",
                                "human_actionability", "human_tone", "human_safety",
                                "human_pass"]], on="root_tweet_id", how="left")
    if merged_js.human_groundedness.isna().any():
        raise SystemExit("judge_sheet_filled.csv is missing some of the 50 replies")
    merged_js.to_csv(ROOT / "labeling/judge_sheet.csv", index=False)

    print(f"golden.csv: 200 rows human-verified "
          f"({n_flip_i} intent + {n_flip_e} escalate changed from adjudicated values)")
    print("relabel_30.csv: 30 blind re-labels merged")
    print("judge_sheet.csv: 50 human rubric scores merged")
    print("\nNEXT: say 'done' — then `make agreement` + eval regeneration + push")


if __name__ == "__main__":
    main()
