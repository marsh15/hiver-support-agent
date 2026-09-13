"""Extract SpotifyCares threads from twcs.csv.

A "thread record" = one first-inbound customer tweet (the agent's input) plus
the brand's earliest reply (the historical resolution signal) and a short
thread head for labeling context. Outputs:
  data/processed/spotify_threads.parquet   (full slice, gitignored)
  data/committed/spotify_subsample.csv.gz  (~30k records, committed)
  results/eda.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.config import ROOT

BRAND = "SpotifyCares"
RAW = ROOT / "data/raw/twcs/twcs.csv"
DM_RE = re.compile(r"\b(dms?|direct messages?)\b", re.I)
HEAD_LEN = 6


def strip_mention(t: str) -> str:
    return re.sub(r"^@\w+\s+", "", str(t)).strip()


def thread_head(root_id: int, root_text: str, reply: str, kids, by_id) -> str:
    head = [f"customer: {root_text}", f"spotify: {reply}"]
    for c in str(kids.get(root_id, "") or "").split(","):
        c = c.strip()
        if not c or len(head) >= HEAD_LEN:
            continue
        try:
            row = by_id.loc[int(c)]
        except (ValueError, KeyError):
            continue
        who = "customer" if bool(row.inbound) else "spotify"
        head.append(f"{who}: {strip_mention(row.text)[:200]}")
    return "\n".join(head)


def find_root(tid: int, parent: dict, root_of: dict) -> int:
    chain = []
    cur = tid
    while cur not in root_of and cur in parent:
        chain.append(cur)
        cur = int(parent[cur])
    r = root_of.get(cur, cur)
    for t in chain:
        root_of[t] = r
    return r


def thread_records() -> pd.DataFrame:
    df = pd.read_csv(
        RAW,
        dtype={"author_id": str, "response_tweet_id": str, "text": str},
        usecols=["tweet_id", "author_id", "inbound", "created_at", "text",
                 "response_tweet_id", "in_response_to_tweet_id"],
    )
    parent = {int(t): p for t, p in zip(df.tweet_id, df.in_response_to_tweet_id)
              if pd.notna(p)}
    kids = df.set_index("tweet_id")["response_tweet_id"]
    by_id = df.set_index("tweet_id")[["text", "inbound"]]

    brand = df[df.author_id == BRAND].copy()
    brand["created"] = pd.to_datetime(brand.created_at, format="%a %b %d %H:%M:%S %z %Y")
    root_of: dict = {}
    brand["root"] = [find_root(int(t), parent, root_of) for t in brand.tweet_id]
    first = (brand.sort_values(["root", "created"])
                  .groupby("root", as_index=False).first())

    roots = df[df.inbound & df.tweet_id.isin(first.root)].set_index("tweet_id")
    first = first[first.root.isin(roots.index)]

    rows, counts = [], first.root.value_counts()
    for r in first.itertuples():
        root = roots.loc[r.root]
        text = strip_mention(root.text)
        if len(text) < 15:  # pure mentions/emoji/media-only
            continue
        rows.append({
            "root_tweet_id": int(r.root),
            "created_at": root.created_at,
            "text": text,
            "brand_reply": strip_mention(r.text),
            "brand_reply_id": int(r.tweet_id),
            "dm_redirect": bool(DM_RE.search(r.text)),
            "n_brand_replies": int(counts[r.root]),
            "thread_head": thread_head(int(r.root), text, strip_mention(r.text), kids, by_id),
        })
    out = pd.DataFrame(rows)
    out["_norm"] = out.text.str.lower().str.replace(r"\W", "", regex=True)
    out = out.drop_duplicates(subset="_norm").drop(columns="_norm")
    return out.sort_values("created_at").reset_index(drop=True)


def main():
    recs = thread_records()
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/committed").mkdir(parents=True, exist_ok=True)
    recs.to_parquet(ROOT / "data/processed/spotify_threads.parquet")

    rng = np.random.default_rng(42)
    commit_n = min(30_000, len(recs))
    sub = recs.iloc[sorted(rng.choice(len(recs), commit_n, replace=False))]
    sub.to_csv(ROOT / "data/committed/spotify_subsample.csv.gz", index=False)

    eda = {
        "brand": BRAND,
        "n_threads_total": int(len(recs)),
        "n_committed": int(commit_n),
        "date_range": [str(recs.created_at.min()), str(recs.created_at.max())],
        "dm_redirect_rate": round(float(recs.dm_redirect.mean()), 4),
        "median_text_len": int(recs.text.str.len().median()),
        "median_reply_len": int(recs.brand_reply.str.len().median()),
        "median_brand_replies_per_thread": int(recs.n_brand_replies.median()),
        "sample_customer_tweets": recs.text.head(200).tolist(),
        "sample_brand_replies": recs.brand_reply.head(200).tolist(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/eda.json").write_text(json.dumps(eda, indent=2))
    print(json.dumps({k: v for k, v in eda.items() if not k.startswith("sample")}, indent=2))


if __name__ == "__main__":
    main()
