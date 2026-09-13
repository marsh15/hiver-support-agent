"""S4: build the retrieval index over the labeled corpus — embed all pool
texts once, store fp16 matrix + meta.json. Everything downstream retrieves
labeled (customer msg, brand reply) pairs from here."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.config import ROOT
from hiver_agent.llm import embed


def main(force: bool = False):
    idx = ROOT / "index"
    emb_path, meta_path = idx / "embeddings.npy", idx / "meta.json"
    if emb_path.exists() and meta_path.exists() and not force:
        print("index exists (use --force to rebuild)")
        return
    pool = pd.read_parquet(ROOT / "data/processed/corpus_labeled.parquet")
    mat = embed(pool.text.tolist()).astype(np.float16)
    idx.mkdir(exist_ok=True)
    np.save(emb_path, mat)
    meta_path.write_text(json.dumps([
        {"text": r.text, "brand_reply": r.brand_reply, "intent": r.intent,
         "root_tweet_id": int(r.root_tweet_id)}
        for r in pool.itertuples()
    ]))
    print(f"index: {len(pool)} vectors, dim {mat.shape[1]} -> {idx.relative_to(ROOT)}")


if __name__ == "__main__":
    main("--force" in sys.argv)
