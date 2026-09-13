"""S2b: label the corpus pool (6k random threads) with the final taxonomy,
zero-shot (no retrieval neighbors yet — the index is built FROM these labels).
Output: data/processed/corpus_labeled.parquet — the retrieval corpus, the
simple baseline's training data, and the golden set's sampling pool."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.classify import SYSTEM, load_intents
from hiver_agent.config import CFG, ROOT
from hiver_agent.llm import LLM


def zero_shot(text: str, defs: str, llm: LLM) -> dict:
    out = llm.json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Intent definitions:\n{defs}\n\nNo neighbor tweets available — classify directly.\n\nTweet: {text}"},
    ])
    if out.get("intent") not in load_intents():
        out["intent"] = "other"
    return out


def main():
    sub = pd.read_csv(ROOT / "data/committed/spotify_subsample.csv.gz")
    rng = np.random.default_rng(7)
    n = min(CFG["corpus"]["label_pool"], len(sub))
    pool = sub.iloc[sorted(rng.choice(len(sub), n, replace=False))].reset_index(drop=True)

    defs = "\n".join(f"- {name}: {i['definition']}" for name, i in load_intents().items())
    llm = LLM(CFG["models"]["bulk"], CFG["temperature"]["classify"])

    with ThreadPoolExecutor(16) as ex:
        preds = list(ex.map(lambda t: zero_shot(t, defs, llm), pool.text))

    pool["intent"] = [p["intent"] for p in preds]
    pool["confidence"] = [p.get("confidence", "medium") for p in preds]
    out = ROOT / "data/processed/corpus_labeled.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pool.to_parquet(out)
    print(pool.intent.value_counts())
    print(f"saved {len(pool)} labeled rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
