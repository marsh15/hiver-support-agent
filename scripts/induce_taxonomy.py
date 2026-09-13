"""S2: induce an intent taxonomy from the data. Samples 2k tweets, asks the
bulk model to propose/assign categories in batches, aggregates proposals, and
prints a consolidation report. Human edits configs/intents.yaml afterwards;
the seed taxonomy already reflects this data's EDA."""
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.config import CFG, ROOT
from hiver_agent.llm import LLM

SYSTEM = """You are designing an intent taxonomy for Spotify customer support tweets.
Given tweets, propose the small set (6-10) of distinct customer intents you see,
with a one-sentence definition each. Use these candidate names where they fit:
account_access, billing_charges, subscription_management, app_technical,
content_availability, device_integration, feature_howto, account_security,
praise_feedback, other. Answer JSON:
{"intents": [{"name": "...", "definition": "...", "example_tweet": "..."}]}"""


def main():
    sub = pd.read_csv(ROOT / "data/committed/spotify_subsample.csv.gz")
    rng = np.random.default_rng(42)
    sample = sub.iloc[rng.choice(len(sub), CFG["corpus"]["induction_sample"], replace=False)]

    llm = LLM(CFG["models"]["bulk"], 0.0)
    batches = [sample.text.iloc[i:i + 50].tolist() for i in range(0, len(sample), 50)]

    def run(batch):
        return llm.json([{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "\n".join(batch)}])

    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(run, batches))

    names, defs = Counter(), {}
    for r in results:
        for intent in r.get("intents", []):
            n = str(intent.get("name", "")).strip().lower().replace(" ", "_")
            if n:
                names[n] += 1
                defs.setdefault(n, intent.get("definition", ""))
    report = {
        "n_batches": len(batches),
        "proposed_intents": [
            {"name": n, "proposed_in_batches": c, "merged_definition": defs[n]}
            for n, c in names.most_common()
        ],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/taxonomy_induction.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nNEXT: hand-consolidate into configs/intents.yaml (names + definitions)")


if __name__ == "__main__":
    main()
