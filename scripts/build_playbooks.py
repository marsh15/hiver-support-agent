"""S5: distill per-intent playbooks from historical brand replies (<=200 per
intent sampled from the labeled corpus) -> configs/playbooks.yaml."""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hiver_agent.classify import load_intents
from hiver_agent.config import CFG, ROOT
from hiver_agent.llm import LLM

SYSTEM = """You are writing the internal playbook a Spotify Cares agent should follow when
replying publicly to a tweet of the given intent, based ONLY on the real historical replies
shown. 5-8 lines: tone, what to ask for, what to offer, what to never do in public.
Answer JSON: {"playbook": "...", "n_replies_used": n}"""


def main():
    pool = pd.read_parquet(ROOT / "data/processed/corpus_labeled.parquet")
    llm = LLM(CFG["models"]["bulk"], 0.0)
    intents = [n for n in load_intents() if n != "other"]

    def build(intent: str) -> tuple[str, dict]:
        replies = pool[pool.intent == intent].brand_reply.head(200).tolist()
        if not replies:
            return intent, {"playbook": "(no historical replies for this intent)", "supporting_thread_ids": []}
        out = llm.json([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Intent: {intent}\n\nHistorical replies:\n" + "\n".join(replies)},
        ])
        ids = pool[pool.intent == intent].root_tweet_id.head(50).astype(int).tolist()
        return intent, {"playbook": out.get("playbook", ""), "supporting_thread_ids": ids}

    with ThreadPoolExecutor(8) as ex:
        playbooks = dict(ex.map(build, intents))
    out = {"playbooks": playbooks}
    (ROOT / "configs/playbooks.yaml").write_text(yaml.safe_dump(out, sort_keys=True, width=100))
    print(f"wrote playbooks for {len(playbooks)} intents")


if __name__ == "__main__":
    main()
