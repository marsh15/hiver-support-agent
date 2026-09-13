"""Few-shot intent classification: intent definitions + retrieved labeled
neighbors in the prompt; the LLM returns an intent and self-assessed
confidence (cross-checked against neighbor agreement by the caller)."""
import yaml

from .config import CFG, ROOT
from .llm import LLM

INTENTS_PATH = ROOT / "configs" / "intents.yaml"


def load_intents() -> dict:
    data = yaml.safe_load(INTENTS_PATH.read_text())
    return {i["name"]: i for i in data["intents"]}


SYSTEM = """You classify customer support tweets to Spotify into exactly one intent.
Answer with JSON: {"intent": "<name>", "confidence": "high|medium|low", "reason": "<=15 words"}.
Use "other" only when nothing fits. Confidence low = ambiguous, multi-issue, or sarcasm."""


def classify(text: str, index, llm: LLM | None = None) -> dict:
    llm = llm or LLM(CFG["models"]["bulk"], CFG["temperature"]["classify"])
    intents = load_intents()
    defs = "\n".join(f"- {n}: {i['definition']}" for n, i in intents.items())
    neighbors = index.retrieve_one(text, CFG["retrieval"]["k_neighbors"])
    shots = "\n".join(
        f"[{n['intent']}] {n['text'][:180]}" for n in neighbors
    )
    out = llm.json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Intent definitions:\n{defs}\n\nSimilar labeled tweets:\n{shots}\n\nTweet: {text}"},
    ])
    neighbor_votes = sum(1 for n in neighbors if n["intent"] == out.get("intent"))
    out["neighbor_agreement"] = neighbor_votes / max(len(neighbors), 1)
    if out.get("intent") not in intents:
        out["intent"] = "other"
    return out
