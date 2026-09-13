"""Cosine-similarity retrieval over the labeled corpus. No vector DB —
numpy over a fp16 matrix is enough at this scale (ponytail: swap for a real
ANN index only if the corpus grows past ~100k)."""
import json

import numpy as np

from .config import ROOT

INDEX_DIR = ROOT / "index"


class Index:
    def __init__(self):
        self.meta = json.loads((INDEX_DIR / "meta.json").read_text())
        self.texts = [r["text"] for r in self.meta]
        self.replies = [r["brand_reply"] for r in self.meta]
        self.intents = [r["intent"] for r in self.meta]
        self.ids = [r["root_tweet_id"] for r in self.meta]
        mat = np.load(INDEX_DIR / "embeddings.npy").astype(np.float32)
        self.matrix = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    def retrieve(self, query_vec: np.ndarray, k: int) -> list[dict]:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        sims = self.matrix @ q
        top = np.argsort(-sims)[:k]
        return [
            {"text": self.texts[i], "brand_reply": self.replies[i],
             "intent": self.intents[i], "root_tweet_id": self.ids[i], "sim": float(sims[i])}
            for i in top
        ]

    def retrieve_one(self, text: str, k: int) -> list[dict]:
        from .llm import embed

        return self.retrieve(embed([text])[0], k)
