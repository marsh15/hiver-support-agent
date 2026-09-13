"""Baselines: trivial (majority class + canned reply + always escalate) and
simple (TF-IDF logreg intent + per-intent historical boilerplate + keyword
escalation). The no-retrieval ablation is just Agent(use_retrieval=False)."""
import re

from .escalate import ANGER_RE
from .llm import LLM


class TrivialAgent:
    intent = None  # set by fit()
    reply = "We hear you and we'd love to help. Please send us a DM and we'll take a look right away."

    def fit(self, corpus_labels: list[str]):
        self.intent = max(set(corpus_labels), key=corpus_labels.count)

    def handle(self, text: str, exclude_ids: set | None = None) -> dict:
        return {"intent": self.intent, "confidence": "n/a", "neighbor_agreement": 0.0,
                "reply": self.reply, "grounding_thread_ids": [],
                "reply_violations": [], "escalate": True,
                "escalate_reason": "trivial baseline: always escalates"}


class SimpleAgent:
    def fit(self, corpus_texts: list[str], corpus_labels: list[str],
            corpus_replies: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
        X = self.vec.fit_transform(corpus_texts)
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        self.clf.fit(X, corpus_labels)
        self.proba = hasattr(self.clf, "predict_proba")
        # per-intent most common historical brand reply = the "template"
        by_intent = {}
        for lab, rep in zip(corpus_labels, corpus_replies):
            by_intent.setdefault(lab, []).append(rep)
        self.templates = {lab: max(set(reps), key=reps.count) for lab, reps in by_intent.items()}

    def handle(self, text: str, exclude_ids: set | None = None) -> dict:
        import numpy as np

        X = self.vec.transform([text])
        pred = self.clf.predict(X)[0]
        margin = float(np.max(self.clf.predict_proba(X))) if self.proba else 0.5
        reply = self.templates.get(pred, TrivialAgent.reply)
        low = margin < 0.25
        angry = bool(ANGER_RE.search(text))
        esc = low or angry
        why = ("keyword: anger/legal" if angry else "low TF-IDF margin") if esc else "routine intent, confident"
        return {"intent": pred, "confidence": "low" if low else "medium",
                "neighbor_agreement": margin, "reply": reply, "grounding_thread_ids": [],
                "reply_violations": [], "escalate": esc, "escalate_reason": why}
