"""The agent: one customer message in, {intent, reply, escalate} out."""
from . import classify, draft, escalate
from .llm import LLM
from .retrieval import Index


class Agent:
    def __init__(self, use_retrieval: bool = True):
        self.index = Index()
        self.use_retrieval = use_retrieval

    def handle(self, text: str) -> dict:
        cls = classify.classify(text, self.index)
        dr = draft.draft(text, cls["intent"], self.index, use_retrieval=self.use_retrieval)
        esc = escalate.decide(text, cls["intent"], cls.get("confidence", "medium"), dr["reply"])
        return {
            "text": text,
            "intent": cls["intent"],
            "confidence": cls.get("confidence", "medium"),
            "neighbor_agreement": cls.get("neighbor_agreement", 0.0),
            "reply": dr["reply"],
            "grounding_thread_ids": dr["grounding_thread_ids"],
            "reply_violations": dr["violations"],
            "escalate": esc["escalate"],
            "escalate_reason": esc["reason"],
        }
