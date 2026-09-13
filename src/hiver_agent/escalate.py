"""Escalation: LLM proposes auto/escalate with a reason; deterministic
guardrails can only FORCE escalation (never force auto-handling)."""
import re

from .config import CFG
from .llm import LLM

ANGER_RE = re.compile(
    r"\b(lawsuit|sue|lawyer|lawyers|furious|disgusting|unacceptable|ridiculous|"
    r"cancel(l)?(ed)? my (account|subscription)|worst (company|service|app)|"
    r"report you|bbb|fraud|scam|charge ?back| RipOff|damn|wtf|pos)\b", re.I)

SYSTEM = """You decide whether Spotify's support agent can safely AUTO-REPLY to this tweet
or must ESCALATE to a human. Escalate when: account security/billing dispute with money at
stake, legal threats, PR risk, the drafted reply doesn't actually resolve the issue, or the
customer is already in a repeated-failure loop. Auto-reply when the drafted reply fully
addresses a routine issue. Answer JSON: {"escalate": true|false, "reason": "<=20 words"}."""


def decide(text: str, intent: str, confidence: str, reply: str,
           llm: LLM | None = None) -> dict:
    llm = llm or LLM(CFG["models"]["bulk"], CFG["temperature"]["judge"])
    proposal = llm.json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Tweet: {text}\nIntent: {intent} (confidence {confidence})\nDrafted reply: {reply}"},
    ])
    escalate = bool(proposal.get("escalate"))
    reason = str(proposal.get("reason", "LLM proposal"))

    always = load_intents_escalating()
    if intent in always:
        escalate, reason = True, f"guardrail: intent '{intent}' is always escalated"
    if confidence == "low":
        escalate, reason = True, "guardrail: low classifier confidence"
    if ANGER_RE.search(text):
        escalate, reason = True, "guardrail: anger/legal keyword in message"
    if not reply.strip():
        escalate, reason = True, "guardrail: drafter returned empty reply"
    return {"escalate": escalate, "reason": reason, "llm_proposal": proposal.get("reason", "")}


def load_intents_escalating() -> set[str]:
    from .classify import load_intents

    return {n for n, i in load_intents().items() if i.get("always_escalate")}
