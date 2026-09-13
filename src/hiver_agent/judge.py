"""LLM-as-judge for reply quality. Judge sees the playbook and the same
exemplars the drafter could use, so it can actually verify groundedness.
Validated against the human scores in labeling/judge_sheet.csv."""
from .config import CFG
from .llm import LLM

RUBRIC = """Score each dimension 1-5:
- groundedness: reply only claims/asks things Spotify historically says for this issue
- actionability: reply moves the customer toward resolution
- tone: matches Spotify Cares' concise, friendly, human voice
- safety: no public PII requests beyond DM, no refund promises, no speculation"""

SYSTEM = f"""You are grading an auto-drafted Spotify support reply against how Spotify
historically handled similar issues. {RUBRIC}
Answer JSON: {{"groundedness": n, "actionability": n, "tone": n, "safety": n,
"pass": true|false, "rationale": "<=25 words"}}. pass = all four dimensions >= 4."""


def judge(text: str, reply: str, intent: str, playbook: str,
          exemplars: list[dict], llm: LLM | None = None) -> dict:
    llm = llm or LLM(CFG["models"]["bulk"], CFG["temperature"]["judge"])
    shots = "\n".join(
        f"- customer: {e['text'][:150]} | spotify replied: {e['brand_reply'][:200]}"
        for e in exemplars
    ) or "(none)"
    out = llm.json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Playbook for '{intent}':\n{playbook}\n\nHistorical exemplars:\n{shots}\n\nCustomer tweet: {text}\nDrafted reply: {reply}"},
    ])
    for k in ("groundedness", "actionability", "tone", "safety"):
        out[k] = int(out.get(k, 0))
    out["pass"] = all(1 <= out[k] <= 5 and out[k] >= 4 for k in ("groundedness", "actionability", "tone", "safety"))
    return out
