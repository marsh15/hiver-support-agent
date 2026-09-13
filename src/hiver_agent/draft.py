"""Reply drafter: per-intent playbook + retrieved historical exemplars →
a <=280-char, link-free, mention-free reply with cited grounding."""
import re

import yaml

from .config import CFG, ROOT
from .llm import LLM

PLAYBOOKS_PATH = ROOT / "configs" / "playbooks.yaml"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
EMAIL_RE = re.compile(r"\S+@\S+\.\S+")
PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")

SYSTEM = """You draft the public first reply a Spotify Cares agent sends on Twitter.
Rules: <=280 characters, no links, no @mentions, no email/phone, no promises of refunds,
never ask for password/payment details in public. Mirror the tone and moves of the real
historical replies shown. Answer JSON: {"reply": "...", "grounding_thread_ids": [ids of the
exemplars you actually borrowed moves from]}."""


def load_playbooks() -> dict:
    return yaml.safe_load(PLAYBOOKS_PATH.read_text())["playbooks"]


def check_reply(reply: str) -> list[str]:
    """Deterministic reply-constraint check; returns violation labels."""
    v = []
    if len(reply) > 280:
        v.append("over_length")
    if URL_RE.search(reply):
        v.append("link")
    if MENTION_RE.search(reply):
        v.append("mention")
    if EMAIL_RE.search(reply) or PHONE_RE.search(reply):
        v.append("pii")
    return v


def clean(reply: str) -> str:
    reply = URL_RE.sub("", reply)
    reply = MENTION_RE.sub("", reply)
    reply = re.sub(r"\s+", " ", reply).strip()
    if len(reply) > 280:  # ponytail: hard truncate at last sentence end, else 280
        cut = max(reply.rfind(". ", 0, 278), reply.rfind("! ", 0, 278))
        reply = reply[: cut + 1] if cut > 120 else reply[:277].rsplit(" ", 1)[0] + "…"
    return reply


def draft(text: str, intent: str, index, llm: LLM | None = None,
          use_retrieval: bool = True, exclude_ids: set | None = None) -> dict:
    llm = llm or LLM(CFG["models"]["strong"], CFG["temperature"]["draft"])
    playbooks = load_playbooks()
    playbook = playbooks.get(intent, {}).get("playbook", "(no playbook — use generic empathetic triage and ask for details)")
    exemplars = (index.retrieve_one(text, CFG["retrieval"]["k_draft_exemplars"], exclude_ids)
                 if use_retrieval else [])
    shots = "\n".join(
        f"[thread {e['root_tweet_id']}]\ncustomer: {e['text'][:180]}\nspotify replied: {e['brand_reply'][:280]}"
        for e in exemplars
    ) or "(no exemplars — rely on the playbook)"
    out = llm.json([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Playbook for intent '{intent}':\n{playbook}\n\nReal historical resolutions:\n{shots}\n\nCustomer tweet: {text}"},
    ])
    reply = clean(out.get("reply", ""))
    return {"reply": reply, "grounding_thread_ids": out.get("grounding_thread_ids", []) or [e["root_tweet_id"] for e in exemplars[:2]], "violations": check_reply(reply)}
