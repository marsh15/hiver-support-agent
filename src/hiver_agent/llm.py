"""Cached OpenAI client. Every chat call is content-addressed on disk so
re-runs (and graders) are free; embeddings are NOT cached here (stored as
fp16 .npy by build_index.py instead — JSON would be ~50x larger)."""
import hashlib
import json
import time

from .config import ROOT, api_key

CACHE = ROOT / "cache"
COSTS = ROOT / "costs.jsonl"
RETRYABLE = (429, 500, 502, 503, 504)
PRICE = {  # $ per 1M tokens, for the running cost log only
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "text-embedding-3-small": (0.02, 0.0),
}


class LLM:
    def __init__(self, model: str, temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self._client = None

    def _openai(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key())
        return self._client

    def chat(self, messages: list[dict], json_mode: bool = False) -> str:
        payload = json.dumps(
            [self.model, self.temperature, messages, json_mode], sort_keys=True
        )
        key = hashlib.sha256(payload.encode()).hexdigest()
        path = CACHE / key[:2] / (key + ".json")
        if path.exists():
            return json.loads(path.read_text())["response"]

        kwargs = dict(model=self.model, temperature=self.temperature, messages=messages)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(5):
            try:
                resp = self._openai().chat.completions.create(**kwargs)
                break
            except Exception as e:  # rate limit / transient server errors
                status = getattr(e, "status_code", None)
                if status not in RETRYABLE and attempt < 4:
                    raise
                time.sleep(2**attempt + 1)
        text = resp.choices[0].message.content
        usage = {
            "model": self.model,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "cached": False,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"response": text, "usage": usage}))
        with COSTS.open("a") as f:
            f.write(json.dumps(usage) + "\n")
        return text

    def json(self, messages: list[dict]) -> dict:
        raw = self.chat(messages, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0:
                return json.loads(raw[start : end + 1])
            raise


def embed(texts: list[str], model: str | None = None) -> "numpy.ndarray":
    """Embeddings bypass the JSON cache (too large); callers persist results."""
    import numpy as np

    from .config import CFG

    model = model or CFG["models"]["embedding"]
    out = []
    client = None
    for i in range(0, len(texts), 512):
        for attempt in range(5):
            try:
                if client is None:
                    from openai import OpenAI

                    client = OpenAI(api_key=api_key())
                resp = client.embeddings.create(model=model, input=texts[i : i + 512])
                break
            except Exception:
                time.sleep(2**attempt + 1)
        out.extend(d.embedding for d in resp.data)
        with COSTS.open("a") as f:
            f.write(
                json.dumps({"model": model, "prompt_tokens": len(texts[i : i + 512]) * 4, "cached": False})
                + "\n"
            )
    return np.asarray(out, dtype=np.float32)
