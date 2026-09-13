"""Config loading. Paths are derived from the repo root, not configurable."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load((ROOT / "config.yaml").read_text())


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY is not set. Copy .env.example to env.sh, add your key, "
            "then `source env.sh` (or export it inline)."
        )
    return key
