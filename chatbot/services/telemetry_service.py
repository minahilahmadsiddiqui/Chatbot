from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from django.conf import settings


def _telemetry_path() -> str:
    base = os.path.join(getattr(settings, "BASE_DIR", "."), "logs")
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "rag_telemetry.jsonl")


def append_rag_telemetry(event: Dict[str, Any]) -> None:
    if not bool(getattr(settings, "RAG_TELEMETRY_ENABLED", True)):
        return
    payload = {"ts": int(time.time() * 1000), **(event or {})}
    try:
        with open(_telemetry_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        return

