from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

from django.conf import settings

_CROSS_ENCODER = None


def _get_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is not None:
        return _CROSS_ENCODER
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        _CROSS_ENCODER = None
        return None
    model_name = str(getattr(settings, "RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    _CROSS_ENCODER = CrossEncoder(model_name)
    return _CROSS_ENCODER


def rerank_with_cross_encoder(*, query: str, items: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
    """
    Optional learned reranker. Returns input items if model/dependency unavailable.
    """
    if not items:
        return items
    ce = _get_cross_encoder()
    if ce is None:
        return items
    limit = max(1, min(int(top_n), len(items)))
    head = items[:limit]
    pairs = []
    for it in head:
        payload = it.get("payload") or {}
        text = str(payload.get("text") or "")
        section = str(payload.get("source_section") or "")
        pairs.append((query, f"{section}\n{text}".strip()))
    try:
        scores = ce.predict(pairs)
    except Exception:
        return items
    scored = []
    for it, s in zip(head, scores):
        clone = dict(it)
        try:
            clone["_ce_score"] = float(s)
        except Exception:
            clone["_ce_score"] = 0.0
        scored.append(clone)
    scored.sort(key=lambda x: float(x.get("_ce_score", 0.0)), reverse=True)
    return scored + items[limit:]


def cross_encoder_healthcheck() -> Dict[str, Any]:
    t0 = time.time()
    ce = _get_cross_encoder()
    if ce is None:
        return {"ready": False, "error": "sentence-transformers/CrossEncoder unavailable"}
    try:
        probe: List[Tuple[str, str]] = [
            ("how many hajj leaves are allowed", "8.10 Hajj\nEmployees are entitled to fifteen paid leaves."),
            ("gym reimbursement policy", "Gym reimbursement policy and annual cap details."),
        ]
        scores = ce.predict(probe)
        elapsed_ms = int((time.time() - t0) * 1000)
        return {
            "ready": True,
            "model": str(getattr(settings, "RAG_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")),
            "elapsed_ms": elapsed_ms,
            "sample_scores": [float(s) for s in list(scores)[:2]],
        }
    except Exception as e:
        return {"ready": False, "error": str(e)}

