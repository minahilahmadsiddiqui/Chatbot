from __future__ import annotations

import time
import math
import re
from typing import Any, Dict, List, Tuple

from django.conf import settings


_CACHE: Dict[str, Any] = {
    "built_at": 0.0,
    "rows": [],
    "postings": {},
    "doc_len": [],
    "avg_doc_len": 0.0,
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text or "") if len(t) >= 2]


def _build_sparse_index(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[int, int]], List[int], float]:
    postings: Dict[str, Dict[int, int]] = {}
    doc_len: List[int] = []
    for idx, payload in enumerate(rows):
        text = str(payload.get("text") or "")
        sec = str(payload.get("source_section") or "")
        toks = _tokenize(f"{sec} {text}")
        doc_len.append(len(toks))
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for term, freq in tf.items():
            slot = postings.setdefault(term, {})
            slot[idx] = freq
    avg_doc_len = (sum(doc_len) / max(1, len(doc_len))) if doc_len else 0.0
    return postings, doc_len, avg_doc_len


def get_lexical_rows(*, qdrant: Any) -> List[Dict[str, Any]]:
    """
    Cached payload snapshot for lexical scoring to avoid full rescans every query.
    """
    ttl = max(1, int(getattr(settings, "RAG_LEXICAL_INDEX_TTL_SEC", 300)))
    now = time.time()
    if _CACHE["rows"] and (now - float(_CACHE["built_at"])) <= ttl:
        return list(_CACHE["rows"])
    scan_fn = getattr(qdrant, "scan_payload_points", None)
    if not callable(scan_fn):
        return []
    per_page = int(getattr(settings, "RAG_LEXICAL_SCAN_LIMIT", 150))
    max_points = int(getattr(settings, "RAG_LEXICAL_MAX_POINTS", 1200))
    try:
        scanned = scan_fn(limit=per_page, max_points=max_points)
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for p in scanned or []:
        payload = p.payload if hasattr(p, "payload") else p.get("payload")
        if not payload or not payload.get("text"):
            continue
        rows.append(payload)
    postings, doc_len, avg_doc_len = _build_sparse_index(rows)
    _CACHE["rows"] = rows
    _CACHE["postings"] = postings
    _CACHE["doc_len"] = doc_len
    _CACHE["avg_doc_len"] = avg_doc_len
    _CACHE["built_at"] = now
    return list(rows)


def sparse_lexical_candidates(*, qdrant: Any, query_terms: List[str], top_n: int) -> List[Dict[str, Any]]:
    """
    Inverted-index BM25 retrieval over cached payloads.
    """
    if not query_terms:
        return []
    rows = get_lexical_rows(qdrant=qdrant)
    if not rows:
        return []
    postings = _CACHE.get("postings") or {}
    doc_len: List[int] = _CACHE.get("doc_len") or []
    avg_len = float(_CACHE.get("avg_doc_len") or 1.0)
    n_docs = max(1, len(rows))
    k1 = 1.2
    b = 0.75
    scores: Dict[int, float] = {}
    overlap: Dict[int, int] = {}
    for term in query_terms:
        plist: Dict[int, int] = postings.get(term, {})
        if not plist:
            continue
        df = len(plist)
        idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
        for doc_id, tf in plist.items():
            dl = doc_len[doc_id] if doc_id < len(doc_len) else 1
            denom = tf + k1 * (1.0 - b + b * (float(dl) / max(1.0, avg_len)))
            bm25 = idf * ((tf * (k1 + 1.0)) / max(1e-9, denom))
            scores[doc_id] = scores.get(doc_id, 0.0) + bm25
            overlap[doc_id] = overlap.get(doc_id, 0) + 1
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, int(top_n))]
    out: List[Dict[str, Any]] = []
    for doc_id, sc in ranked:
        if doc_id >= len(rows):
            continue
        out.append(
            {
                "score": None,
                "payload": rows[doc_id],
                "_lexical": True,
                "_bm25": float(sc),
                "_overlap": int(overlap.get(doc_id, 0)),
            }
        )
    return out

