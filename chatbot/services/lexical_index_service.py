from __future__ import annotations

import time
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from django.conf import settings


_CACHE_BY_SCOPE: Dict[str, Dict[str, Any]] = {}


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


def _scope_cache_key(
    *,
    company_id: Optional[int] = None,
    bot_id: Optional[int] = None,
    doc_ids: Optional[Sequence[int]] = None,
) -> str:
    doc_part = ",".join(str(int(v)) for v in sorted(set(doc_ids or [])))
    return f"company:{company_id}|bot:{bot_id}|docs:{doc_part}"


def get_lexical_rows(
    *,
    qdrant: Any,
    company_id: Optional[int] = None,
    bot_id: Optional[int] = None,
    doc_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Cached payload snapshot for lexical scoring to avoid full rescans every query.
    """
    scope_key = _scope_cache_key(company_id=company_id, bot_id=bot_id, doc_ids=doc_ids)
    cache = _CACHE_BY_SCOPE.setdefault(
        scope_key,
        {"built_at": 0.0, "rows": [], "postings": {}, "doc_len": [], "avg_doc_len": 0.0},
    )
    ttl = max(1, int(getattr(settings, "RAG_LEXICAL_INDEX_TTL_SEC", 300)))
    now = time.time()
    if cache["rows"] and (now - float(cache["built_at"])) <= ttl:
        return list(cache["rows"])
    scan_fn = getattr(qdrant, "scan_payload_points", None)
    if not callable(scan_fn):
        return []
    per_page = int(getattr(settings, "RAG_LEXICAL_SCAN_LIMIT", 150))
    max_points = int(getattr(settings, "RAG_LEXICAL_MAX_POINTS", 1200))
    try:
        scanned = scan_fn(
            limit=per_page,
            max_points=max_points,
            company_id=company_id,
            bot_id=bot_id,
            doc_ids=doc_ids,
        )
    except Exception:
        return []
    rows: List[Dict[str, Any]] = []
    for p in scanned or []:
        payload = p.payload if hasattr(p, "payload") else p.get("payload")
        if not payload or not payload.get("text"):
            continue
        rows.append(payload)
    postings, doc_len, avg_doc_len = _build_sparse_index(rows)
    cache["rows"] = rows
    cache["postings"] = postings
    cache["doc_len"] = doc_len
    cache["avg_doc_len"] = avg_doc_len
    cache["built_at"] = now
    return list(rows)


def sparse_lexical_candidates(
    *,
    qdrant: Any,
    query_terms: List[str],
    top_n: int,
    company_id: Optional[int] = None,
    bot_id: Optional[int] = None,
    doc_ids: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Inverted-index BM25 retrieval over cached payloads.
    """
    if not query_terms:
        return []
    rows = get_lexical_rows(qdrant=qdrant, company_id=company_id, bot_id=bot_id, doc_ids=doc_ids)
    if not rows:
        return []
    scope_key = _scope_cache_key(company_id=company_id, bot_id=bot_id, doc_ids=doc_ids)
    cache = _CACHE_BY_SCOPE.get(scope_key) or {}
    postings = cache.get("postings") or {}
    doc_len: List[int] = cache.get("doc_len") or []
    avg_len = float(cache.get("avg_doc_len") or 1.0)
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

