from __future__ import annotations

import logging
import time
import re
import math
import difflib
from typing import Any, Dict, List, Optional, TypedDict

from django.conf import settings

from chatbot.services.embeddings_service import get_embeddings
from chatbot.services.gemini_service import (
    UNKNOWN_POLICY_PHRASE,
    append_raw_answer_before_ui_processing,
    generate_answer,
)
from chatbot.services.qdrant_service import QdrantService
from chatbot.services.text_splitter import count_tokens, line_looks_like_toc_leader, truncate_text_to_token_budget

logger = logging.getLogger(__name__)

FALLBACK_PHRASE = "Contact the HR department"
_HANDBOOK_ASSISTANT_GREETING = (
    "Hello! I can help you find information in the employee handbook. What would you like to know?"
)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "yours",
    "u",
    # Common intent/meta words that shouldn't be used for grounding checks.
    "explain",
    "describe",
    "tell",
    "summarize",
    "summarise",
    "show",
    "give",
    "provide",
    "many",
    "much",
}
# Strip only meta / filler words from anchors — do NOT strip handbook vocabulary
# like policy, employee, work, or company names (those are real topics).
_GENERIC_QUERY_TERMS = {
    "explain",
    "describe",
    "tell",
    "summarize",
    "summarise",
    "show",
    "give",
    "provide",
    "please",
    "kindly",
    "something",
    "anything",
    "everything",
    "just",
    "also",
    "really",
    "very",
}
# Terms that match too many unrelated handbook lines if used alone for strict extraction.
# Do not add policy nouns that appear in real answers (e.g. "entitled", "leave").
_WEAK_FOCUS_TERMS = frozenset(
    {
        "allowed",
        "allow",
        "allows",
        "use",
        "using",
        "used",
        "personal",
        "phone",
        "phones",
        "cell",
        "limited",
        "please",
        "make",
        "sure",
        "however",
        "also",
        "same",
        "other",
        "any",
        "some",
        "such",
        "salary",
        "salaries",
        "people",
        "efficiency",
        "efficient",
        "proportional",
        "company",
        "companies",
        "work",
        "working",
        "workplace",
    }
)
_TYPO_CANONICAL_TERMS = {
    # Greeting/conversation
    "how",
    "are",
    "you",
    "hello",
    "hi",
    "hey",
    "fine",
    "good",
    "well",
    # HR handbook domain
    "policy",
    "policies",
    "rule",
    "rules",
    "employee",
    "employees",
    "handbook",
    "reimbursement",
    "gym",
    "benefits",
    "leave",
    "attendance",
    "salary",
    "payment",
    "payments",
    "overtime",
    "confidentiality",
    "training",
    "certification",
    "travel",
    "allowance",
    "medical",
}


def _pipeline_fields(*, rag_retrieval_ran: bool, pipeline: str) -> Dict[str, Any]:
    """Diagnostics: whether embeddings/Qdrant ran; which code path produced the answer."""
    return {"rag_retrieval_ran": rag_retrieval_ran, "pipeline": pipeline}


class RagState(TypedDict, total=False):
    query: str
    query_embedding: List[float]
    retrieved: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    answer: str
    fallback_used: bool
    latency_ms: int


def _sanitize_query(query: str) -> str:
    q = (query or "").strip()
    # Keep queries reasonably sized for embedding and model context.
    if len(q) > 4000:
        q = q[:4000]
    return _normalize_query_typos(q)


def _normalize_query_typos(query: str) -> str:
    """
    Lightweight fuzzy typo normalization for user queries.
    Keeps punctuation/spacing mostly intact and only fixes alphabetic tokens
    that are very close to known conversational/HR terms.
    """
    if not query:
        return query

    # Explicit typo fixes we rely on in unit tests and common HR queries.
    # (We keep this small and deterministic; everything else uses fuzzy matching.)
    common_typo_map = {
        "reimbursment": "reimbursement",
        "polciy": "policy",
        "polciies": "policies",
        "employes": "employees",
    }

    def fix_token(tok: str) -> str:
        lower = tok.lower()
        # Avoid over-correcting short tokens and already-known words.
        if len(lower) < 4 or lower in _TYPO_CANONICAL_TERMS:
            return tok
        if lower in common_typo_map:
            corrected = common_typo_map[lower]
            if tok.isupper():
                return corrected.upper()
            if tok[0].isupper():
                return corrected.capitalize()
            return corrected
        if not lower.isalpha():
            return tok
        # Sort to keep matching deterministic across Python runs (since _TYPO_CANONICAL_TERMS is a set).
        match = difflib.get_close_matches(
            lower, sorted(_TYPO_CANONICAL_TERMS), n=1, cutoff=0.86
        )
        if not match:
            return tok
        corrected = match[0]
        # Preserve uppercase-ish style where practical.
        if tok.isupper():
            return corrected.upper()
        if tok[0].isupper():
            return corrected.capitalize()
        return corrected

    # Replace only word tokens; keep separators as-is.
    return re.sub(r"[A-Za-z]+", lambda m: fix_token(m.group(0)), query)


def _is_greeting(query: str) -> bool:
    q = _normalize_query_typos((query or "").strip()).lower()
    if not q:
        return False
    q_compact = re.sub(r"[^a-z0-9\s]", "", q)
    q_compact = re.sub(r"\s+", " ", q_compact).strip()
    tokens = q_compact.split()

    def similar(a: str, b: str, cutoff: float = 0.72) -> bool:
        if a == b:
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= cutoff

    def looks_like_how_are_you(seq: List[str]) -> bool:
        if len(seq) < 2:
            return False
        how_typos = {"hui", "hoe", "hwo", "haw", "houw", "howw"}
        first_tok = seq[0]
        first_ok = similar(first_tok, "how", 0.60) or first_tok in how_typos
        if len(seq) == 2:
            # Handle compact variants like "how you", "hui you"
            second_ok = similar(seq[1], "you", 0.70) or seq[1] == "u"
            return first_ok and second_ok
        second_ok = similar(seq[1], "are", 0.60) or seq[1] in {"r", "re"}
        third_ok = similar(seq[2], "you", 0.70) or seq[2] == "u"
        return first_ok and second_ok and third_ok

    # Common "I'm fine / I'm good" responses after the bot asks "How are you?"
    if re.search(r"\b(i['’]m|i am)\s+(fine|good|well|okay|ok)\b", q):
        return True
    if any(p in q for p in ("doing well", "all good", "im good", "i'm good")):
        return True
    # Accept typo and shorthand variants: "how re you", "how r you", "howreyou", etc.
    if (
        re.search(r"\bhow\s*(are|re|r)\s*(you|u)\b", q_compact)
        or "howreyou" in q_compact.replace(" ", "")
        or "howru" in q_compact.replace(" ", "")
    ):
        return True
    if looks_like_how_are_you(tokens):
        return True
    # Word-boundary only — substring checks like `"hi" in q` match "hiring", "highlight", etc.
    if re.search(r"(?<![a-z0-9])(assalam|assalam-o-alaikum|assalamualaikum)(?![a-z0-9])", q_compact):
        return True
    if re.search(r"(?<![a-z0-9])salam(?![a-z0-9])", q_compact) and len(tokens) <= 5:
        return True
    if re.search(r"(?<![a-z0-9])aoa(?![a-z0-9])", q_compact) and len(tokens) <= 4:
        return True
    # Short standalone hi / hello / hey (not substrings inside longer words)
    if len(q_compact) <= 48:
        if re.fullmatch(
            r"(hi|hello|hey)(\s+(there|team|everyone|all))?(\s*[!\.]*)?",
            q_compact,
        ):
            return True
    return False


def _greeting_answer(query: str) -> str:
    raw_q = (query or "").lower()
    raw_compact = re.sub(r"[^a-z0-9\s]", "", raw_q)
    raw_compact = re.sub(r"\s+", " ", raw_compact).strip()
    q = _normalize_query_typos(query or "").lower()
    q_compact = re.sub(r"[^a-z0-9\s]", "", q)
    q_compact = re.sub(r"\s+", " ", q_compact).strip()
    tokens = q_compact.split()

    def similar(a: str, b: str, cutoff: float = 0.72) -> bool:
        if a == b:
            return True
        return difflib.SequenceMatcher(None, a, b).ratio() >= cutoff

    def looks_like_how_are_you(seq: List[str]) -> bool:
        if len(seq) < 2:
            return False
        how_typos = {"hui", "hoe", "hwo", "haw", "houw", "howw"}
        first_tok = seq[0]
        first_ok = similar(first_tok, "how", 0.60) or first_tok in how_typos
        if len(seq) == 2:
            second_ok = similar(seq[1], "you", 0.70) or seq[1] == "u"
            return first_ok and second_ok
        second_ok = similar(seq[1], "are", 0.60) or seq[1] in {"r", "re"}
        third_ok = similar(seq[2], "you", 0.70) or seq[2] == "u"
        return first_ok and second_ok and third_ok

    # Exact canonical phrasing should get a normal response (no typo hint).
    if raw_compact == "how are you":
        return "I am doing well, thank you. How can I help you?"

    if (
        re.search(r"\bhow\s*(are|re|r)\s*(you|u)\b", q_compact)
        or "howreyou" in q_compact.replace(" ", "")
        or "howru" in q_compact.replace(" ", "")
        or looks_like_how_are_you(tokens)
    ):
        return "I'm good, looks like you meant \"How are you?\" How can I help you?"
    if re.search(r"\b(i['’]m|i am)\s+(fine|good|well|okay|ok)\b", q) or any(
        p in q for p in ("doing well", "all good", "im good", "i'm good")
    ):
        return "Glad to hear it. What can I help you find in the employee handbook?"
    if re.search(r"(?<![a-z0-9])(assalam|assalam-o-alaikum|assalamualaikum)(?![a-z0-9])", q_compact):
        return "Wa alaikum assalam. How can I help you with the employee handbook?"
    if re.search(r"(?<![a-z0-9])salam(?![a-z0-9])", q_compact) and len(tokens) <= 5:
        return "Wa alaikum assalam. How can I help you with the employee handbook?"
    if re.search(r"(?<![a-z0-9])aoa(?![a-z0-9])", q_compact):
        return "Wa alaikum assalam. How can I help you with the employee handbook?"
    if len(q_compact) <= 48 and re.fullmatch(
        r"(hi|hello|hey)(\s+(there|team|everyone|all))?(\s*[!\.]*)?",
        q_compact,
    ):
        return _HANDBOOK_ASSISTANT_GREETING
    return _HANDBOOK_ASSISTANT_GREETING


def _build_context(
    retrieved: List[Dict[str, Any]],
    *,
    max_context_tokens: int,
) -> List[Dict[str, Any]]:
    """
    Pack chunks in retrieval (rerank) order so the best hits use the budget first.
    Truncates an oversized or final chunk to fill remaining tokens instead of skipping it.
    """
    context: List[Dict[str, Any]] = []
    running_tokens = 0
    min_tail_tokens = int(getattr(settings, "RAG_CONTEXT_MIN_TAIL_TOKENS", 100))

    for r in retrieved:
        payload = r.get("payload") or {}
        text = str(payload.get("text") or "")
        if not text:
            continue

        full_tc = payload.get("token_count")
        if full_tc is None:
            full_tc = count_tokens(text)
        try:
            full_tc = int(full_tc)
        except (TypeError, ValueError):
            full_tc = count_tokens(text)

        room = max_context_tokens - running_tokens
        if room <= 0:
            break

        use_text = text
        token_count = full_tc
        if token_count > room:
            if room < min_tail_tokens and context:
                break
            use_text = truncate_text_to_token_budget(text, room)
            token_count = count_tokens(use_text)
            if token_count <= 0:
                continue

        context.append(
            {
                "text": use_text,
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": token_count,
                "score": r.get("score"),
                "source_section": (payload.get("source_section") or "").strip(),
                "page_number": payload.get("page_number"),
            }
        )
        running_tokens += token_count

    return context


def _dense_candidate_limit(top_k: int) -> int:
    mult = max(1, int(getattr(settings, "RAG_SEMANTIC_CANDIDATE_MULTIPLIER", 16)))
    floor = max(1, int(getattr(settings, "RAG_SEMANTIC_CANDIDATE_MIN", 96)))
    return max(top_k * mult, floor)


def _retrieved_row_chunk_key(row: Dict[str, Any]) -> str:
    pl = row.get("payload") or {}
    cid = pl.get("chunk_id")
    if cid is not None and str(cid).strip():
        return str(cid)
    return f"{pl.get('doc_id')}_{pl.get('chunk_index')}"


def _hybrid_filter_for_context(
    retrieved: List[Dict[str, Any]],
    *,
    query: str,
    top_k: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    """
    Keep chunks that pass dense threshold OR are within a relative margin of the best
    dense score OR have lexical (BM25) signal. Always union top RRF rows so the best
    fused ranking still reaches strict mode even when raw cosine sits below the floor.
    """
    if not retrieved:
        return []

    _ = query  # kept for API symmetry / future query-conditioned filtering

    bm25_min = float(getattr(settings, "RAG_LEXICAL_BM25_MIN", 0.0))
    query_terms = _retrieval_terms(query)
    # For longer queries, require stronger lexical agreement to reduce noisy chunk admission.
    min_lexical_overlap = 2 if len(query_terms) >= 8 else 1

    sem_scores: List[float] = []
    for r in retrieved:
        s = r.get("score")
        if s is None:
            continue
        try:
            sem_scores.append(float(s))
        except (TypeError, ValueError):
            pass
    max_sem = max(sem_scores) if sem_scores else None
    min_sem = min(sem_scores) if sem_scores else None
    relative = float(getattr(settings, "RAG_SEMANTIC_RELATIVE_FLOOR", 0.78))
    lower_is_better = bool(getattr(settings, "RAG_SEMANTIC_SCORE_LOWER_IS_BETTER", False))

    def passes_semantic(sem_f: float) -> bool:
        if lower_is_better:
            if sem_f <= threshold:
                return True
            if min_sem is not None and min_sem > 1e-12:
                return sem_f <= min_sem / max(relative, 0.5)
            return False
        if sem_f >= threshold:
            return True
        if max_sem is not None and max_sem > 1e-12:
            return sem_f >= max_sem * relative
        return False

    primary: List[Dict[str, Any]] = []
    for r in retrieved:
        sem_f: Optional[float] = None
        if r.get("score") is not None:
            try:
                sem_f = float(r["score"])
            except (TypeError, ValueError):
                sem_f = None
        sem_ok = passes_semantic(sem_f) if sem_f is not None else False
        ov = int(r.get("_overlap") or 0)
        bm25 = float(r.get("_bm25") or 0)
        lex_ok = bm25 > bm25_min and ov >= min_lexical_overlap
        if sem_ok or lex_ok:
            primary.append(r)

    seen: set[str] = set()
    merged: List[Dict[str, Any]] = []
    for r in primary:
        k = _retrieved_row_chunk_key(r)
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)

    rrf_slots = max(int(getattr(settings, "RAG_RRF_MIN_GUARANTEE", 18)), top_k * 3)
    by_rrf = sorted(retrieved, key=lambda x: -float(x.get("_rrf", 0)))
    for r in by_rrf[:rrf_slots]:
        k = _retrieved_row_chunk_key(r)
        if k in seen:
            continue
        seen.add(k)
        merged.append(r)

    if not merged:
        # Do not widen to weak semantic hits — that pulls in unrelated handbook text.
        return []

    return merged


def _citation_source_fields(chunk: Dict[str, Any]) -> tuple[Optional[str], Optional[int]]:
    sec = (chunk.get("source_section") or "").strip() or None
    page: Optional[int] = None
    raw_page = chunk.get("page_number")
    if raw_page is not None:
        try:
            page = int(raw_page)
        except (TypeError, ValueError):
            page = None
    return sec, page


def _citation_entry(chunk: Dict[str, Any]) -> Dict[str, Any]:
    sec, pg = _citation_source_fields(chunk)
    return {
        "doc_id": chunk.get("doc_id"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_id": chunk.get("chunk_id"),
        "text_preview": _text_preview(chunk.get("text") or ""),
        "token_count": chunk.get("token_count"),
        "source_section": sec,
        "page": pg,
    }


def _text_preview(text: str, *, max_chars: int = 250) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "…"


def _tokenize_keywords(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
        if len(w) > 2 and w not in _STOPWORDS
    }


def _retrieval_terms(question: str) -> set[str]:
    """
    Terms for matching user questions to chunk text. Excludes only generic
    filler/meta tokens (not normal handbook words like policy, employee, work).
    """
    terms = _tokenize_keywords(question)
    out = {t for t in terms if t not in _GENERIC_QUERY_TERMS and len(t) >= 2}
    return out if out else terms


def _strong_focus_terms(terms: set[str]) -> set[str]:
    """
    Subset of query terms that are specific enough to anchor strict extraction.
    Weak terms alone match unrelated policy lines (e.g. "allowed", "many").
    """
    if not terms:
        return set()
    weak = _WEAK_FOCUS_TERMS | _GENERIC_QUERY_TERMS
    strong = {t for t in terms if t not in weak and len(t) >= 3}
    if not strong:
        strong = {t for t in terms if t not in weak and len(t) >= 2}
    if not strong:
        strong = set(terms)
    return strong


def _matched_query_terms_in_line(line: str, terms: set[str]) -> set[str]:
    low = line.lower()
    out: set[str] = set()
    for t in terms:
        if len(t) < 2:
            continue
        if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", low, flags=re.IGNORECASE):
            out.add(t)
    return out


def _line_matches_focus_for_expansion(line: str, terms: set[str], strong_terms: set[str]) -> bool:
    """True if this line is a safe anchor for query-focused block expansion."""
    if not terms:
        return False
    matched = _matched_query_terms_in_line(line, terms)
    if not matched:
        return False
    if not strong_terms or strong_terms == terms:
        return True
    if matched.intersection(strong_terms):
        return True
    return len(matched) >= 2


def _chunk_reading_key(chunk: Dict[str, Any]) -> tuple[int, int, str]:
    try:
        doc_id = int(chunk.get("doc_id") or 0)
    except (TypeError, ValueError):
        doc_id = 0
    raw = chunk.get("chunk_index")
    try:
        idx = int(raw) if raw is not None else -1
    except (TypeError, ValueError):
        idx = -1
    return (doc_id, idx, str(chunk.get("chunk_id") or ""))


def _sort_chunks_by_reading_order(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(chunks, key=_chunk_reading_key)


def _bm25_score_documents(
    docs: List[tuple[str, str]],
    keywords: List[str],
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> List[float]:
    """
    BM25 over an in-memory corpus (scanned Qdrant payloads or retrieved context).
    Keywords come from the live query only — no static topic lists.
    """
    if not docs:
        return []
    if not keywords:
        return [0.0] * len(docs)

    N = len(docs)
    blobs: List[str] = []
    lengths: List[int] = []
    for body, sec in docs:
        blob = f"{body} {sec}".strip().lower()
        blobs.append(blob)
        lengths.append(max(len(blob.split()), 1))
    avgdl = sum(lengths) / N

    df: Dict[str, int] = {kw: 0 for kw in keywords}
    term_freqs: List[Dict[str, int]] = []
    for blob in blobs:
        tf: Dict[str, int] = {}
        for kw in keywords:
            pat = rf"(?<!\w){re.escape(kw)}(?!\w)"
            n = len(re.findall(pat, blob, flags=re.IGNORECASE))
            if n > 0:
                tf[kw] = n
                df[kw] += 1
        term_freqs.append(tf)

    idf: Dict[str, float] = {}
    for kw in keywords:
        d = df.get(kw, 0)
        idf[kw] = math.log(1.0 + (N - d + 0.5) / (d + 0.5))

    scores: List[float] = []
    for i in range(N):
        dl = lengths[i]
        s = 0.0
        for kw in keywords:
            f = term_freqs[i].get(kw, 0)
            if f == 0:
                continue
            denom = f + k1 * (1.0 - b + b * dl / avgdl)
            s += idf[kw] * (f * (k1 + 1.0)) / denom
        scores.append(s)
    return scores


def _strict_chunk_scores(
    context_chunks: List[Dict[str, Any]],
    *,
    q_terms: set[str],
    strong_terms: set[str],
    min_overlap: int,
    chunk_order: Dict[str, int],
) -> List[tuple[float, int, int, Dict[str, Any]]]:
    """
    Rank context chunks with BM25 + token overlap on the retrieved set (dynamic IDF).
    """
    keywords = list(q_terms)
    rows: List[tuple[str, str, Dict[str, Any]]] = []
    for c in context_chunks:
        text = str(c.get("text") or "").strip()
        if not text:
            continue
        sec = str(c.get("source_section") or "")
        combined_terms = _tokenize_keywords(text) | _tokenize_keywords(sec)
        overlap = len(q_terms.intersection(combined_terms))
        if overlap < min_overlap:
            continue
        if strong_terms and not (strong_terms.intersection(combined_terms)):
            continue
        rows.append((text, sec, c))

    if not rows:
        return []

    docs = [(t, s) for t, s, _ in rows]
    bm25s = _bm25_score_documents(docs, keywords) if keywords else [0.0] * len(rows)

    scored: List[tuple[float, int, int, Dict[str, Any]]] = []
    for i, (_t, _s, c) in enumerate(rows):
        text = str(c.get("text") or "").strip()
        sec = str(c.get("source_section") or "")
        overlap = len(q_terms.intersection(_tokenize_keywords(text) | _tokenize_keywords(sec)))
        cid = str(c.get("chunk_id") or "")
        scored.append((bm25s[i] if i < len(bm25s) else 0.0, overlap, chunk_order.get(cid, 999), c))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return scored


def _strict_body_from_chunks_ordered(
    chunks: List[Dict[str, Any]],
    *,
    max_chars: int,
    respect_input_order: bool = False,
) -> str:
    """
    Verbatim handbook prose. By default, sorts by document reading order.
    When respect_input_order is True, uses the caller's list order (e.g. relevance first).
    """
    ordered = chunks if respect_input_order else _sort_chunks_by_reading_order(chunks)
    parts: List[str] = []
    total = 0
    for c in ordered:
        lines: List[str] = []
        for raw in str(c.get("text") or "").splitlines():
            ln = raw.strip()
            if not ln or line_looks_like_toc_leader(ln):
                continue
            lines.append(ln)
        block = _filter_to_complete_sentences(
            _clean_extractive_sentence(" ".join(lines)).strip(" -•*")
        )
        if not block:
            continue
        add = len(block) + (1 if parts else 0)
        if parts and total + add > max_chars:
            room = max_chars - total - 1
            if room > 120:
                block = block[:room].rsplit(" ", 1)[0].rstrip(",;:") + " …"
                block = _filter_to_complete_sentences(block) or block
                parts.append(block)
            break
        parts.append(block)
        total += add
    joined = " ".join(parts).strip()
    return _filter_to_complete_sentences(joined) or joined


def _strict_body_query_focused(
    chunks: List[Dict[str, Any]],
    *,
    query_terms: set[str],
    strong_terms: set[str],
    max_chars: int,
    respect_input_order: bool,
) -> str:
    """Prefer lines that contain query keywords (verbatim); reduces unrelated policy prose."""
    ordered = chunks if respect_input_order else _sort_chunks_by_reading_order(chunks)
    terms = {t for t in query_terms if len(t) >= 2}
    if not terms:
        return ""
    strong = strong_terms & terms if strong_terms else _strong_focus_terms(terms)
    parts: List[str] = []
    total = 0
    for c in ordered:
        raw_lines = [ln for ln in str(c.get("text") or "").splitlines() if ln.strip()]
        block_src = _expand_query_focus_lines(raw_lines, terms, strong)
        block = _filter_to_complete_sentences(_clean_extractive_sentence(" ".join(block_src)).strip(" -•*"))
        if not block:
            continue
        add = len(block) + (1 if parts else 0)
        if parts and total + add > max_chars:
            room = max_chars - total - 1
            if room > 120:
                block = block[:room].rsplit(" ", 1)[0].rstrip(",;:") + " …"
                parts.append(block)
            break
        parts.append(block)
        total += add
    return " ".join(parts).strip()


def _strict_verbatim_top_chunks(chunks: List[Dict[str, Any]], *, max_chars: int) -> str:
    """Last-resort verbatim excerpt from the highest-ranked context chunks (still handbook-only)."""
    if not chunks or max_chars < 80:
        return ""
    parts: List[str] = []
    for c in chunks[:5]:
        text = str(c.get("text") or "").strip()
        if not text:
            continue
        lines = [
            ln.strip()
            for ln in text.splitlines()
            if ln.strip() and not line_looks_like_toc_leader(ln.strip())
        ]
        block = _filter_to_complete_sentences(
            _clean_extractive_sentence(" ".join(lines)).strip(" -•*")
        )
        if not block:
            continue
        if len(block) < 200 and _is_extractive_noise_sentence(block):
            continue
        joined = " ".join(parts + [block])
        if len(joined) > max_chars:
            room = max_chars - len(" ".join(parts)) - (1 if parts else 0)
            if room < 100:
                break
            block = block[:room].rsplit(" ", 1)[0].rstrip(",;:") + " …"
        parts.append(block)
        if len(" ".join(parts)) >= max_chars:
            break
    return " ".join(parts).strip()


def _is_extractive_noise_sentence(s: str) -> bool:
    """Filter table-of-contents lines, dot leaders, and other non-prose fragments."""
    if not s:
        return True
    t = s.strip()
    if len(t) < 22:
        return True
    if re.search(r"\.{4,}", t):
        return True
    letters = sum(c.isalpha() for c in t)
    if letters < 14:
        return True
    dots = t.count(".")
    if dots >= 6 and letters > 0 and dots > letters * 0.35:
        return True

    # Treat short, title-like lines with mostly capitalized words and no clear
    # verb as headings, not answer content (e.g. "Employee Referral Policy",
    # "Duration of Employment Referral Amount").
    if not any(ch in t for ch in ".!?"):
        # Remove obvious trailing punctuation like ":" for heuristic checks.
        core = re.sub(r"[:\-–]+$", "", t).strip()
        words = core.split()
        if 2 <= len(words) <= 8:
            title_like = sum(1 for w in words if w[:1].isupper())
            lower_like = sum(1 for w in words if w.islower())
            if title_like >= max(2, len(words) - 1) and lower_like <= 1:
                # Also ensure we don't accidentally drop imperative/verb phrases.
                verb_markers = {"is", "are", "was", "were", "must", "should", "shall", "may", "can", "could", "will"}
                if not any(v in core.lower().split() for v in verb_markers):
                    return True

    return False


def _clean_extractive_sentence(s: str) -> str:
    t = re.sub(r"\.{3,}", " ", s)
    return re.sub(r"\s+", " ", t).strip()


_INCOMPLETE_TAIL_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "to",
        "for",
        "and",
        "or",
        "that",
        "which",
        "whose",
        "by",
        "on",
        "in",
        "as",
        "at",
        "is",
        "are",
        "be",
        "been",
        "was",
        "were",
        "defines",
        "define",
        "including",
        "such",
        "from",
        "with",
        "into",
        "all",
        "any",
        "each",
        "every",
        "their",
        "our",
        "your",
        "this",
        "these",
        "those",
    }
)


def _filter_to_complete_sentences(text: str) -> str:
    """
    Drop mid-sentence leading fragments and trailing clauses that end on function words
    without terminal punctuation (common when lines are stitched from PDFs).
    """
    t = _clean_extractive_sentence(text)
    if not t:
        return t
    segments = re.split(r"(?<=[.!?])\s+", t)
    kept: List[str] = []
    for seg in segments:
        s = seg.strip()
        if not s or len(s) < 18:
            continue
        if s[0].isalpha() and s[0].islower():
            continue
        terminal = s[-1] in ".!?\"'"
        if not terminal:
            parts = s.split()
            if parts:
                tail = parts[-1].lower().rstrip(".,;:'\"")
                if tail in _INCOMPLETE_TAIL_WORDS:
                    continue
        kept.append(s)
    return " ".join(kept) if kept else t


def _expand_query_focus_lines(
    raw_lines: List[str],
    terms: set[str],
    strong_terms: set[str],
) -> List[str]:
    """
    When some lines match query terms, include the full contiguous block from the first
    through the last matching line (and non-empty lines in between) so we do not splice
    isolated mid-paragraph lines into half-sentences.
    Uses anchor-aware matching so weak-only hits (e.g. "allowed" alone) do not expand.
    """
    cleaned = [ln.strip() for ln in raw_lines if ln.strip()]
    cleaned = [ln for ln in cleaned if not line_looks_like_toc_leader(ln)]
    if not cleaned:
        return []
    strong = strong_terms if strong_terms else _strong_focus_terms(terms)
    match = [_line_matches_focus_for_expansion(ln, terms, strong) for ln in cleaned]
    if not any(match):
        return cleaned
    first = min(i for i, m in enumerate(match) if m)
    last = max(i for i, m in enumerate(match) if m)
    return cleaned[first : last + 1]


def _section_label_plausible_for_chunk(sec: str, chunk_text: str) -> bool:
    """
    Trust ingestion-provided source_section unless it clearly comes from a TOC line.

    Handbook chunks often store the heading only in payload; the body does not repeat
    it, so we must not require a substring match (that incorrectly forced "Unknown").
    """
    sec = (sec or "").strip()
    if not sec:
        return False
    if re.search(r"\.{4,}", sec):
        return False
    chunk_text = chunk_text or ""
    sec_l = sec.lower()
    found_on_clean = False
    found_on_toc = False
    for ln in chunk_text.splitlines():
        st = ln.strip()
        if not st:
            continue
        if sec not in st and sec_l not in st.lower():
            continue
        if re.search(r"\.{4,}", st):
            found_on_toc = True
        else:
            found_on_clean = True
    if found_on_clean:
        return True
    if found_on_toc:
        return False
    # Section label not repeated in body — normal for SOURCE_SECTION / auto-structured ingest.
    return True


_VOWEL_CHARS = frozenset("aeiouy")


def _is_obvious_gibberish_query(query: str) -> bool:
    """
    Cheap, high-precision filter for random keyboard noise (e.g. 'nfbhgbgh').
    Runs before embeddings/Qdrant so retrieval is not wasted. Does not use the
    old HR typo whitelist — only structural / vowel heuristics.
    """
    raw = (query or "").strip()
    if not raw:
        return False

    letters_only = re.sub(r"[^a-zA-Z]", "", raw)
    letters_low = letters_only.lower()

    if not letters_low:
        compact = re.sub(r"\s+", "", raw)
        return len(compact) >= 4

    if len(letters_low) >= 4 and not any(c in _VOWEL_CHARS for c in letters_low):
        return True

    if len(letters_low) >= 12:
        vc = sum(1 for c in letters_low if c in _VOWEL_CHARS)
        ratio = vc / len(letters_low)
        if vc == 0 or ratio < 0.1:
            return True

    if re.search(r"(.)\1{4,}", letters_low):
        return True

    return False


def _is_probably_gibberish(query: str) -> bool:
    """
    Optional pre-RAG filter (see RAG_ENABLE_GIBBERISH_FILTER; default off).
    Previously ran for every query and returned UNKNOWN without hitting embeddings/Qdrant/LLM
    whenever no word matched a tiny HR typo list — causing many false "document doesn't mention".
    """
    if not getattr(settings, "RAG_ENABLE_GIBBERISH_FILTER", False):
        return False

    q = _sanitize_query(query)
    if not q:
        return False

    q_low = q.lower()
    if re.search(
        r"\b(what|when|where|who|why|how|which|whose|whom|tell|explain|describe|define|list|name|"
        r"is|are|was|were|does|did|do|can|could|should|must|may|will|would|about|regarding)\b",
        q_low,
    ):
        return False

    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", q)]
    meaningful = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    if not meaningful:
        return False

    _canonical_sorted = sorted(_TYPO_CANONICAL_TERMS)

    def known_like(tok: str) -> bool:
        if tok in _TYPO_CANONICAL_TERMS:
            return True
        return bool(difflib.get_close_matches(tok, _canonical_sorted, n=1, cutoff=0.84))

    known_count = sum(1 for t in meaningful if known_like(t))
    if known_count > 0:
        return False

    vowel_re = re.compile(r"[aeiou]", re.I)
    no_vowel_word = any(len(t) >= 5 and not vowel_re.search(t) for t in meaningful)
    repeated_noise = any(re.search(r"(.)\1\1", t) for t in meaningful)
    return no_vowel_word or repeated_noise


def _rerank_by_query_overlap(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple lexical reranker to keep chunks that mention query terms closer to the top.
    This improves precision for policy-style questions where exact phrasing matters.
    """
    q_terms = _tokenize_keywords(query)
    rterms = _retrieval_terms(query)
    if not q_terms:
        return items

    def rank_key(item: Dict[str, Any]) -> tuple[int, int, int, int, int, float, float, float]:
        payload = item.get("payload") or {}
        text = str(payload.get("text") or "")
        section = str(payload.get("source_section") or "")

        text_terms = _tokenize_keywords(text)
        sec_terms = _tokenize_keywords(section)

        overlap = len(q_terms.intersection(text_terms))
        anchor_overlap = len(rterms.intersection(text_terms))

        sec_overlap = len(q_terms.intersection(sec_terms))
        sec_anchor_overlap = len(rterms.intersection(sec_terms))

        phrase_hits = sum(1 for a in rterms if a in text.lower() or a in section.lower())
        rrf = float(item.get("_rrf") or 0.0)
        bm25 = float(item.get("_bm25") or 0.0)
        score = float(item.get("score") or 0.0)
        # Prefer chunks whose section title itself matches the query (e.g. "Reimbursement Policy")
        return (
            sec_anchor_overlap,
            anchor_overlap,
            sec_overlap,
            overlap,
            phrase_hits,
            rrf,
            bm25,
            score,
        )

    return sorted(items, key=rank_key, reverse=True)


def _payload_from_scanned_point(p: Any) -> Optional[Dict[str, Any]]:
    if p is None:
        return None
    return p.payload if hasattr(p, "payload") else p.get("payload")


def _retrieved_chunk_id(item: Dict[str, Any]) -> str:
    pl = item.get("payload") or {}
    return str(pl.get("chunk_id") or "")


def _first_rank_maps(
    items: List[Dict[str, Any]],
) -> tuple[Dict[str, int], Dict[str, Dict[str, Any]]]:
    rank: Dict[str, int] = {}
    by_id: Dict[str, Dict[str, Any]] = {}
    for r, item in enumerate(items):
        k = _retrieved_chunk_id(item)
        if not k or k in rank:
            continue
        rank[k] = r
        by_id[k] = item
    return rank, by_id


def _lexical_candidates_bm25(
    *,
    query: str,
    scanned_points: List[Any],
    top_n: int,
) -> List[Dict[str, Any]]:
    """
    Keyword channel: BM25 over scrolled payloads using tokens derived from the query only.
    """
    keywords = list(_retrieval_terms(query))
    if not keywords or not scanned_points:
        return []

    rows: List[tuple[str, str, Dict[str, Any]]] = []
    for p in scanned_points:
        payload = _payload_from_scanned_point(p)
        if not payload or not payload.get("text"):
            continue
        text = str(payload.get("text") or "")
        sec = str(payload.get("source_section") or "")
        rows.append((text, sec, payload))

    if not rows:
        return []

    docs = [(t, s) for t, s, _ in rows]
    scores = _bm25_score_documents(docs, keywords)

    out: List[Dict[str, Any]] = []
    for idx, (text, sec, payload) in enumerate(rows):
        blob = f"{text} {sec}".lower()
        overlap = 0
        for kw in keywords:
            pat = rf"(?<!\w){re.escape(kw)}(?!\w)"
            if re.search(pat, blob, flags=re.IGNORECASE):
                overlap += 1
        if overlap <= 0:
            continue
        out.append(
            {
                "score": None,
                "payload": payload,
                "_lexical": True,
                "_bm25": float(scores[idx]),
                "_overlap": overlap,
            }
        )

    out.sort(key=lambda x: (-float(x["_bm25"]), -int(x["_overlap"])))
    return out[:top_n]


def _hybrid_reciprocal_rank_fuse(
    semantic: List[Dict[str, Any]],
    lexical: List[Dict[str, Any]],
    *,
    rrf_k: int,
) -> List[Dict[str, Any]]:
    """
    Merge dense (semantic) and BM25 (lexical) ranked lists with reciprocal rank fusion.
    """
    sem_rank, sem_by = _first_rank_maps(semantic)
    lex_rank, lex_by = _first_rank_maps(lexical)
    all_keys = set(sem_rank) | set(lex_rank)
    merged: List[Dict[str, Any]] = []

    for key in all_keys:
        rrf_score = 0.0
        if key in sem_rank:
            rrf_score += 1.0 / (rrf_k + sem_rank[key] + 1)
        if key in lex_rank:
            rrf_score += 1.0 / (rrf_k + lex_rank[key] + 1)

        if key in sem_by:
            base = dict(sem_by[key])
        else:
            lx = lex_by[key]
            base = {
                "score": None,
                "payload": lx.get("payload"),
                "_lexical": True,
                "_bm25": float(lx.get("_bm25") or 0),
                "_overlap": int(lx.get("_overlap") or 0),
            }

        if key in lex_by:
            lx = lex_by[key]
            base["_bm25"] = max(float(base.get("_bm25") or 0), float(lx.get("_bm25") or 0))
            base["_overlap"] = max(int(base.get("_overlap") or 0), int(lx.get("_overlap") or 0))
            base["_lexical"] = bool(base.get("_lexical")) or bool(lx.get("_lexical"))
        else:
            base.setdefault("_bm25", 0.0)
            base.setdefault("_overlap", 0)

        base["_rrf"] = rrf_score
        merged.append(base)

    merged.sort(key=lambda x: -float(x.get("_rrf", 0.0)))
    return merged


def _hybrid_merge_semantic_and_lexical(
    *,
    query: str,
    qdrant: QdrantService,
    semantic_items: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Always runs the lexical channel when the query yields keywords and Qdrant scroll is
    available; fuses with semantic hits via RRF so both signals contribute every time.
    """
    lexical_items: List[Dict[str, Any]] = []
    keywords = _retrieval_terms(query)
    scan_fn = getattr(qdrant, "scan_payload_points", None)
    if keywords and callable(scan_fn):
        per_page = int(getattr(settings, "RAG_LEXICAL_SCAN_LIMIT", 150))
        max_points = int(getattr(settings, "RAG_LEXICAL_MAX_POINTS", 400))
        try:
            scanned = scan_fn(limit=per_page, max_points=max_points)
        except Exception:
            scanned = []
        lexical_top = max(top_k * 6, 36)
        lexical_items = _lexical_candidates_bm25(
            query=query,
            scanned_points=scanned or [],
            top_n=lexical_top,
        )

    rrf_k = int(getattr(settings, "RAG_HYBRID_RRF_K", 60))
    return _hybrid_reciprocal_rank_fuse(semantic_items, lexical_items, rrf_k=rrf_k)


def _extractive_answer_from_context(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Deterministic fallback: extract relevant sentences directly from retrieved context.
    Prevents false "I don't know" on policy questions where text is present.
    """
    q_terms = set(_retrieval_terms(question)) or _tokenize_keywords(question)
    if not q_terms:
        return ""
    strong_terms = _strong_focus_terms(q_terms)

    ordered_chunks = _sort_chunks_by_reading_order(list(context_chunks))
    candidates: List[tuple[int, int, int, str]] = []
    for ck_idx, c in enumerate(ordered_chunks):
        text = str(c.get("text") or "")
        if not text:
            continue
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
        for pos, p in enumerate(parts):
            sentence = p.strip()
            if len(sentence) < 20:
                continue
            if _is_extractive_noise_sentence(sentence):
                continue
            if sentence[0].isalpha() and sentence[0].islower():
                continue
            sent_terms = _tokenize_keywords(sentence)
            overlap = len(q_terms.intersection(sent_terms))
            if overlap <= 0:
                continue
            if strong_terms and not (strong_terms.intersection(sent_terms)) and overlap < 2:
                continue
            candidates.append((ck_idx, pos, overlap, sentence))

    if not candidates:
        return ""

    # Reading order first, then overlap — avoids shuffled half-thoughts from score-only sorting.
    candidates.sort(key=lambda x: (x[0], x[1], -x[2]))
    top = []
    seen = set()
    for _ci, _pos, _ov, sentence in candidates:
        cleaned = _clean_extractive_sentence(sentence)
        if _is_extractive_noise_sentence(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        top.append(cleaned)
        cap = int(getattr(settings, "RAG_STRICT_EXTRACTIVE_MAX_BULLETS", 10))
        if len(top) >= cap:
            break

    if not top:
        return ""
    return "Based on the policy:\n- " + "\n- ".join(top)


def _extractive_answer_with_sources_from_context(
    question: str,
    context_chunks: List[Dict[str, Any]],
    *,
    max_sentences: int = 4,
    query_embedding: Optional[List[float]] = None,
    force_lexical_sentence_ranking: bool = False,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Strict, non-hallucinating mode:
    - Selects verbatim sentences from retrieved chunks (skips TOC / dot-leader lines).
    - Prefers sentences from higher-ranked context chunks before mixing in lower chunks.
    - Source section / page come from payload only when the section label is not TOC-only.
    When force_lexical_sentence_ranking is True (e.g. embedding API unavailable), ranks
    sentences by anchor overlap only — no embedding calls.
    Returns (answer_text, used_context_chunks).
    """
    _ = (query_embedding, force_lexical_sentence_ranking, max_sentences)

    # Match questions to chunks using broad retrieval terms (includes policy,
    # employee, reimbursement, etc.). Anchor-only matching was too strict and
    # caused false "document doesn't mention it" for normal handbook questions.
    q_terms = set(_retrieval_terms(question))
    if not q_terms:
        q_terms = _tokenize_keywords(question)
    if not q_terms:
        return "", []

    strong_terms = _strong_focus_terms(q_terms)

    min_overlap = int(getattr(settings, "RAG_STRICT_MIN_SENTENCE_OVERLAP", 1))
    if len(q_terms) >= 8:
        min_overlap = max(min_overlap, 2)
    if len(q_terms) >= 3:
        min_overlap = max(min_overlap, 2)

    max_chunks_for_body = int(getattr(settings, "RAG_STRICT_MAX_CHUNKS_FOR_BODY", 12))
    chunk_order: Dict[str, int] = {}
    for i, c in enumerate(context_chunks):
        cid = str(c.get("chunk_id") or "")
        if cid:
            chunk_order[cid] = i

    chunk_scored = _strict_chunk_scores(
        context_chunks,
        q_terms=q_terms,
        strong_terms=strong_terms,
        min_overlap=min_overlap,
        chunk_order=chunk_order,
    )

    selected_scored: List[tuple[float, int, Dict[str, Any]]] = []
    seen_chunk_ids: set[str] = set()
    for _bm25, ov, _ord, c in chunk_scored:
        cid = str(c.get("chunk_id") or "")
        if cid and cid in seen_chunk_ids:
            continue
        if cid:
            seen_chunk_ids.add(cid)
        selected_scored.append((_bm25, ov, c))
        if len(selected_scored) >= max_chunks_for_body:
            break

    before_bm_trim = list(selected_scored)
    rel_floor = float(getattr(settings, "RAG_STRICT_BM25_RELATIVE_FLOOR", 0.0))
    if rel_floor > 0 and len(selected_scored) > 1:
        top_bm = max((s[0] for s in selected_scored), default=0.0)
        if top_bm > 1e-9:
            selected_scored = [s for s in selected_scored if s[0] >= rel_floor * top_bm]
    if not selected_scored and before_bm_trim:
        selected_scored = before_bm_trim[:1]

    if not selected_scored:
        return "", []

    # BM25 within context, then overlap, then reading order for ties.
    selected_scored.sort(
        key=lambda oc: (-oc[0], -oc[1], _chunk_reading_key(oc[2]))
    )
    selected_chunks = [c for _bm25, _ov, c in selected_scored]

    body_char_cap = int(getattr(settings, "RAG_STRICT_ANSWER_BODY_CHARS", 8000))
    answer_paragraph = ""
    if getattr(settings, "RAG_STRICT_QUERY_FOCUSED_BODY", True):
        answer_paragraph = _strict_body_query_focused(
            selected_chunks,
            query_terms=q_terms,
            strong_terms=strong_terms,
            max_chars=body_char_cap,
            respect_input_order=True,
        )
    min_focus_chars = max(48, min(200, body_char_cap // 40))
    if len(answer_paragraph.strip()) < min_focus_chars:
        answer_paragraph = _strict_body_from_chunks_ordered(
            selected_chunks,
            max_chars=body_char_cap,
            respect_input_order=True,
        )
    if not answer_paragraph.strip():
        fb = _extractive_answer_from_context(question, selected_chunks)
        if fb.strip():
            answer_paragraph = fb
    if not answer_paragraph.strip():
        answer_paragraph = _strict_verbatim_top_chunks(
            selected_chunks,
            max_chars=body_char_cap,
        )
    if not answer_paragraph.strip():
        return "", []

    ap_stripped = answer_paragraph.strip()
    ap_filtered = _filter_to_complete_sentences(ap_stripped)
    if ap_filtered.strip():
        answer_paragraph = ap_filtered

    if answer_paragraph and not answer_paragraph.endswith("."):
        answer_paragraph += "."

    if not _answer_includes_question_anchors(answer_paragraph, question):
        return "", []

    # Return body only; source/page metadata is appended in the formatter layer.
    return answer_paragraph.strip(), selected_chunks


def _should_use_extractive_fallback(question: str, context_chunks: List[Dict[str, Any]]) -> bool:
    """
    Use extractive fallback only when key terms from the question
    are actually present in retrieved context.
    """
    rterms = _retrieval_terms(question)
    if not rterms:
        return False

    context_terms: set[str] = set()
    for c in context_chunks:
        context_terms |= _tokenize_keywords(str(c.get("text") or ""))

    overlap = len(rterms.intersection(context_terms))
    return overlap >= 1


def _query_is_covered_by_context(question: str, context_chunks: List[Dict[str, Any]]) -> bool:
    rterms = _retrieval_terms(question)
    if not rterms:
        return False
    context_terms: set[str] = set()
    for c in context_chunks:
        context_terms |= _tokenize_keywords(str(c.get("text") or ""))
    overlap = len(rterms.intersection(context_terms))
    return overlap >= 1


def _answer_includes_question_anchors(answer_body: str, question: str) -> bool:
    """True when the draft still reflects at least one substantive term from the question."""
    terms = set(_retrieval_terms(question))
    if not terms:
        terms = _tokenize_keywords(question)
    if not terms:
        return True
    low = (answer_body or "").lower()
    strong = _strong_focus_terms(terms)
    if strong:
        return any(t in low for t in strong)
    return any(t in low for t in terms)


def _llm_draft_grounded_in_context(draft: str, context_chunks: List[Dict[str, Any]]) -> bool:
    """
    Cheap lexical check: most content words in the model reply should appear in retrieved text.
    Reduces hallucinations when RAG_STRICT_FALLBACK_TO_LLM is enabled.
    """
    d = (draft or "").strip()
    if not d or d == UNKNOWN_POLICY_PHRASE:
        return True
    ctx = "\n".join(str(c.get("text") or "") for c in context_chunks).lower()
    d_body = re.sub(r"(?is)\n\s*sources:.*", "", d).strip()
    terms = [t for t in _tokenize_keywords(d_body) if len(t) >= 3]
    if not terms:
        terms = list(_tokenize_keywords(d_body))
    if not terms:
        return False
    hits = sum(1 for t in terms if t in ctx)
    need = 3 if len(terms) >= 5 else max(1, min(len(terms), 2))
    return hits >= need


def _prefer_extractive_for_question(question: str) -> bool:
    q = question.lower()
    triggers = ("prohibited", "not allowed", "must not", "forbidden", "policy on", "rules")
    return any(t in q for t in triggers)


def _manual_pipeline(
    *,
    query: str,
    top_k: int,
    threshold: float,
    max_context_tokens: int,
) -> Dict[str, Any]:
    qdrant = QdrantService()
    t0 = time.time()
    try:
        sanitized = _sanitize_query(query)
        query_embedding: Optional[List[float]] = None
        embedding_failed = False
        try:
            query_embedding = get_embeddings([sanitized])[0]
        except Exception:
            embedding_failed = True

        retrieved: List[Dict[str, Any]] = []
        if not embedding_failed and query_embedding is not None:
            # Pull a wider semantic candidate set, then rerank by query overlap.
            search_results = qdrant.search(query_embedding, limit=_dense_candidate_limit(top_k)) or []

            # Qdrant score: higher is more similar (depending on distance metric).
            for r in search_results:
                score = r.score if hasattr(r, "score") else r.get("score")
                payload = r.payload if hasattr(r, "payload") else r.get("payload")
                retrieved.append({"score": score, "payload": payload})

        retrieved = _hybrid_merge_semantic_and_lexical(
            query=sanitized,
            qdrant=qdrant,
            semantic_items=retrieved,
            top_k=top_k,
        )
        filtered = _hybrid_filter_for_context(
            retrieved,
            query=sanitized,
            top_k=top_k,
            threshold=threshold,
        )
        filtered = _rerank_by_query_overlap(sanitized, filtered)

        context_chunks = _build_context(filtered, max_context_tokens=max_context_tokens)
        if not context_chunks:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "answer": UNKNOWN_POLICY_PHRASE,
                "fallback_used": True,
                "retrieved": [
                    {
                        "score": r.get("score"),
                        "doc_id": (r.get("payload") or {}).get("doc_id"),
                        "chunk_index": (r.get("payload") or {}).get("chunk_index"),
                        "chunk_id": (r.get("payload") or {}).get("chunk_id"),
                        "text_preview": _text_preview((r.get("payload") or {}).get("text") or ""),
                        "source_section": ((r.get("payload") or {}).get("source_section") or "").strip(),
                        "page_number": (r.get("payload") or {}).get("page_number"),
                    }
                    for r in retrieved
                ],
                "citations": [],
                "latency_ms": latency_ms,
                "top_k": top_k,
                "threshold": threshold,
                **_pipeline_fields(rag_retrieval_ran=True, pipeline="manual_rag_empty_context"),
            }

        if getattr(settings, "RAG_STRICT_NO_HALLUCINATE", True):
            max_sentences = int(getattr(settings, "RAG_STRICT_MAX_SENTENCES", 4))
            extracted_answer, used_chunks = _extractive_answer_with_sources_from_context(
                sanitized,
                context_chunks,
                max_sentences=max_sentences,
                query_embedding=query_embedding,
                force_lexical_sentence_ranking=embedding_failed,
            )
            if not extracted_answer and getattr(settings, "RAG_STRICT_FALLBACK_TO_LLM", True) and context_chunks:
                llm_tok = int(getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 1024))
                answer = generate_answer(
                    question=sanitized,
                    context_chunks=context_chunks,
                    fallback_phrase=FALLBACK_PHRASE,
                    model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
                    max_output_tokens=llm_tok,
                    temperature=float(getattr(settings, "OPENROUTER_TEMPERATURE", 0.2)),
                    prefer_answer_from_context=True,
                )
                if not answer.strip() or answer.strip() == UNKNOWN_POLICY_PHRASE:
                    answer = UNKNOWN_POLICY_PHRASE
                    context_chunks = []
                elif not _llm_draft_grounded_in_context(answer, context_chunks):
                    answer = UNKNOWN_POLICY_PHRASE
                    context_chunks = []
            elif not extracted_answer:
                answer = UNKNOWN_POLICY_PHRASE
                context_chunks = []
            else:
                answer = extracted_answer
                append_raw_answer_before_ui_processing(extracted_answer)
                context_chunks = used_chunks
        else:
            answer = generate_answer(
                question=sanitized,
                context_chunks=context_chunks,
                fallback_phrase=FALLBACK_PHRASE,
                model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
                max_output_tokens=int(getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 1024)),
                temperature=float(getattr(settings, "OPENROUTER_TEMPERATURE", 0.2)),
            )
            if not _query_is_covered_by_context(sanitized, context_chunks):
                answer = UNKNOWN_POLICY_PHRASE
            elif (
                answer.strip() == UNKNOWN_POLICY_PHRASE
                or "does not explicitly" in answer.lower()
                or _prefer_extractive_for_question(sanitized)
            ) and _should_use_extractive_fallback(sanitized, context_chunks):
                extracted = _extractive_answer_from_context(sanitized, context_chunks)
                if extracted:
                    answer = extracted
                    append_raw_answer_before_ui_processing(extracted)
    except Exception:
        logger.exception("RAG manual pipeline failed")
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "answer": FALLBACK_PHRASE,
            "fallback_used": True,
            "retrieved": [],
            "citations": [],
            "latency_ms": latency_ms,
            "top_k": top_k,
            "threshold": threshold,
            **_pipeline_fields(rag_retrieval_ran=True, pipeline="manual_rag_exception"),
        }

    latency_ms = int((time.time() - t0) * 1000)
    fallback_used = answer.strip() in {FALLBACK_PHRASE, UNKNOWN_POLICY_PHRASE}
    return {
        "answer": answer,
        "fallback_used": fallback_used,
        "retrieved": [
            {
                "score": r.get("score"),
                "doc_id": (r.get("payload") or {}).get("doc_id"),
                "chunk_index": (r.get("payload") or {}).get("chunk_index"),
                "chunk_id": (r.get("payload") or {}).get("chunk_id"),
                "text_preview": _text_preview((r.get("payload") or {}).get("text") or ""),
                "source_section": ((r.get("payload") or {}).get("source_section") or "").strip(),
                "page_number": (r.get("payload") or {}).get("page_number"),
            }
            for r in filtered[:top_k]
        ],
        "citations": [_citation_entry(c) for c in context_chunks],
        "latency_ms": latency_ms,
        "top_k": top_k,
        "threshold": threshold,
        **_pipeline_fields(rag_retrieval_ran=True, pipeline="manual_rag"),
    }


def run_rag_query(
    *,
    query: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    max_context_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Runs the RAG flow. Uses LangGraph when installed; otherwise falls back to
    a manual orchestration so the service stays functional during setup.
    """
    top_k = top_k if top_k is not None else getattr(settings, "RAG_TOP_K", 5)
    threshold = threshold if threshold is not None else getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.3)
    max_context_tokens = (
        max_context_tokens if max_context_tokens is not None else getattr(settings, "RAG_MAX_CONTEXT_TOKENS", 1600)
    )
    if _is_obvious_gibberish_query(query):
        return {
            "answer": UNKNOWN_POLICY_PHRASE,
            "fallback_used": True,
            "retrieved": [],
            "citations": [],
            "latency_ms": 0,
            "top_k": top_k,
            "threshold": threshold,
            **_pipeline_fields(rag_retrieval_ran=False, pipeline="obvious_gibberish_short_circuit"),
        }
    if _is_greeting(query):
        return {
            "answer": _greeting_answer(query),
            "fallback_used": False,
            "retrieved": [],
            "citations": [],
            "latency_ms": 0,
            "top_k": top_k,
            "threshold": threshold,
            **_pipeline_fields(rag_retrieval_ran=False, pipeline="greeting_short_circuit"),
        }
    if _is_probably_gibberish(query):
        return {
            "answer": UNKNOWN_POLICY_PHRASE,
            "fallback_used": True,
            "retrieved": [],
            "citations": [],
            "latency_ms": 0,
            "top_k": top_k,
            "threshold": threshold,
            **_pipeline_fields(rag_retrieval_ran=False, pipeline="gibberish_short_circuit"),
        }

    # In strict mode (extractive/policy mode), we want more retrieved context
    # to avoid cutting off part of the rules list.
    if getattr(settings, "RAG_STRICT_NO_HALLUCINATE", True):
        max_context_tokens = getattr(settings, "RAG_STRICT_MAX_CONTEXT_TOKENS", max_context_tokens)

    try:
        from langgraph.graph import StateGraph, END  # type: ignore
    except Exception:
        return _manual_pipeline(query=query, top_k=top_k, threshold=threshold, max_context_tokens=max_context_tokens)

    # Build nodes.
    def sanitize_node(state: RagState) -> RagState:
        return {"query": _sanitize_query(state["query"])}

    def embed_node(state: RagState) -> RagState:
        query_embedding = get_embeddings([state["query"]])[0]
        return {"query_embedding": query_embedding}

    def retrieve_node(state: RagState) -> RagState:
        qdrant = QdrantService()
        results = qdrant.search(state["query_embedding"], limit=_dense_candidate_limit(top_k)) or []
        retrieved: List[Dict[str, Any]] = []
        for r in results:
            score = r.score if hasattr(r, "score") else r.get("score")
            payload = r.payload if hasattr(r, "payload") else r.get("payload")
            retrieved.append({"score": score, "payload": payload})
        retrieved = _hybrid_merge_semantic_and_lexical(
            query=state["query"],
            qdrant=qdrant,
            semantic_items=retrieved,
            top_k=top_k,
        )
        return {"retrieved": retrieved}

    def filter_context_node(state: RagState) -> RagState:
        retrieved = state.get("retrieved") or []
        filtered = _hybrid_filter_for_context(
            retrieved,
            query=state["query"],
            top_k=top_k,
            threshold=threshold,
        )
        filtered = _rerank_by_query_overlap(state["query"], filtered)
        context_chunks = _build_context(filtered, max_context_tokens=max_context_tokens)
        if not context_chunks:
            return {"context_chunks": [], "fallback_used": True}
        return {"context_chunks": context_chunks, "fallback_used": False}

    def answer_node(state: RagState) -> RagState:
        context_chunks = state.get("context_chunks") or []
        question = state["query"]

        # Strict mode: answer is built from verbatim sentences in retrieved chunks.
        # This removes paraphrase-based hallucinations as much as possible.
        if getattr(settings, "RAG_STRICT_NO_HALLUCINATE", True):
            max_sentences = int(getattr(settings, "RAG_STRICT_MAX_SENTENCES", 4))
            extracted_answer, used_chunks = _extractive_answer_with_sources_from_context(
                question,
                context_chunks,
                max_sentences=max_sentences,
                query_embedding=state.get("query_embedding"),
            )
            if not extracted_answer and getattr(settings, "RAG_STRICT_FALLBACK_TO_LLM", True) and context_chunks:
                llm_tok = int(getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 1024))
                answer = generate_answer(
                    question=question,
                    context_chunks=context_chunks,
                    fallback_phrase=FALLBACK_PHRASE,
                    model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
                    max_output_tokens=llm_tok,
                    temperature=float(getattr(settings, "OPENROUTER_TEMPERATURE", 0.2)),
                    prefer_answer_from_context=True,
                )
                if not answer.strip() or answer.strip() == UNKNOWN_POLICY_PHRASE:
                    return {
                        "answer": UNKNOWN_POLICY_PHRASE,
                        "fallback_used": True,
                        "context_chunks": [],
                    }
                if not _llm_draft_grounded_in_context(answer, context_chunks):
                    return {
                        "answer": UNKNOWN_POLICY_PHRASE,
                        "fallback_used": True,
                        "context_chunks": [],
                    }
                return {
                    "answer": answer,
                    "fallback_used": False,
                    "context_chunks": context_chunks,
                }
            if not extracted_answer:
                return {
                    "answer": UNKNOWN_POLICY_PHRASE,
                    "fallback_used": True,
                    "context_chunks": [],
                }
            append_raw_answer_before_ui_processing(extracted_answer)
            return {
                "answer": extracted_answer,
                "fallback_used": False,
                "context_chunks": used_chunks,
            }

        answer = generate_answer(
            question=question,
            context_chunks=context_chunks,
            fallback_phrase=FALLBACK_PHRASE,
            model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
            max_output_tokens=int(getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 1024)),
            temperature=float(getattr(settings, "OPENROUTER_TEMPERATURE", 0.2)),
        )
        if not _query_is_covered_by_context(question, context_chunks):
            answer = UNKNOWN_POLICY_PHRASE
        elif (
            answer.strip() == UNKNOWN_POLICY_PHRASE
            or "does not explicitly" in answer.lower()
            or _prefer_extractive_for_question(question)
        ) and _should_use_extractive_fallback(question, context_chunks):
            extracted = _extractive_answer_from_context(question, context_chunks)
            if extracted:
                answer = extracted
                append_raw_answer_before_ui_processing(extracted)
        return {
            "answer": answer,
            "fallback_used": answer.strip() in {FALLBACK_PHRASE, UNKNOWN_POLICY_PHRASE},
        }

    def fallback_node(state: RagState) -> RagState:
        return {"answer": UNKNOWN_POLICY_PHRASE, "fallback_used": True, "context_chunks": []}

    def route_fn(state: RagState) -> str:
        return "fallback" if state.get("fallback_used") else "answer"

    graph = StateGraph(RagState)
    graph.add_node("sanitizeQuery", sanitize_node)
    graph.add_node("embedQuery", embed_node)
    graph.add_node("retrieveTopK", retrieve_node)
    graph.add_node("filterContext", filter_context_node)
    graph.add_node("answerWithContext", answer_node)
    graph.add_node("fallbackNode", fallback_node)

    graph.set_entry_point("sanitizeQuery")
    graph.add_edge("sanitizeQuery", "embedQuery")
    graph.add_edge("embedQuery", "retrieveTopK")
    graph.add_edge("retrieveTopK", "filterContext")
    graph.add_conditional_edges(
        "filterContext",
        route_fn,
        {"answer": "answerWithContext", "fallback": "fallbackNode"},
    )
    graph.add_edge("answerWithContext", END)
    graph.add_edge("fallbackNode", END)

    initial: RagState = {"query": query}
    t0 = time.time()
    try:
        compiled_graph = graph.compile()  # type: ignore[attr-defined]
        final_state: RagState = compiled_graph.invoke(initial)  # type: ignore
        latency_ms = int((time.time() - t0) * 1000)
    except Exception:
        return _manual_pipeline(query=query, top_k=top_k, threshold=threshold, max_context_tokens=max_context_tokens)

    retrieved_summary = []
    for r in (final_state.get("retrieved") or []):
        payload = r.get("payload") or {}
        retrieved_summary.append(
            {
                "score": r.get("score"),
                "doc_id": payload.get("doc_id"),
                "chunk_index": payload.get("chunk_index"),
                "chunk_id": payload.get("chunk_id"),
                "text_preview": _text_preview(payload.get("text") or ""),
                "source_section": (payload.get("source_section") or "").strip(),
                "page_number": payload.get("page_number"),
            }
        )

    citations = [_citation_entry(c) for c in (final_state.get("context_chunks") or [])]

    fchunks = final_state.get("context_chunks") or []
    pipeline = "langgraph_rag" if fchunks else "langgraph_empty_context"

    return {
        "answer": final_state.get("answer") or FALLBACK_PHRASE,
        "fallback_used": bool(final_state.get("fallback_used")),
        "retrieved": retrieved_summary,
        "citations": citations,
        "latency_ms": latency_ms,
        "top_k": top_k,
        "threshold": threshold,
        **_pipeline_fields(rag_retrieval_ran=True, pipeline=pipeline),
    }

