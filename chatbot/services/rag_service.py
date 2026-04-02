from __future__ import annotations

import time
import re
import math
import difflib
from typing import Any, Dict, List, Optional, TypedDict

from django.conf import settings

from chatbot.services.embeddings_service import get_embeddings
from chatbot.services.gemini_service import UNKNOWN_POLICY_PHRASE, generate_answer
from chatbot.services.qdrant_service import QdrantService
from chatbot.services.text_splitter import count_tokens

FALLBACK_PHRASE = "Contact the HR department"
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
}
_GENERIC_QUERY_TERMS = {
    "acme",
    "company",
    "consultants",
    "document",
    "employee",
    "employees",
    "policy",
    "policies",
    "rule",
    "rules",
    "using",
    "use",
    "work",
    "working",
    "specific",
    "specifically",
    "name",
    "current",
    "activities",
    "network",
    "information",
    "mentioned",
    "present",
    "about",
}
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

    def fix_token(tok: str) -> str:
        lower = tok.lower()
        # Avoid over-correcting short tokens and already-known words.
        if len(lower) < 4 or lower in _TYPO_CANONICAL_TERMS:
            return tok
        if not lower.isalpha():
            return tok
        match = difflib.get_close_matches(lower, _TYPO_CANONICAL_TERMS, n=1, cutoff=0.86)
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
    greetings = (
        "hi",
        "hello",
        "hey",
        "salam",
        "assalam",
        "assalam-o-alaikum",
        "assalamualaikum",
        "aoa",
    )
    return any(g in q for g in greetings)


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
    if any(k in q for k in ("assalam", "assalam-o-alaikum", "assalamualaikum", "salam", "aoa")):
        return "waalaikum assalam"
    if "hello" in q:
        return "hello"
    if "hi" in q:
        return "hi"
    return "hello"


def _build_context(
    retrieved: List[Dict[str, Any]],
    *,
    max_context_tokens: int,
) -> List[Dict[str, Any]]:
    context: List[Dict[str, Any]] = []
    running_tokens = 0
    for r in retrieved:
        payload = r.get("payload") or {}
        text = payload.get("text") or ""
        if not text:
            continue

        token_count = payload.get("token_count")
        if token_count is None:
            token_count = count_tokens(text)

        # Hard cap: stop adding more once max is reached.
        if running_tokens + token_count > max_context_tokens and context:
            break

        context.append(
            {
                "text": text,
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": token_count,
                "score": r.get("score"),
            }
        )
        running_tokens += token_count

    return context


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


def _anchor_terms(query: str) -> set[str]:
    terms = _tokenize_keywords(query)
    # Allow shorter topic tokens (e.g. "gym") to participate in grounding,
    # while still excluding generic query terms and stopwords.
    anchors = {t for t in terms if t not in _GENERIC_QUERY_TERMS and len(t) >= 3}
    return anchors if anchors else terms


def _is_probably_gibberish(query: str) -> bool:
    """
    Conservative gibberish detector to avoid answering nonsense with unrelated policy text.
    """
    q = _sanitize_query(query)
    if not q:
        return False

    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", q)]
    meaningful = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]
    if not meaningful:
        return False

    def known_like(tok: str) -> bool:
        if tok in _TYPO_CANONICAL_TERMS:
            return True
        return bool(difflib.get_close_matches(tok, _TYPO_CANONICAL_TERMS, n=1, cutoff=0.84))

    known_count = sum(1 for t in meaningful if known_like(t))
    if known_count > 0:
        return False

    # If all meaningful tokens are unknown and at least one is long/noisy, treat as gibberish.
    noisy_long = any(len(t) >= 8 for t in meaningful)
    repeated_noise = any(re.search(r"(.)\1\1", t) for t in meaningful)
    return noisy_long or repeated_noise


def _rerank_by_query_overlap(query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple lexical reranker to keep chunks that mention query terms closer to the top.
    This improves precision for policy-style questions where exact phrasing matters.
    """
    q_terms = _tokenize_keywords(query)
    anchors = _anchor_terms(query)
    if not q_terms:
        return items

    def rank_key(item: Dict[str, Any]) -> tuple[int, int, int, float]:
        payload = item.get("payload") or {}
        text = str(payload.get("text") or "")
        text_terms = _tokenize_keywords(text)
        overlap = len(q_terms.intersection(text_terms))
        anchor_overlap = len(anchors.intersection(text_terms))
        phrase_hits = sum(1 for a in anchors if a in text.lower())
        score = float(item.get("score") or 0.0)
        return (anchor_overlap, overlap, phrase_hits, score)

    return sorted(items, key=rank_key, reverse=True)


def _max_overlap(query: str, items: List[Dict[str, Any]]) -> int:
    anchors = _anchor_terms(query)
    if not anchors:
        return 0
    best = 0
    for item in items:
        payload = item.get("payload") or {}
        text = str(payload.get("text") or "")
        overlap = len(anchors.intersection(_tokenize_keywords(text)))
        if overlap > best:
            best = overlap
    return best


def _augment_with_lexical_candidates(
    *,
    query: str,
    retrieved: List[Dict[str, Any]],
    qdrant: QdrantService,
    top_k: int,
) -> List[Dict[str, Any]]:
    """
    Hybrid fallback: when semantic recall misses exact policy phrasing,
    scan payload text and inject high-overlap chunks.
    """
    anchors = _anchor_terms(query)
    if not anchors:
        return retrieved

    # Skip expensive scroll when semantic hits already cover query anchors.
    mo = _max_overlap(query, retrieved)
    need = 1 if len(anchors) == 1 else min(2, len(anchors))
    if mo >= need:
        return retrieved

    scan_cap = int(getattr(settings, "RAG_LEXICAL_SCAN_LIMIT", 150))
    scanned = qdrant.scan_payload_points(limit=max(min(scan_cap, top_k * 25), 80))
    lexical_candidates: List[Dict[str, Any]] = []
    for p in scanned:
        payload = p.payload if hasattr(p, "payload") else p.get("payload")
        if not payload or not payload.get("text"):
            continue
        text_terms = _tokenize_keywords(str(payload.get("text") or ""))
        overlap = len(anchors.intersection(text_terms))
        if overlap <= 0:
            continue
        lexical_candidates.append(
            {
                # Give lexical candidates a non-zero score so they can survive thresholding.
                "score": 0.15 + (0.05 * overlap),
                "payload": payload,
                "_lexical": True,
                "_overlap": overlap,
            }
        )

    lexical_ranked = _rerank_by_query_overlap(query, lexical_candidates)
    boosted = lexical_ranked[: max(top_k * 8, 40)]

    seen = {
        (r.get("payload") or {}).get("chunk_id")
        for r in retrieved
        if (r.get("payload") or {}).get("chunk_id") is not None
    }
    merged = list(retrieved)
    for c in boosted:
        chunk_id = (c.get("payload") or {}).get("chunk_id")
        if chunk_id is not None and chunk_id in seen:
            continue
        merged.append(c)
    return merged


def _extractive_answer_from_context(question: str, context_chunks: List[Dict[str, Any]]) -> str:
    """
    Deterministic fallback: extract relevant sentences directly from retrieved context.
    Prevents false "I don't know" on policy questions where text is present.
    """
    q_terms = _tokenize_keywords(question)
    if not q_terms:
        return ""

    candidates: List[tuple[int, str]] = []
    for c in context_chunks:
        text = str(c.get("text") or "")
        if not text:
            continue
        # Coarse sentence split works well enough for policy prose.
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
        for p in parts:
            sentence = p.strip()
            if len(sentence) < 20:
                continue
            overlap = len(q_terms.intersection(_tokenize_keywords(sentence)))
            if overlap > 0:
                candidates.append((overlap, sentence))

    if not candidates:
        return ""

    candidates.sort(key=lambda x: x[0], reverse=True)
    top = []
    seen = set()
    for _, sentence in candidates:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        top.append(sentence)
        if len(top) >= 4:
            break

    if not top:
        return ""
    return "Based on the policy:\n- " + "\n- ".join(top)


def _extract_page_number(text: str) -> Optional[int]:
    """
    Extracts a page number only when it's explicitly present in the chunk text.
    """
    if not text:
        return None
    m = re.search(r"\bpage\s*[:#]?\s*(\d{1,5})\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_section_title(text: str) -> Optional[str]:
    """
    Extracts a section title only using explicit nearby formatting in the chunk.
    Falls back to None when the title can't be identified reliably.
    """
    if not text:
        return None

    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None

    page_idx = None
    page_re = re.compile(r"\bpage\s*[:#]?\s*(\d{1,5})\b", flags=re.IGNORECASE)
    for i, ln in enumerate(lines):
        if page_re.search(ln):
            page_idx = i
            break

    if page_idx is not None and page_idx > 0:
        # Heuristic: the line immediately before an explicit "Page X" is often the section title.
        candidate = lines[page_idx - 1]
        if 3 <= len(candidate) <= 120 and not candidate.lower().startswith("page"):
            # Avoid sentence-like lines.
            if not candidate.endswith("."):
                return candidate

    # Section-number style headings (e.g. "3.6 small", "14.7 Purpose", "14.12 Payment Method").
    m = re.search(
        r"\b(\d+(?:\.\d+)+)\s+([A-Za-z][^\n]{0,80})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        num = m.group(1).strip()
        title_tail = m.group(2).strip()
        # Keep it short and avoid trailing sentence fragments.
        title_tail = re.sub(r"[,:;\.].*$", "", title_tail).strip()
        if title_tail:
            return f"{num} {title_tail}".strip()

    # Also support explicit patterns like "Section: Employee Wellness Benefits".
    m = re.search(
        r"\b(?:section|policy|chapter)\s*[:\-]\s*([^\n]{3,120})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
        if candidate and 3 <= len(candidate) <= 120:
            return candidate

    return None


def _extractive_answer_with_sources_from_context(
    question: str,
    context_chunks: List[Dict[str, Any]],
    *,
    max_sentences: int = 4,
    query_embedding: Optional[List[float]] = None,
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Strict, non-hallucinating mode:
    - Selects verbatim sentences from retrieved chunks.
    - Includes "Source section" and "Page" only if explicitly detectable in the chunk text.
    Returns (answer_text, used_context_chunks).
    """
    # Use anchor terms for relevance (not just any tokens), so intent/meta words
    # like "tell/explain/how" and generic words like "policy/rules" don't trigger
    # sentence selection for unrelated queries.
    q_terms = _anchor_terms(question)
    if not q_terms:
        return "", []

    # Require a bit more overlap when the question contains many anchor terms.
    # This reduces "random single-word matches" that can otherwise return
    # unrelated sentences for illogical inputs.
    min_overlap = int(getattr(settings, "RAG_STRICT_MIN_SENTENCE_OVERLAP", 1))
    if len(q_terms) >= 4:
        min_overlap = max(min_overlap, 2)

    sentences_per_chunk = int(getattr(settings, "RAG_STRICT_SENTENCES_PER_CHUNK", 6))
    max_candidate_sentences = int(getattr(settings, "RAG_STRICT_MAX_CANDIDATE_SENTENCES", 30))

    # Candidate extraction: still verbatim sentences, but we may choose them semantically.
    candidates: List[tuple[float, int, str, Dict[str, Any]]] = []  # (overlap, sentence_len, sentence, chunk_meta)
    for c in context_chunks:
        text = str(c.get("text") or "")
        if not text:
            continue
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
        chunk_candidates: List[tuple[int, int, str]] = []  # (overlap, sentence_len, sentence)
        for p in parts:
            sentence = p.strip()
            if len(sentence) < 20:
                continue
            overlap = len(q_terms.intersection(_tokenize_keywords(sentence)))
            if overlap < min_overlap:
                continue
            chunk_candidates.append((overlap, len(sentence), sentence))

        # Keep the best candidates per chunk to control latency.
        chunk_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        for overlap, sentence_len, sentence in chunk_candidates[:sentences_per_chunk]:
            candidates.append((float(overlap), sentence_len, sentence, c))

    if not candidates:
        return "", []

    # Hard cap on candidates to bound embedding cost.
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    candidates = candidates[:max_candidate_sentences]

    # Compute query embedding if not provided.
    if query_embedding is None:
        query_embedding = get_embeddings([question])[0]

    q_norm = math.sqrt(sum(v * v for v in (query_embedding or [])))

    sentence_texts = [c[2] for c in candidates]
    sentence_embeddings = get_embeddings(sentence_texts, batch_size=int(getattr(settings, "EMBEDDING_BATCH_SIZE", 64)))

    # Cosine similarity; if we can't compute norms (e.g., zero vectors), fall back to overlap.
    sim_threshold = float(getattr(settings, "RAG_STRICT_SENTENCE_SIM_THRESHOLD", 0.15))

    scored: List[tuple[float, int, str, Dict[str, Any]]] = []
    for (overlap, sentence_len, sentence, chunk_meta), sent_emb in zip(candidates, sentence_embeddings):
        s_norm = math.sqrt(sum(v * v for v in (sent_emb or [])))
        if not q_norm or not s_norm:
            similarity = float(overlap)
        else:
            dot = sum(qe * se for qe, se in zip(query_embedding, sent_emb))
            similarity = dot / (q_norm * s_norm)
        scored.append((similarity, sentence_len, sentence, chunk_meta))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    selected_sentences: List[str] = []
    selected_sentence_keys: set[str] = set()
    selected_chunks: List[Dict[str, Any]] = []
    selected_chunk_ids: set[str] = set()

    # First pass: similarity threshold.
    for sim, _, sentence, chunk_meta in scored:
        if sim < sim_threshold:
            continue
        key = sentence.lower()
        if key in selected_sentence_keys:
            continue
        selected_sentence_keys.add(key)
        selected_sentences.append(sentence)
        chunk_id = str(chunk_meta.get("chunk_id") or "")
        if chunk_id and chunk_id not in selected_chunk_ids:
            selected_chunk_ids.add(chunk_id)
            selected_chunks.append(chunk_meta)
        if len(selected_sentences) >= max_sentences:
            break

    # If nothing meets the semantic threshold, do not "best-effort" select random
    # overlapping sentences. In a strict HR bot, returning UNKNOWN is safer.
    if not selected_sentences:
        return "", []

    # Pick section/page from whichever selected chunk actually contains it.
    section: Optional[str] = None
    page: Optional[int] = None
    for chunk_meta in selected_chunks:
        chunk_text = str(chunk_meta.get("text") or "")
        if section is None:
            cand = _extract_section_title(chunk_text)
            if cand:
                section = cand
        if page is None:
            cand_page = _extract_page_number(chunk_text)
            if cand_page is not None:
                page = cand_page
        if section is not None and page is not None:
            break
    section_str = section or "Unknown"
    page_str = str(page) if page is not None else "Unknown"

    # Human-readable intro + answer formatting.
    q_terms_for_intro = list(_anchor_terms(question))
    topic_hint = " ".join(q_terms_for_intro[:4]).strip() or "your question"

    cleaned_sentences: List[str] = []
    seen_sentence_keys: set[str] = set()
    for s in selected_sentences:
        t = re.sub(r"\s+", " ", (s or "").strip())
        t = t.strip(" -•*")
        if not t:
            continue
        key = t.lower()
        if key in seen_sentence_keys:
            continue
        seen_sentence_keys.add(key)
        cleaned_sentences.append(t)

    # Keep response concise and readable.
    answer_paragraph = " ".join(cleaned_sentences[:3]).strip()
    if answer_paragraph and not answer_paragraph.endswith("."):
        answer_paragraph += "."

    answer_lines: List[str] = [
        f"Here is what I found about {topic_hint}.",
        "",
        "### Answer",
        "",
        answer_paragraph,
        "",
        "### Source Section",
        section_str,
        "",
        "### Page Number",
        page_str,
        "",
        "If you have any further questions, do let me know.",
    ]

    return "\n".join(answer_lines).strip(), selected_chunks


def _should_use_extractive_fallback(question: str, context_chunks: List[Dict[str, Any]]) -> bool:
    """
    Use extractive fallback only when key terms from the question
    are actually present in retrieved context.
    """
    anchors = _anchor_terms(question)
    if not anchors:
        return False

    context_terms: set[str] = set()
    for c in context_chunks:
        context_terms |= _tokenize_keywords(str(c.get("text") or ""))

    overlap = len(anchors.intersection(context_terms))
    return overlap >= 1


def _query_is_covered_by_context(question: str, context_chunks: List[Dict[str, Any]]) -> bool:
    anchors = _anchor_terms(question)
    if not anchors:
        return False
    q_lower = question.lower()
    strict_factoid = any(
        k in q_lower for k in ("name of", "who is", "who's", "current", "specifically", "exact")
    )

    context_terms: set[str] = set()
    for c in context_chunks:
        context_terms |= _tokenize_keywords(str(c.get("text") or ""))
    overlap = len(anchors.intersection(context_terms))
    anchor_count = len(anchors)
    if strict_factoid:
        # For specific factoid asks, require full anchor coverage.
        return overlap == anchor_count
    if anchor_count == 1:
        return overlap == 1
    # Require stronger evidence for multi-term queries to avoid false positives.
    return overlap >= 2 and (overlap / anchor_count) >= 0.5


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
        query_embedding = get_embeddings([sanitized])[0]

        # Pull a wider semantic candidate set, then rerank by query overlap.
        candidate_limit = max(top_k * 8, 40)
        search_results = qdrant.search(query_embedding, limit=candidate_limit) or []

        # Qdrant score: higher is more similar (depending on distance metric).
        retrieved: List[Dict[str, Any]] = []
        for r in search_results:
            score = r.score if hasattr(r, "score") else r.get("score")
            payload = r.payload if hasattr(r, "payload") else r.get("payload")
            retrieved.append({"score": score, "payload": payload})

        retrieved = _augment_with_lexical_candidates(
            query=sanitized,
            retrieved=retrieved,
            qdrant=qdrant,
            top_k=top_k,
        )
        filtered = [
            r
            for r in retrieved
            if (
                r.get("_lexical")
                or (r.get("score") is not None and r["score"] >= threshold)
            )
        ]
        if not filtered:
            # Keep lexically-boosted candidates even if they use neutral fallback score.
            filtered = retrieved[: max(top_k * 4, 20)]
        filtered = _rerank_by_query_overlap(sanitized, filtered)

        context_chunks = _build_context(filtered, max_context_tokens=max_context_tokens)
        if not context_chunks:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "answer": FALLBACK_PHRASE,
                "fallback_used": True,
                "retrieved": [
                    {
                        "score": r.get("score"),
                        "doc_id": (r.get("payload") or {}).get("doc_id"),
                        "chunk_index": (r.get("payload") or {}).get("chunk_index"),
                        "chunk_id": (r.get("payload") or {}).get("chunk_id"),
                        "text_preview": _text_preview((r.get("payload") or {}).get("text") or ""),
                    }
                    for r in retrieved
                ],
                "citations": [],
                "latency_ms": latency_ms,
                "top_k": top_k,
                "threshold": threshold,
            }

        if getattr(settings, "RAG_STRICT_NO_HALLUCINATE", True):
            max_sentences = int(getattr(settings, "RAG_STRICT_MAX_SENTENCES", 4))
            extracted_answer, used_chunks = _extractive_answer_with_sources_from_context(
                sanitized,
                context_chunks,
                max_sentences=max_sentences,
                query_embedding=query_embedding,
            )
            if not extracted_answer:
                answer = UNKNOWN_POLICY_PHRASE
                context_chunks = []
            else:
                answer = extracted_answer
                context_chunks = used_chunks
        else:
            answer = generate_answer(
                question=sanitized,
                context_chunks=context_chunks,
                fallback_phrase=FALLBACK_PHRASE,
                model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
                max_output_tokens=getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 512),
                temperature=getattr(settings, "OPENROUTER_TEMPERATURE", 0.2),
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
    except Exception:
        # TEMP: log the real error so we can debug why RAG fell back.
        import traceback

        traceback.print_exc()
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "answer": FALLBACK_PHRASE,
            "fallback_used": True,
            "retrieved": [],
            "citations": [],
            "latency_ms": latency_ms,
            "top_k": top_k,
            "threshold": threshold,
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
            }
            for r in filtered[:top_k]
        ],
        "citations": [
            {
                "doc_id": c.get("doc_id"),
                "chunk_index": c.get("chunk_index"),
                "chunk_id": c.get("chunk_id"),
                "text_preview": _text_preview(c.get("text") or ""),
                "token_count": c.get("token_count"),
                "source_section": _extract_section_title(c.get("text") or ""),
                "page": _extract_page_number(c.get("text") or ""),
            }
            for c in context_chunks
        ],
        "latency_ms": latency_ms,
        "top_k": top_k,
        "threshold": threshold,
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
    if _is_greeting(query):
        return {
            "answer": _greeting_answer(query),
            "fallback_used": False,
            "retrieved": [],
            "citations": [],
            "latency_ms": 0,
            "top_k": top_k,
            "threshold": threshold,
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
        }

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
        candidate_limit = max(top_k * 8, 40)
        results = qdrant.search(state["query_embedding"], limit=candidate_limit) or []
        retrieved: List[Dict[str, Any]] = []
        for r in results:
            score = r.score if hasattr(r, "score") else r.get("score")
            payload = r.payload if hasattr(r, "payload") else r.get("payload")
            retrieved.append({"score": score, "payload": payload})
        retrieved = _augment_with_lexical_candidates(
            query=state["query"],
            retrieved=retrieved,
            qdrant=qdrant,
            top_k=top_k,
        )
        return {"retrieved": retrieved}

    def filter_context_node(state: RagState) -> RagState:
        retrieved = state.get("retrieved") or []
        filtered = [
            r
            for r in retrieved
            if (
                r.get("_lexical")
                or (r.get("score") is not None and r["score"] >= threshold)
            )
        ]
        if not filtered:
            filtered = retrieved[: max(top_k * 4, 20)]
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
            if not extracted_answer:
                return {
                    "answer": UNKNOWN_POLICY_PHRASE,
                    "fallback_used": True,
                    "context_chunks": [],
                }
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
            max_output_tokens=getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 512),
            temperature=getattr(settings, "OPENROUTER_TEMPERATURE", 0.2),
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
        return {
            "answer": answer,
            "fallback_used": answer.strip() in {FALLBACK_PHRASE, UNKNOWN_POLICY_PHRASE},
        }

    def fallback_node(state: RagState) -> RagState:
        return {"answer": FALLBACK_PHRASE, "fallback_used": True, "context_chunks": []}

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
            }
        )

    citations = []
    for c in (final_state.get("context_chunks") or []):
        chunk_text = c.get("text") or ""
        citations.append(
            {
                "doc_id": c.get("doc_id"),
                "chunk_index": c.get("chunk_index"),
                "chunk_id": c.get("chunk_id"),
                "text_preview": _text_preview(c.get("text") or ""),
                "token_count": c.get("token_count"),
                "source_section": _extract_section_title(chunk_text),
                "page": _extract_page_number(chunk_text),
            }
        )

    return {
        "answer": final_state.get("answer") or FALLBACK_PHRASE,
        "fallback_used": bool(final_state.get("fallback_used")),
        "retrieved": retrieved_summary,
        "citations": citations,
        "latency_ms": latency_ms,
        "top_k": top_k,
        "threshold": threshold,
    }

