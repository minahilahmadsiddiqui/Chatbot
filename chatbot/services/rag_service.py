from __future__ import annotations

import difflib
import re
import time
from typing import Any, Dict, List, Optional

from django.conf import settings

from chatbot.services.cross_encoder_service import rerank_with_cross_encoder
from chatbot.services.embeddings_service import get_embeddings
from chatbot.services.gemini_service import UNKNOWN_POLICY_PHRASE, generate_answer
from chatbot.services.qdrant_service import QdrantService
from chatbot.services.text_splitter import count_tokens, truncate_text_to_token_budget

FALLBACK_PHRASE = UNKNOWN_POLICY_PHRASE
_HANDBOOK_ASSISTANT_GREETING = (
    "Hello! I can help you find information in the employee handbook. What would you like to know?"
)
def _append_rag_llm_input_log(
    *,
    query: str,
    query_vector: List[float],
    selected: List[Dict[str, Any]],
) -> None:
    del query, query_vector, selected
    return


def _pipeline_fields(*, rag_retrieval_ran: bool, pipeline: str) -> Dict[str, Any]:
    return {"rag_retrieval_ran": rag_retrieval_ran, "pipeline": pipeline}


def _sanitize_query(query: str) -> str:
    q = (query or "").strip()
    if len(q) > 4000:
        q = q[:4000]
    return q


def _is_obvious_gibberish_query(query: str) -> bool:
    raw = (query or "").strip()
    if not raw:
        return False
    letters = re.sub(r"[^a-zA-Z]", "", raw).lower()
    if not letters:
        return len(re.sub(r"\s+", "", raw)) >= 4
    vowels = set("aeiouy")
    if len(letters) >= 4 and not any(c in vowels for c in letters):
        return True
    if len(letters) >= 12:
        ratio = sum(1 for c in letters if c in vowels) / max(len(letters), 1)
        if ratio < 0.1:
            return True
    if re.search(r"(.)\1{4,}", letters):
        return True
    return False


def _is_greeting(query: str) -> bool:
    q = (query or "").lower().strip()
    qn = re.sub(r"[^a-z0-9\s]", "", q)
    qn = re.sub(r"\s+", " ", qn).strip()
    if re.fullmatch(r"(hi|hello|hey)(\s+(there|team|all|everyone))?", qn):
        return True
    if re.search(r"\bhow\s*(are|re|r)\s*(you|u)\b", qn):
        return True
    if difflib.SequenceMatcher(None, qn, "how are you").ratio() >= 0.74:
        return True
    if re.search(r"\b(i['?]?m|i am)\s+(fine|good|well|okay|ok)\b", q):
        return True
    return False


def _greeting_answer(query: str) -> str:
    q = re.sub(r"\s+", " ", (query or "").lower()).strip()
    qn = re.sub(r"[^a-z0-9\s]", "", q)
    qn = re.sub(r"\s+", " ", qn).strip()
    # Treat exact/clean variants as correct phrasing.
    if qn == "how are you":
        return "I am doing well, thank you. How can I help you?"
    # Typo-ish variants still match via fuzzy/regex greeting detection.
    if re.search(r"\bhow\s*(are|re|r)\s*(you|u)\b", qn):
        return "I'm good, looks like you meant \"How are you?\" How can I help you?"
    if difflib.SequenceMatcher(None, qn, "how are you").ratio() >= 0.74:
        return "I'm good, looks like you meant \"How are you?\" How can I help you?"
    if re.search(r"\b(i['?]?m|i am)\s+(fine|good|well|okay|ok)\b", q):
        return "Glad to hear it. What can I help you find in the employee handbook?"
    return _HANDBOOK_ASSISTANT_GREETING


def _text_preview(text: str, *, max_chars: int = 250) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rstrip() + "..."


def _build_context(retrieved: List[Dict[str, Any]], *, max_context_tokens: int) -> List[Dict[str, Any]]:
    context: List[Dict[str, Any]] = []
    running = 0
    for r in retrieved:
        payload = r.get("payload") or {}
        text = str(payload.get("text") or "").strip()
        if not text:
            continue
        tok = payload.get("token_count")
        try:
            tok = int(tok) if tok is not None else count_tokens(text)
        except Exception:
            tok = count_tokens(text)
        room = max_context_tokens - running
        if room <= 0:
            break
        use_text = text
        use_tok = tok
        if use_tok > room:
            use_text = truncate_text_to_token_budget(text, room)
            use_tok = count_tokens(use_text)
            if use_tok <= 0:
                continue
        context.append(
            {
                "text": use_text,
                "doc_id": payload.get("doc_id"),
                "chunk_id": payload.get("chunk_id"),
                "chunk_index": payload.get("chunk_index"),
                "token_count": use_tok,
                "score": r.get("score"),
                "source_section": (payload.get("source_section") or "").strip(),
                "page_number": payload.get("page_number"),
            }
        )
        running += use_tok
    return context


def _citation_entry(chunk: Dict[str, Any]) -> Dict[str, Any]:
    page = chunk.get("page_number")
    try:
        page = int(page) if page is not None else None
    except Exception:
        page = None
    sec = (chunk.get("source_section") or "").strip() or None
    return {
        "doc_id": chunk.get("doc_id"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_id": chunk.get("chunk_id"),
        "text_preview": _text_preview(chunk.get("text") or ""),
        "token_count": chunk.get("token_count"),
        "source_section": sec,
        "page": page,
    }


def _section_label_plausible_for_chunk(sec: str, chunk_text: str) -> bool:
    sec = (sec or "").strip()
    if not sec:
        return False
    if re.search(r"\.{4,}", sec):
        return False
    text = chunk_text or ""
    sec_l = sec.lower()
    found_clean = False
    found_toc = False
    for ln in text.splitlines():
        st = ln.strip()
        if not st:
            continue
        if sec not in st and sec_l not in st.lower():
            continue
        if re.search(r"\.{4,}", st):
            found_toc = True
        else:
            found_clean = True
    if found_clean:
        return True
    if found_toc:
        return False
    return True


def _extractive_answer_with_sources_from_context(
    question: str,
    context_chunks: List[Dict[str, Any]],
    *,
    max_sentences: int = 4,
    query_embedding: Optional[List[float]] = None,
    force_lexical_sentence_ranking: bool = False,
) -> tuple[str, List[Dict[str, Any]]]:
    del query_embedding, force_lexical_sentence_ranking
    q_terms = {
        w
        for w in re.findall(r"[a-zA-Z0-9]+", (question or "").lower())
        if len(w) > 2
    }
    sentences: List[str] = []
    used: List[Dict[str, Any]] = []
    for c in context_chunks:
        text = str(c.get("text") or "")
        if not text:
            continue
        matched_here = False
        for seg in re.split(r"(?<=[.!?])\s+|\n+", text):
            s = seg.strip()
            if len(s) < 20:
                continue
            if q_terms:
                s_terms = {
                    w for w in re.findall(r"[a-zA-Z0-9]+", s.lower()) if len(w) > 2
                }
                if not q_terms.intersection(s_terms):
                    continue
            sentences.append(s)
            matched_here = True
            if max_sentences > 0 and len(sentences) >= max_sentences:
                break
        if matched_here:
            used.append(c)
        if max_sentences > 0 and len(sentences) >= max_sentences:
            break
    if not sentences:
        return "", []
    answer = " ".join(sentences).strip()
    if answer and not answer.endswith("."):
        answer += "."
    return answer, used


def _cross_encoder_relevance_diagnostics(
    *,
    query: str,
    selected: List[Dict[str, Any]],
    threshold: float,
) -> Dict[str, Any]:
    """
    Industry-style abstention: learned relevance model (cross-encoder) with
    confidence threshold + ambiguity margin.
    """
    top_dense = float(selected[0].get("score") or 0.0) if selected else 0.0
    out: Dict[str, Any] = {
        "top_dense_score": top_dense,
        "dense_threshold": float(threshold),
        "dense_pass": bool(selected) and top_dense >= float(threshold),
    }
    if not selected:
        out.update(
            {
                "ce_available": False,
                "ce_applied": False,
                "ce_top_score": None,
                "ce_second_score": None,
                "ce_margin": None,
                "ce_pass": False,
            }
        )
        return out

    ce_top = selected[0].get("_ce_score")
    if ce_top is None:
        out.update(
            {
                "ce_available": False,
                "ce_applied": False,
                "ce_top_score": None,
                "ce_second_score": None,
                "ce_margin": None,
                "ce_pass": None,
            }
        )
        return out

    top = float(ce_top)
    second = float(selected[1].get("_ce_score") or 0.0) if len(selected) > 1 else 0.0
    margin = top - second
    min_ce = float(getattr(settings, "RAG_CE_MIN_RELEVANCE", 0.45))
    min_margin = float(getattr(settings, "RAG_CE_MIN_MARGIN", 0.03))
    ce_pass = top >= min_ce and margin >= min_margin

    out.update(
        {
            "ce_available": True,
            "ce_applied": True,
            "ce_top_score": top,
            "ce_second_score": second,
            "ce_margin": margin,
            "ce_min_relevance": min_ce,
            "ce_min_margin": min_margin,
            "ce_pass": ce_pass,
        }
    )
    return out


def run_rag_query(
    *,
    query: str,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
    max_context_tokens: Optional[int] = None,
    session_id: str = "",
) -> Dict[str, Any]:
    del session_id
    top_k = int(top_k if top_k is not None else getattr(settings, "RAG_TOP_K", 5))
    threshold = float(
        threshold if threshold is not None else getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.15)
    )
    # Never allow client-provided threshold to be weaker than server default.
    threshold = max(threshold, float(getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.15)))
    max_context_tokens = int(
        max_context_tokens
        if max_context_tokens is not None
        else getattr(settings, "RAG_MAX_CONTEXT_TOKENS", 3200)
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
            "retrieval_diagnostics": {},
            "sentence_evidence": [],
            "ab_variant": "control",
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
            "retrieval_diagnostics": {},
            "sentence_evidence": [],
            "ab_variant": "control",
            **_pipeline_fields(rag_retrieval_ran=False, pipeline="greeting_short_circuit"),
        }

    t0 = time.time()
    try:
        sanitized = _sanitize_query(query)
        query_embedding = get_embeddings([sanitized])[0]
        qdrant = QdrantService()
        candidate_limit = max(top_k * 4, top_k)
        raw = qdrant.search(query_embedding, limit=candidate_limit) or []

        retrieved: List[Dict[str, Any]] = []
        for item in raw:
            score = item.score if hasattr(item, "score") else item.get("score")
            payload = item.payload if hasattr(item, "payload") else item.get("payload")
            retrieved.append({"score": score, "payload": payload})

        retrieved.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        strong = [r for r in retrieved if float(r.get("score") or 0.0) >= threshold]
        selected = (strong if strong else retrieved)[:top_k]
        if bool(getattr(settings, "RAG_ENABLE_CROSS_ENCODER_RERANK", False)):
            selected = rerank_with_cross_encoder(
                query=sanitized,
                items=selected,
                top_n=int(getattr(settings, "RAG_CROSS_ENCODER_TOP_N", max(1, len(selected)))),
            )
        context_chunks = _build_context(selected, max_context_tokens=max_context_tokens)

        if not context_chunks:
            latency_ms = int((time.time() - t0) * 1000)
            return {
                "answer": UNKNOWN_POLICY_PHRASE,
                "fallback_used": True,
                "retrieved": [],
                "citations": [],
                "latency_ms": latency_ms,
                "top_k": top_k,
                "threshold": threshold,
                "retrieval_diagnostics": {
                    "retrieved_candidates": len(retrieved),
                    "above_threshold": len(strong),
                },
                "sentence_evidence": [],
                "ab_variant": "control",
                **_pipeline_fields(rag_retrieval_ran=True, pipeline="dense_rag_empty_context"),
            }

        relevance_diag = _cross_encoder_relevance_diagnostics(
            query=sanitized,
            selected=selected,
            threshold=threshold,
        )
        ce_enabled = bool(getattr(settings, "RAG_ENABLE_CROSS_ENCODER_RERANK", False))
        gate_by_dense = not bool(relevance_diag.get("dense_pass"))
        gate_by_cross_encoder = ce_enabled and relevance_diag.get("ce_pass") is False
        if gate_by_dense or gate_by_cross_encoder:
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
                    for r in selected
                ],
                "citations": [_citation_entry(c) for c in context_chunks],
                "latency_ms": latency_ms,
                "top_k": top_k,
                "threshold": threshold,
                "retrieval_diagnostics": {
                    "retrieved_candidates": len(retrieved),
                    "above_threshold": len(strong),
                    "selected_for_context": len(selected),
                    "abstained_by_relevance_gate": True,
                    "gate_by_dense": gate_by_dense,
                    "gate_by_cross_encoder": gate_by_cross_encoder,
                    **relevance_diag,
                },
                "sentence_evidence": [],
                "ab_variant": "control",
                **_pipeline_fields(rag_retrieval_ran=True, pipeline="dense_rag_abstain_relevance"),
            }

        extractive_answer, _extractive_used = _extractive_answer_with_sources_from_context(
            sanitized,
            context_chunks,
            max_sentences=int(getattr(settings, "RAG_STRICT_MAX_SENTENCES", 5)),
            query_embedding=query_embedding,
        )

        llm_error: Optional[str] = None
        try:
            _append_rag_llm_input_log(
                query=sanitized,
                query_vector=query_embedding,
                selected=selected,
            )
            answer = generate_answer(
                question=sanitized,
                context_chunks=context_chunks,
                fallback_phrase=FALLBACK_PHRASE,
                model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
                max_output_tokens=int(getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 1024)),
                temperature=float(getattr(settings, "OPENROUTER_TEMPERATURE", 0.3)),
                prefer_answer_from_context=True,
            ).strip()
        except Exception as e:
            llm_error = str(e)
            answer = (extractive_answer or "").strip()
        if not answer:
            answer = UNKNOWN_POLICY_PHRASE

        latency_ms = int((time.time() - t0) * 1000)
        fallback = answer == UNKNOWN_POLICY_PHRASE
        pipeline = "dense_rag"
        if llm_error and answer != UNKNOWN_POLICY_PHRASE:
            pipeline = "dense_rag_llm_error_extractive_fallback"
        elif llm_error:
            pipeline = "dense_rag_llm_error"
        return {
            "answer": answer,
            "fallback_used": fallback,
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
                for r in selected
            ],
            "citations": [_citation_entry(c) for c in context_chunks],
            "latency_ms": latency_ms,
            "top_k": top_k,
            "threshold": threshold,
            "retrieval_diagnostics": {
                "retrieved_candidates": len(retrieved),
                "above_threshold": len(strong),
                "selected_for_context": len(selected),
                **({"llm_error": llm_error} if llm_error else {}),
            },
            "sentence_evidence": [],
            "ab_variant": "control",
            **_pipeline_fields(rag_retrieval_ran=True, pipeline=pipeline),
        }
    except Exception as e:
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "answer": UNKNOWN_POLICY_PHRASE,
            "fallback_used": True,
            "retrieved": [],
            "citations": [],
            "latency_ms": latency_ms,
            "top_k": top_k,
            "threshold": threshold,
            "retrieval_diagnostics": {
                "error": str(e),
            },
            "sentence_evidence": [],
            "ab_variant": "control",
            **_pipeline_fields(rag_retrieval_ran=True, pipeline="dense_rag_exception"),
        }
