from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from django.conf import settings

from chatbot.services.embeddings_service import get_embeddings
from chatbot.services.gemini_service import generate_answer
from chatbot.services.qdrant_service import QdrantService
from chatbot.services.text_splitter import count_tokens

FALLBACK_PHRASE = "Contact the HR department"


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
    return q


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

        search_results = qdrant.search(query_embedding, limit=top_k) or []

        # Qdrant score: higher is more similar (depending on distance metric).
        retrieved: List[Dict[str, Any]] = []
        for r in search_results:
            score = r.score if hasattr(r, "score") else r.get("score")
            payload = r.payload if hasattr(r, "payload") else r.get("payload")
            retrieved.append({"score": score, "payload": payload})

        filtered = [r for r in retrieved if (r.get("score") is not None and r["score"] >= threshold)]

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

        answer = generate_answer(
            question=sanitized,
            context_chunks=context_chunks,
            fallback_phrase=FALLBACK_PHRASE,
            model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
            max_output_tokens=getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 512),
            temperature=getattr(settings, "OPENROUTER_TEMPERATURE", 0.2),
        )
    except Exception:
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
    return {
        "answer": answer,
        "fallback_used": answer.strip() == FALLBACK_PHRASE,
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
        results = qdrant.search(state["query_embedding"], limit=top_k) or []
        retrieved: List[Dict[str, Any]] = []
        for r in results:
            score = r.score if hasattr(r, "score") else r.get("score")
            payload = r.payload if hasattr(r, "payload") else r.get("payload")
            retrieved.append({"score": score, "payload": payload})
        return {"retrieved": retrieved}

    def filter_context_node(state: RagState) -> RagState:
        retrieved = state.get("retrieved") or []
        filtered = [r for r in retrieved if (r.get("score") is not None and r["score"] >= threshold)]
        context_chunks = _build_context(filtered, max_context_tokens=max_context_tokens)
        if not context_chunks:
            return {"context_chunks": [], "fallback_used": True}
        return {"context_chunks": context_chunks, "fallback_used": False}

    def answer_node(state: RagState) -> RagState:
        context_chunks = state.get("context_chunks") or []
        question = state["query"]
        answer = generate_answer(
            question=question,
            context_chunks=context_chunks,
            fallback_phrase=FALLBACK_PHRASE,
            model=getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"),
            max_output_tokens=getattr(settings, "OPENROUTER_MAX_OUTPUT_TOKENS", 512),
            temperature=getattr(settings, "OPENROUTER_TEMPERATURE", 0.2),
        )
        return {"answer": answer, "fallback_used": answer.strip() == FALLBACK_PHRASE}

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
        final_state: RagState = graph.invoke(initial)  # type: ignore
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
        citations.append(
            {
                "doc_id": c.get("doc_id"),
                "chunk_index": c.get("chunk_index"),
                "chunk_id": c.get("chunk_id"),
                "text_preview": _text_preview(c.get("text") or ""),
                "token_count": c.get("token_count"),
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

