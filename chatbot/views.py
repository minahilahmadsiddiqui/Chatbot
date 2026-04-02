import hashlib
import json
import time
from typing import Any, Dict, List

from django.conf import settings
from django.http import StreamingHttpResponse
from django.db.models import Sum
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from django.http import StreamingHttpResponse


from chatbot.models import ChatMessage, Document
from chatbot.services.embeddings_service import get_embeddings
from chatbot.services.qdrant_service import QdrantService
from chatbot.services.rag_service import FALLBACK_PHRASE, run_rag_query
from chatbot.services.text_splitter import normalize_text, split_text_into_token_chunks



def extract_text(file):
    if not file:
        return None

    filename = getattr(file, "name", "")
    if not filename.lower().endswith(".txt"):
        return None

    raw = file.read()
    try:
        return raw.decode("utf-8")
    except Exception:
        return None

@csrf_exempt
@api_view(["POST"])
def upload_document(request):
    """
    Backwards-compatible ingestion endpoint.

    Accepts either:
      - JSON: {"text": "...", "name": "...optional"}
      - multipart: file upload under key "file" (must be .txt)

    No file storage is used; only text is embedded into Qdrant.
    """
    raw_text = request.data.get("text")
    file_obj = request.FILES.get("file")

    if raw_text and file_obj:
        return Response({"error": "Provide either 'text' or 'file', not both."}, status=status.HTTP_400_BAD_REQUEST)

    content: str
    doc_name: str
    if file_obj:
        content = extract_text(file_obj) or ""
        if not content:
            return Response({"error": "Only .txt uploads are supported."}, status=status.HTTP_400_BAD_REQUEST)
        doc_name = request.data.get("name") or getattr(file_obj, "name", "uploaded_text")
    else:
        content = str(raw_text or "")
        if not content.strip():
            return Response({"error": "No text provided."}, status=status.HTTP_400_BAD_REQUEST)
        doc_name = request.data.get("name") or "raw_text"

    normalized = normalize_text(content)
    if not normalized:
        return Response({"error": "Text is empty after normalization."}, status=status.HTTP_400_BAD_REQUEST)

    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    existing = Document.objects.filter(content_hash=content_hash).first()
    if existing:
        return Response(
            {
                "message": "Document already exists",
                "document_id": existing.id,
                "chunk_count": existing.chunk_count,
                "token_count": existing.token_count,
                "embedding_count": existing.embedding_count,
                "status": existing.status,
            },
            status=status.HTTP_200_OK,
        )

    chunk_size_tokens = int(getattr(settings, "RAG_INGEST_CHUNK_SIZE_TOKENS", 500))
    overlap_tokens = int(getattr(settings, "RAG_INGEST_CHUNK_OVERLAP_TOKENS", 75))
    embedding_batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 64))

    chunks, token_counts = split_text_into_token_chunks(
        normalized, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
    )
    if not chunks:
        return Response({"error": "Unable to chunk provided text."}, status=status.HTTP_400_BAD_REQUEST)

    chunk_count = len(chunks)
    token_count = sum(token_counts)
    embedding_count = chunk_count

    doc = Document.objects.create(
        name=doc_name,
        content_hash=content_hash,
        content_length=len(normalized),
        chunk_count=chunk_count,
        token_count=token_count,
        embedding_count=embedding_count,
        status="processing",
    )

    try:
        embeddings = get_embeddings(chunks, batch_size=embedding_batch_size)
        if len(embeddings) != chunk_count:
            raise RuntimeError("Embeddings returned unexpected count.")

        points: List[Dict[str, Any]] = []
        for i, (chunk_text, embedding, tok_count) in enumerate(zip(chunks, embeddings, token_counts)):
            point_id = doc.id * 1_000_000 + i
            points.append(
                {
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "doc_id": doc.id,
                        "chunk_index": i,
                        "chunk_id": f"{doc.id}_{i}",
                        "text": chunk_text,
                        "token_count": tok_count,
                    },
                }
            )

        qdrant = QdrantService()
        qdrant.add_embeddings(points)

        doc.status = "ready"
        doc.error_message = ""
        doc.save(update_fields=["status", "error_message"])
    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
        doc.save(update_fields=["status", "error_message"])
        raise

    return Response(
        {
            "message": "Ingested successfully",
            "document_id": doc.id,
            "chunk_count": doc.chunk_count,
            "token_count": doc.token_count,
            "embedding_count": doc.embedding_count,
            "status": doc.status,
        },
        status=status.HTTP_201_CREATED,
    )

@csrf_exempt
@api_view(['DELETE'])
def delete_document(request, doc_id):
    doc = Document.objects.filter(id=doc_id).first()

    if not doc:
        return Response({"error": "Document not found"}, status=404)

    qdrant = QdrantService()
    qdrant.delete_by_doc_id(doc_id)

    doc.delete()

    return Response({
        "message": "Deleted successfully"
    })

@csrf_exempt
@api_view(['GET'])
def get_stats(request):
    total_documents = Document.objects.count()
    total_chunks = Document.objects.aggregate(Sum('chunk_count')).get('chunk_count__sum') or 0
    total_vector_embeddings = Document.objects.aggregate(Sum('embedding_count')).get('embedding_count__sum') or 0
    total_tokens = Document.objects.aggregate(Sum('token_count')).get('token_count__sum') or 0

    return Response({
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        "total_vector_embeddings": total_vector_embeddings,
        "total_tokens": total_tokens,
    })

@csrf_exempt
@api_view(['GET'])
def get_all_documents(request):
    documents = Document.objects.all().order_by('-created_at')

    data = []
    for doc in documents:
        data.append({
            "id": doc.id,
            "name": doc.name,
            "chunk_count": doc.chunk_count,
            "embedding_count": doc.embedding_count,
            "token_count": doc.token_count,
            "status": doc.status,
            "created_at": doc.created_at
        })

    return Response(data)

@csrf_exempt
@api_view(["POST"])
def chat_query(request):
    """
    POST /api/chat/query
    Body: {"query": "...", "top_k": 5, "threshold": 0.3, "session_id": "optional"}
    """
    query = request.data.get("query") or ""
    top_k = request.data.get("top_k", getattr(settings, "RAG_TOP_K", 5))
    threshold = request.data.get("threshold", getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.3))
    session_id = request.data.get("session_id") or "default"

    try:
        top_k_int = int(top_k)
        threshold_f = float(threshold)
    except Exception:
        return Response({"error": "Invalid 'top_k' or 'threshold' types."}, status=status.HTTP_400_BAD_REQUEST)

    t0 = time.time()
    result = run_rag_query(query=query, top_k=top_k_int, threshold=threshold_f)
    latency_ms = int((time.time() - t0) * 1000)

    retrieved = result.get("retrieved") or []
    citations = result.get("citations") or []
    retrieved_chunk_ids: List[str] = []
    for c in citations:
        chunk_id = c.get("chunk_id")
        chunk_index = c.get("chunk_index")
        if chunk_id is not None:
            retrieved_chunk_ids.append(str(chunk_id))
        elif chunk_index is not None:
            retrieved_chunk_ids.append(str(chunk_index))

    ChatMessage.objects.create(
        session_id=str(session_id),
        query=str(query),
        retrieved_chunk_ids=retrieved_chunk_ids,
        model_used=str(getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")),
        response_text=result.get("answer") or FALLBACK_PHRASE,
        latency_ms=latency_ms,
        fallback_used=bool(result.get("fallback_used")),
    )

    return Response(
        {
            "answer": result.get("answer") or FALLBACK_PHRASE,
            "fallback_used": bool(result.get("fallback_used")),
            "retrieved": retrieved,
            "citations": citations,
            "latency_ms": latency_ms,
            "top_k": result.get("top_k"),
            "threshold": result.get("threshold"),
        },
        status=status.HTTP_200_OK,
    )

@csrf_exempt
@api_view(["POST"])
def chat_query_stream(request):
    """
    Server-Sent Events (SSE) variant of chat/query.
    Emits: start -> answer -> done (+ citations payload).

    Body: {"query": "...", "top_k": 5, "threshold": 0.3, "session_id": "optional"}
    """
    query = request.data.get("query") or ""
    top_k = request.data.get("top_k", getattr(settings, "RAG_TOP_K", 5))
    threshold = request.data.get("threshold", getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.3))
    session_id = request.data.get("session_id") or "default"

    try:
        top_k_int = int(top_k)
        threshold_f = float(threshold)
    except Exception:
        return Response({"error": "Invalid 'top_k' or 'threshold' types."}, status=status.HTTP_400_BAD_REQUEST)

    result = run_rag_query(query=query, top_k=top_k_int, threshold=threshold_f)
    citations = result.get("citations") or []
    top_k_used = result.get("top_k")
    threshold_used = result.get("threshold")

    # Persist audit row (kept non-blocking for the client; still synchronous server-side).
    retrieved_chunk_ids: List[str] = []
    for c in citations:
        chunk_id = c.get("chunk_id")
        chunk_index = c.get("chunk_index")
        if chunk_id is not None:
            retrieved_chunk_ids.append(str(chunk_id))
        elif chunk_index is not None:
            retrieved_chunk_ids.append(str(chunk_index))
    ChatMessage.objects.create(
        session_id=str(session_id),
        query=str(query),
        retrieved_chunk_ids=retrieved_chunk_ids,
        model_used=str(getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")),
        response_text=result.get("answer") or FALLBACK_PHRASE,
        latency_ms=int(result.get("latency_ms") or 0),
        fallback_used=bool(result.get("fallback_used")),
    )

    answer = result.get("answer") or FALLBACK_PHRASE

    def event_stream():
        # Minimal SSE envelope; frontend can render as it arrives.
        yield "event: start\ndata: {}\n\n"
        yield f"event: answer\ndata: {answer}\n\n"
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
        yield f"event: done\ndata: {json.dumps({'top_k': top_k_used, 'threshold': threshold_used})}\n\n"

   # return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
   
    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )

    # Basic SSE-friendly headers; avoid hop-by-hop headers like 'Connection'
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"   # helps if you have nginx/reverse proxy

    return response