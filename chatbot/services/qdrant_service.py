import time
from typing import Optional

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    FilterSelector,
    PayloadSchemaType,
)


class QdrantService:
    """
    Thin wrapper around Qdrant for ingestion (upsert), retrieval (search),
    and deletion (remove all vectors belonging to a document).
    """

    def __init__(self, *, validate_vector_dimension: bool = True) -> None:
        q_timeout = float(getattr(settings, "QDRANT_TIMEOUT_SEC", 30.0))
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=q_timeout,
        )
        self.collection_name = getattr(settings, "QDRANT_COLLECTION_NAME", "documents")
        self.vector_size = getattr(settings, "QDRANT_VECTOR_SIZE", 4096)
        self._validate_vector_dimension = bool(validate_vector_dimension)
        # Named-vector collections (common when created in Qdrant Cloud UI) require
        # vector={"name": [...]} on upsert and `using=name` on query.
        explicit = getattr(settings, "QDRANT_VECTOR_NAME", None)
        self._vector_using: str | None = (str(explicit).strip() if explicit else "") or None
        self._ensure_collection()
        self._vector_using = self._resolve_vector_using()

    def _named_vector_keys(self, vectors_config) -> list[str]:
        if vectors_config is None:
            return []
        if isinstance(vectors_config, VectorParams):
            return []
        if isinstance(vectors_config, dict):
            return [str(k) for k in vectors_config.keys() if k is not None]
        keys = getattr(vectors_config, "keys", None)
        if callable(keys):
            try:
                return [str(k) for k in list(keys())]
            except Exception:
                return []
        return []

    def _resolve_vector_using(self) -> str | None:
        explicit = getattr(settings, "QDRANT_VECTOR_NAME", None)
        if explicit:
            return str(explicit).strip() or None
        if not self.client.collection_exists(self.collection_name):
            return None
        try:
            info = self.client.get_collection(self.collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None) if params is not None else None
            names = self._named_vector_keys(vectors)
            if len(names) == 1:
                return names[0]
            if len(names) > 1:
                # Prefer a common default name if present.
                for candidate in ("default", "dense", "text", "embedding"):
                    if candidate in names:
                        return candidate
                return names[0]
        except Exception:
            return None
        return None

    def _read_existing_vector_size(self) -> Optional[int]:
        """
        Read the effective vector dimension from an existing collection.
        Supports both unnamed and named-vector configs.
        """
        if not self.client.collection_exists(self.collection_name):
            return None
        try:
            info = self.client.get_collection(self.collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None) if params is not None else None
            if isinstance(vectors, VectorParams):
                size = getattr(vectors, "size", None)
                return int(size) if size is not None else None
            if isinstance(vectors, dict):
                # Prefer configured vector name if provided.
                if self._vector_using and self._vector_using in vectors:
                    vp = vectors[self._vector_using]
                    size = getattr(vp, "size", None)
                    return int(size) if size is not None else None
                # Otherwise use first available vector config.
                if vectors:
                    first = next(iter(vectors.values()))
                    size = getattr(first, "size", None)
                    return int(size) if size is not None else None
            return None
        except Exception:
            return None

    def _wrap_vector(self, embedding: list) -> list | dict:
        if self._vector_using:
            return {self._vector_using: embedding}
        return embedding

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
        else:
            existing_dim = self._read_existing_vector_size()
            if (
                self._validate_vector_dimension
                and existing_dim is not None
                and int(existing_dim) != int(self.vector_size)
            ):
                auto_recreate = bool(
                    getattr(settings, "QDRANT_AUTO_RECREATE_ON_DIMENSION_MISMATCH", False)
                )
                if auto_recreate:
                    self.client.delete_collection(collection_name=self.collection_name)
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                    )
                else:
                    raise RuntimeError(
                        "Qdrant vector dimension mismatch for collection "
                        f"'{self.collection_name}': existing={existing_dim}, configured={self.vector_size}. "
                        "Use a new QDRANT_COLLECTION_NAME, recreate the collection, or set "
                        "QDRANT_AUTO_RECREATE_ON_DIMENSION_MISMATCH=1 (destructive)."
                    )

        # Ensure there is an index on the payload field `doc_id` for efficient filtered deletes.
        # Safe to call if it already exists.
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception:
            # If the index already exists or the client version handles this differently,
            # we ignore errors here to avoid breaking startup.
            pass

    def add_embeddings(self, points) -> None:
        batch: list[PointStruct] = []
        for p in points:
            if isinstance(p, PointStruct):
                batch.append(p)
            else:
                vec = self._wrap_vector(p["vector"])
                batch.append(
                    PointStruct(
                        id=p["id"],
                        vector=vec,
                        payload=p.get("payload") or {},
                    )
                )
        if not batch:
            return

        upsert_batch_size = int(getattr(settings, "QDRANT_UPSERT_BATCH_SIZE", 64))
        upsert_retries = int(getattr(settings, "QDRANT_UPSERT_RETRIES", 3))
        initial_backoff_sec = float(getattr(settings, "QDRANT_UPSERT_RETRY_BACKOFF_SEC", 1.0))

        for i in range(0, len(batch), max(1, upsert_batch_size)):
            chunk = batch[i : i + max(1, upsert_batch_size)]
            attempt = 0
            while True:
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=chunk,
                        wait=True,
                    )
                    break
                except Exception:
                    if attempt >= upsert_retries:
                        raise
                    sleep_for = initial_backoff_sec * (2 ** attempt)
                    time.sleep(min(sleep_for, 8.0))
                    attempt += 1

    def search(self, query_vector, *, limit: int = 5):
        # Prefer query_points (current qdrant-client); pass `using` for named-vector collections.
        query_fn = getattr(self.client, "query_points", None)
        if query_fn is not None:
            result = query_fn(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                using=self._vector_using,
            )
            points = getattr(result, "points", None)
            return points if points is not None else result
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )
        return []

    def scan_payload_points(self, *, limit: int = 256, max_points: Optional[int] = None):
        """
        Scroll payloads (no vectors) for lexical / BM25 scoring over the collection.
        Paginates until `max_points` or the collection is exhausted.
        """
        if not hasattr(self.client, "scroll"):
            return []

        cap = max_points if max_points is not None else limit
        out: list = []
        offset = None
        batch = max(1, min(int(limit), 256))

        while len(out) < cap:
            take = min(batch, cap - len(out))
            try:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    with_payload=True,
                    with_vectors=False,
                    limit=take,
                    offset=offset,
                )
            except Exception:
                break

            if isinstance(result, tuple):
                points, offset = result[0] or [], result[1]
            else:
                points = getattr(result, "points", None) or []
                offset = getattr(result, "next_page_offset", None)

            if not points:
                break
            out.extend(points)
            if offset is None:
                break

        return out

    def delete_by_doc_id(self, document_id: int) -> None:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=document_id))])
        # Newer qdrant-client expects a typed selector, not a raw dict.
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=flt),
        )
    