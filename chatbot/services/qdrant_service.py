
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
    FilterSelector,
    PayloadSchemaType,
)


class QdrantService:
    """
    Thin wrapper around Qdrant for ingestion (upsert), retrieval (search),
    and deletion (remove all vectors belonging to a document).
    """

    def __init__(self) -> None:
        q_timeout = float(getattr(settings, "QDRANT_TIMEOUT_SEC", 30.0))
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=q_timeout,
        )
        self.collection_name = getattr(settings, "QDRANT_COLLECTION_NAME", "documents")
        self.vector_size = getattr(settings, "QDRANT_VECTOR_SIZE", 1536)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
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
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, *, limit: int = 5):
        # Support both old and new qdrant-client APIs.
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
            )

        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        points = getattr(result, "points", None)
        return points if points is not None else result

    def scan_payload_points(self, *, limit: int = 500):
        """
        Returns raw points with payloads by scrolling the collection.
        Used as a lexical fallback when semantic retrieval misses exact wording.
        """
        if not hasattr(self.client, "scroll"):
            return []

        result = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=limit,
        )
        if isinstance(result, tuple):
            # qdrant-client commonly returns (points, next_page_offset)
            return result[0] or []

        points = getattr(result, "points", None)
        return points or []

    def delete_by_doc_id(self, document_id: int) -> None:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=document_id))])
        # Newer qdrant-client expects a typed selector, not a raw dict.
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=flt),
        )
    