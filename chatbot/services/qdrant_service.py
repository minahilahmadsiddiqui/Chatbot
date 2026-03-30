
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, VectorParams


class QdrantService:
    """
    Thin wrapper around Qdrant for ingestion (upsert), retrieval (search),
    and deletion (remove all vectors belonging to a document).
    """

    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
        self.collection_name = getattr(settings, "QDRANT_COLLECTION_NAME", "documents")
        self.vector_size = getattr(settings, "QDRANT_VECTOR_SIZE", 1536)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def add_embeddings(self, points) -> None:
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector, *, limit: int = 5):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
        )

    def delete_by_doc_id(self, document_id: int) -> None:
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=document_id))])
        self.client.delete(collection_name=self.collection_name, points_selector={"filter": flt})
    