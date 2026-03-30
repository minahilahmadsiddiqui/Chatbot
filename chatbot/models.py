
from django.db import models


class Document(models.Model):
    """
    Tracks an uploaded text source after ingestion into Qdrant.
    """

    name = models.CharField(max_length=255, default="", blank=True)
    # sha256 hash of normalized full text; used for deduplication/idempotency.
    content_hash = models.CharField(max_length=64, unique=True, null=True, blank=True)

    content_length = models.IntegerField(help_text="Total length of normalized text (characters).", null=True, blank=True)
    chunk_count = models.IntegerField(null=True, blank=True)
    token_count = models.IntegerField(help_text="Total number of (approx) tokens embedded across all chunks.", null=True, blank=True)
    embedding_count = models.IntegerField(help_text="Total number of vectors stored in Qdrant.", null=True, blank=True)

    status = models.CharField(max_length=32, default="ready")
    error_message = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.name} ({self.id})"


class ChatMessage(models.Model):
    """
    Minimal chat auditing for production troubleshooting.
    """

    session_id = models.CharField(max_length=64, db_index=True)
    query = models.TextField()

    # chunk identifiers returned by retrieval (e.g. ["12_0","12_1",...])
    retrieved_chunk_ids = models.JSONField(default=list, blank=True)

    model_used = models.CharField(max_length=128)
    response_text = models.TextField()
    latency_ms = models.IntegerField()
    fallback_used = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"ChatMessage(session_id={self.session_id}, id={self.id})"