import time
from typing import List

from django.conf import settings
from openai import OpenAI


def get_embeddings(texts: List[str], *, batch_size: int = 64) -> List[List[float]]:
    """
    Compute embeddings with batching to avoid provider request size issues.
    """
    api_key = getattr(settings, "OPENROUTER_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    embedding_model = getattr(settings, "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    extra_headers = {
        "HTTP-Referer": str(getattr(settings, "OPENROUTER_REFERER", "http://localhost:8000")),
        "X-Title": str(getattr(settings, "OPENROUTER_TITLE", "chatbot-backend")),
    }
    client = OpenAI(api_key=api_key, base_url=base_url)

    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=embedding_model,
                    input=batch,
                    extra_headers=extra_headers,
                )
                embeddings.extend([item.embedding for item in response.data])
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                time.sleep(2**attempt)
        if last_exc is not None:
            raise last_exc
    return embeddings