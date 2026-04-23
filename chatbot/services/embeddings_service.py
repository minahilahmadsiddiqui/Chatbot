import time
from typing import List

from django.conf import settings
from openai import APIStatusError, OpenAI


def _embedding_error_should_retry(exc: Exception, attempt: int, max_attempts: int) -> bool:
    if attempt >= max_attempts - 1:
        return False
    if isinstance(exc, APIStatusError):
        if exc.status_code in (400, 401, 403, 404, 422):
            return False
        return True
    return True


def get_embeddings(
    texts: List[str],
    *,
    batch_size: int = 64,
    openrouter_api_key: str = "",
) -> List[List[float]]:
    """
    Compute embeddings with batching to avoid provider request size issues.
    """
    api_key = str(openrouter_api_key or "").strip()
    if not api_key:
        raise RuntimeError("Missing bot OpenRouter API key")

    base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    embedding_model = getattr(settings, "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    extra_headers = {
        "HTTP-Referer": str(getattr(settings, "OPENROUTER_REFERER", "http://localhost:8000")),
        "X-Title": str(getattr(settings, "OPENROUTER_TITLE", "chatbot-backend")),
    }
    timeout = float(getattr(settings, "OPENROUTER_EMBEDDING_TIMEOUT_SEC", 180))
    max_attempts = max(1, int(getattr(settings, "OPENROUTER_EMBEDDING_MAX_RETRIES", 6)))
    # Our own retry loop; disable SDK default retries to avoid double-backoff.
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)

    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
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
                if not _embedding_error_should_retry(e, attempt, max_attempts):
                    raise
                delay = min(32.0, 2.0 ** (attempt + 1))
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
    return embeddings
