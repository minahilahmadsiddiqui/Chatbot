from __future__ import annotations

from typing import List, Dict, Any

from django.conf import settings
from openai import OpenAI


def _openrouter_client() -> OpenAI:
    api_key = getattr(settings, "OPENROUTER_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def generate_answer(
    *,
    question: str,
    context_chunks: List[Dict[str, Any]],
    fallback_phrase: str = "Contact the HR department",
    model: str = "google/gemini-2.5-flash",
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Calls OpenRouter Chat Completions API.
    Response is forced to be grounded by prompt rules.
    """
    client = _openrouter_client()

    # Keep the prompt tight and explicit; fallback must match exact phrase.
    context_text = "\n\n".join(
        [
            f"[Chunk {i + 1}] {c.get('text', '')}".strip()
            for i, c in enumerate(context_chunks)
            if c.get("text")
        ]
    )

    prompt = (
        "You are a retrieval-augmented enterprise assistant.\n"
        "Answer ONLY using the provided context. If the context does not contain the answer, "
        f"respond with EXACTLY: {fallback_phrase}\n"
        "Do not guess. Do not include any other text.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_text}\n"
    )

    extra_headers = {
        "HTTP-Referer": str(getattr(settings, "OPENROUTER_REFERER", "http://localhost:8000")),
        "X-Title": str(getattr(settings, "OPENROUTER_TITLE", "chatbot-backend")),
    }
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_output_tokens,
        extra_headers=extra_headers,
    )
    content = (response.choices[0].message.content or "").strip()
    return content or fallback_phrase


# Backwards-compatible alias for existing imports.
def generate_gemini_answer(
    *,
    question: str,
    context_chunks: List[Dict[str, Any]],
    fallback_phrase: str = "Contact the HR department",
    model: str = "google/gemini-2.5-flash",
    max_output_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    return generate_answer(
        question=question,
        context_chunks=context_chunks,
        fallback_phrase=fallback_phrase,
        model=model,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

