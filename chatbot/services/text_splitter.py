from __future__ import annotations

from typing import List, Tuple

import tiktoken
_ENCODING = None



def normalize_text(text: str) -> str:
    # Normalize unicode whitespace and line breaks.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse consecutive whitespace.
    text = " ".join(text.split())
    return text.strip()


def approx_tokenize(text: str) -> List[str]:
    # Exact-ish tokenization to match embedding provider behavior as closely as possible.
    enc = _encoding()
    return [str(t) for t in enc.encode(text)]


def _encoding():
    """
    Cache and return an encoding for the embedding model.
    Falls back gracefully if model-specific encoding can't be resolved.
    """
    global _ENCODING
    if _ENCODING is not None:
        return _ENCODING

    try:
        _ENCODING = tiktoken.encoding_for_model("text-embedding-3-small")
    except Exception:
        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def count_tokens(text: str) -> int:
    enc = _encoding()
    return len(enc.encode(text))


def split_text_into_token_chunks(
    text: str,
    *,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 75,
) -> Tuple[List[str], List[int]]:
    """
    Returns:
      chunks: list of chunk texts
      token_counts: token count per chunk (aligned with chunks list)
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be < chunk_size_tokens")

    enc = _encoding()
    token_ids = enc.encode(text)
    if not token_ids:
        return [], []

    chunks: List[str] = []
    token_counts: List[int] = []

    start = 0
    while start < len(token_ids):
        end = min(start + chunk_size_tokens, len(token_ids))
        chunk_token_ids = token_ids[start:end]
        chunks.append(enc.decode(chunk_token_ids))
        token_counts.append(len(chunk_token_ids))

        if end == len(token_ids):
            break
        start = end - overlap_tokens

    return chunks, token_counts

