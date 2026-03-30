from __future__ import annotations

from unittest.mock import patch

import tiktoken
from django.test import SimpleTestCase

from chatbot.services.rag_service import FALLBACK_PHRASE, run_rag_query
from chatbot.services.text_splitter import count_tokens, normalize_text, split_text_into_token_chunks


class TextSplitterTests(SimpleTestCase):
    def test_normalize_text_collapses_whitespace(self) -> None:
        raw = "  hello \n\n  world   "
        assert normalize_text(raw) == "hello world"

    def test_split_text_into_token_chunks_respects_chunk_size_and_overlap(self) -> None:
        # Use a deterministic tokenization basis.
        enc = tiktoken.encoding_for_model("text-embedding-3-small")

        # Create a text with enough tokens.
        text = ("word " * 60).strip()
        chunk_size = 10
        overlap = 2

        chunks, token_counts = split_text_into_token_chunks(
            text, chunk_size_tokens=chunk_size, overlap_tokens=overlap
        )
        assert chunks
        assert len(chunks) == len(token_counts)
        assert all(tc <= chunk_size for tc in token_counts)

        token_ids = enc.encode(text)
        # Verify overlap tokens between adjacent chunks.
        for i in range(len(chunks) - 1):
            a_ids = enc.encode(chunks[i])
            b_ids = enc.encode(chunks[i + 1])
            # Overlap should exist unless the first chunk ended exactly at the end of the token list.
            if overlap > 0 and len(a_ids) >= overlap and len(b_ids) >= overlap:
                assert a_ids[-overlap:] == b_ids[:overlap]


class RagServiceTests(SimpleTestCase):
    @patch("chatbot.services.rag_service.get_embeddings")
    @patch("chatbot.services.rag_service.generate_answer")
    def test_fallback_when_no_context(self, mock_answer, mock_embed) -> None:
        mock_embed.return_value = [[0.0] * 1536]

        class DummyQdrant:
            def search(self, *_args, **_kwargs):
                return []

        with patch("chatbot.services.rag_service.QdrantService", return_value=DummyQdrant()):
            result = run_rag_query(query="What is HR policy?", top_k=3, threshold=0.9, max_context_tokens=200)

        assert result["answer"] == FALLBACK_PHRASE
        assert result["fallback_used"] is True
        assert result["top_k"] == 3
        assert result["threshold"] == 0.9
        mock_answer.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    @patch("chatbot.services.rag_service.generate_answer")
    def test_citations_return_when_context_exists(self, mock_answer, mock_embed) -> None:
        mock_embed.return_value = [[0.0] * 1536]
        mock_answer.return_value = "Mock answer"

        payload = {
            "doc_id": 1,
            "chunk_index": 0,
            "chunk_id": "1_0",
            "text": "Employee handbook says you must wear a badge.",
            "token_count": count_tokens("Employee handbook says you must wear a badge."),
        }

        class DummyScoredPoint:
            def __init__(self) -> None:
                self.score = 0.95
                self.payload = payload

        class DummyQdrant:
            def search(self, *_args, **_kwargs):
                return [DummyScoredPoint()]

        with patch("chatbot.services.rag_service.QdrantService", return_value=DummyQdrant()):
            result = run_rag_query(query="What do employees wear?", top_k=1, threshold=0.5, max_context_tokens=200)

        assert result["answer"] == "Mock answer"
        assert result["fallback_used"] is False
        assert len(result["citations"]) == 1
        assert result["citations"][0]["chunk_id"] == "1_0"
        mock_answer.assert_called()
