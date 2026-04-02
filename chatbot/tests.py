from __future__ import annotations

from unittest.mock import patch

import tiktoken
from django.test import SimpleTestCase

from chatbot.services.rag_service import FALLBACK_PHRASE, UNKNOWN_POLICY_PHRASE, run_rag_query, _sanitize_query
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
    def test_sanitize_query_fixes_common_typos(self) -> None:
        fixed = _sanitize_query("Explain reimbursment polciy for employes")
        assert "reimbursement" in fixed.lower()
        assert "policy" in fixed.lower()
        assert "employees" in fixed.lower()

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

        # Strict mode should build the answer from verbatim retrieved text.
        assert result["answer"] == (
            "Here is what I found about wear.\n"
            "\n"
            "### Answer\n"
            "\n"
            "Employee handbook says you must wear a badge.\n"
            "\n"
            "### Source Section\n"
            "Unknown\n"
            "\n"
            "### Page Number\n"
            "Unknown\n"
            "\n"
            "If you have any further questions, do let me know."
        )
        assert result["fallback_used"] is False
        assert len(result["citations"]) == 1
        assert result["citations"][0]["chunk_id"] == "1_0"
        mock_answer.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_im_fine_is_handled_as_greeting_status(self, mock_embed) -> None:
        # Greeting/status replies should not trigger RAG retrieval.
        result = run_rag_query(query="I'm fine", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "employee handbook" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_how_re_you_typo_is_handled_as_greeting(self, mock_embed) -> None:
        result = run_rag_query(query="how re you?", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "looks like you meant" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_how_are_you_exact_has_no_typo_hint(self, mock_embed) -> None:
        result = run_rag_query(query="How are you?", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "looks like you meant" not in result["answer"].lower()
        assert "doing well" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_hoe_r_u_typo_is_handled_as_greeting(self, mock_embed) -> None:
        result = run_rag_query(query="hoe r u", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "looks like you meant" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_hui_are_you_typo_is_handled_as_greeting(self, mock_embed) -> None:
        result = run_rag_query(query="hui are you?", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "looks like you meant" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_hui_r_you_typo_is_handled_as_greeting(self, mock_embed) -> None:
        result = run_rag_query(query="hui r you?", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "looks like you meant" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_hoe_are_you_typo_is_handled_as_greeting(self, mock_embed) -> None:
        result = run_rag_query(query="hoe are you?", top_k=1, threshold=0.9, max_context_tokens=200)
        assert "looks like you meant" in result["answer"].lower()
        assert result["fallback_used"] is False
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_gibberish_query_returns_unknown_phrase(self, mock_embed) -> None:
        result = run_rag_query(query="huiiiuihuh are u=you?", top_k=1, threshold=0.9, max_context_tokens=200)
        assert result["answer"] == UNKNOWN_POLICY_PHRASE
        assert result["fallback_used"] is True
        mock_embed.assert_not_called()
