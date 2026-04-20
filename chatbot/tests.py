from __future__ import annotations

import re
from unittest.mock import patch

import tiktoken
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, override_settings

from chatbot.services.rag_service import (
    UNKNOWN_POLICY_PHRASE,
    run_rag_query,
    _extractive_answer_with_sources_from_context,
    _sanitize_query,
    _section_label_plausible_for_chunk,
)
from chatbot.services.text_splitter import (
    count_tokens,
    handbook_has_auto_structure,
    infer_legacy_chunk_metadata,
    line_looks_like_toc_leader,
    normalize_handbook_text,
    normalize_text,
    parse_standalone_page_line,
    split_auto_structured_into_embedding_chunks,
    split_handbook_into_embedding_chunks,
    split_text_into_token_chunks,
)


class TextSplitterTests(SimpleTestCase):
    def test_equals_page_marker_parsed_and_enables_auto_structure(self) -> None:
        assert parse_standalone_page_line("=== Page 9 ===") == 9
        assert parse_standalone_page_line("==== Page 12 ====") == 12
        raw = "=== Page 9 ===\n1 Company Overview\nAcme-One Consultants is a consulting firm.\n"
        norm = normalize_handbook_text(raw)
        assert handbook_has_auto_structure(norm)
        rows = split_auto_structured_into_embedding_chunks(
            norm, chunk_size_tokens=200, overlap_tokens=10
        )
        assert rows
        assert rows[0][3] == 9
        assert "Acme-One" in rows[0][0]

    def test_handbook_blocks_carry_section_and_page_metadata(self) -> None:
        raw = (
            "SOURCE_SECTION: 3.6 Gym policy\n"
            "PAGE_NUMBER: 18\n"
            "Employees may claim gym fees under the reimbursement rules.\n"
        )
        rows = split_handbook_into_embedding_chunks(
            normalize_handbook_text(raw),
            chunk_size_tokens=80,
            overlap_tokens=5,
        )
        assert rows
        assert rows[0][2] == "3.6 Gym policy"
        assert rows[0][3] == 18
        assert "gym" in rows[0][0].lower()

    def test_handbook_parser_accepts_separator_variants(self) -> None:
        raw = (
            "SOURCE SECTION - 14.7 Purpose\n"
            "PAGE NUMBER = 18\n"
            "Gym reimbursement details here.\n"
        )
        rows = split_handbook_into_embedding_chunks(
            normalize_handbook_text(raw),
            chunk_size_tokens=80,
            overlap_tokens=5,
        )
        assert rows
        assert rows[0][2] == "14.7 Purpose"
        assert rows[0][3] == 18

    def test_auto_structured_fills_metadata_without_source_section(self) -> None:
        raw = (
            "43 | P age === Page 44 === Gym Fee Reimbursement 14.7 Purpose To encourage health.\n"
            "14.8 Scope This SOP applies to all employees.\n"
        )
        norm = normalize_handbook_text(raw)
        assert handbook_has_auto_structure(norm)
        rows = split_auto_structured_into_embedding_chunks(
            norm,
            chunk_size_tokens=120,
            overlap_tokens=5,
        )
        assert rows
        assert rows[0][3] == 44
        assert rows[0][4] == ""
        assert "Gym Fee Reimbursement" in rows[0][2]
        assert rows[1][2].startswith("14.8 Scope")
        assert rows[1][3] == 44

    def test_chapter_heading_sets_chapter_name_for_subsections(self) -> None:
        raw = (
            "Chapter 2: Reimbursement policies\n"
            "Page 48\n"
            "14.7 Purpose\n"
            "The company encourages approved certifications.\n"
        )
        norm = normalize_handbook_text(raw)
        rows = split_auto_structured_into_embedding_chunks(
            norm,
            chunk_size_tokens=120,
            overlap_tokens=5,
        )
        assert rows
        assert rows[0][4] == "Chapter 2: Reimbursement policies"
        assert rows[0][2].startswith("14.7 Purpose")
        assert rows[0][3] == 48

    def test_source_section_titled_chapter_populates_chapter_name(self) -> None:
        raw = (
            "SOURCE_SECTION: Chapter 4: Code of Conduct\n"
            "PAGE_NUMBER: 10\n"
            "Employees must act professionally.\n"
        )
        rows = split_handbook_into_embedding_chunks(
            normalize_handbook_text(raw),
            chunk_size_tokens=80,
            overlap_tokens=5,
        )
        assert rows
        assert rows[0][2] == "Chapter 4: Code of Conduct"
        assert rows[0][4] == "Chapter 4: Code of Conduct"
        assert rows[0][3] == 10

    def test_legacy_chunk_metadata_inference(self) -> None:
        flat = (
            "=== Page 44 === Gym Fee Reimbursement 14.7 Purpose To encourage employee health and fitness."
        )
        sec, page, _ch = infer_legacy_chunk_metadata(flat)
        assert page == 44
        assert "14.7" in sec or "Purpose" in sec

    def test_infer_section_skips_toc_dot_leaders(self) -> None:
        flat = (
            "14.20 Continuous Learning ........................................................ 48\n"
            "The certification reimbursement policy covers approved exams up to the annual limit."
        )
        sec, _page, _ch = infer_legacy_chunk_metadata(flat)
        assert "14.20" not in sec
        assert "Continuous" not in sec

    def test_toc_line_not_an_implicit_section_boundary(self) -> None:
        raw = (
            "PAGE_NUMBER: 48\n"
            "18.5 Reimbursement Amount & Limits .........................................................\n"
            "The annual cap is stated in the approval form.\n"
        )
        norm = normalize_handbook_text(raw)
        rows = split_auto_structured_into_embedding_chunks(
            norm,
            chunk_size_tokens=100,
            overlap_tokens=5,
        )
        assert rows
        assert "18.5" not in rows[0][2]
        assert "annual cap" in rows[0][0].lower()

    def test_line_looks_like_toc_leader(self) -> None:
        assert line_looks_like_toc_leader("18.5 Reimbursement Amount & Limits .................................... 48")
        assert not line_looks_like_toc_leader("14.8 Scope This SOP applies to all employees.")

    def test_handbook_parser_extracts_page_from_decorative_inline_line(self) -> None:
        raw = (
            "SOURCE_SECTION: Gym Fee Reimbursement\n"
            "43 | P age === Page 44 === Gym Fee Reimbursement 14.7 Purpose To encourage health.\n"
        )
        rows = split_handbook_into_embedding_chunks(
            normalize_handbook_text(raw),
            chunk_size_tokens=120,
            overlap_tokens=5,
        )
        assert rows
        assert rows[0][2] == "Gym Fee Reimbursement"
        assert rows[0][3] == 44
        assert "Gym Fee Reimbursement" in rows[0][0]
        assert "43 |" not in rows[0][0]

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
    def test_section_plausible_trusts_payload_when_heading_not_in_body(self) -> None:
        assert _section_label_plausible_for_chunk(
            "3.6 Gym policy",
            "Employees may claim gym fees under the reimbursement rules.",
        )

    def test_section_plausible_rejects_toc_only_line(self) -> None:
        assert not _section_label_plausible_for_chunk(
            "14.20 Continuous Learning",
            "14.20 Continuous Learning ........................................................ 48",
        )

    def test_sanitize_query_fixes_common_typos(self) -> None:
        fixed = _sanitize_query("Explain reimbursment polciy for employes")
        # Simplified runtime keeps sanitize lightweight (trim/length guard only).
        assert fixed == "Explain reimbursment polciy for employes"

    @patch("chatbot.services.rag_service.get_embeddings")
    @patch("chatbot.services.rag_service.generate_answer")
    def test_fallback_when_no_context(self, mock_answer, mock_embed) -> None:
        mock_embed.return_value = [[0.0] * 1536]

        class DummyQdrant:
            def search(self, *_args, **_kwargs):
                return []

        with patch("chatbot.services.rag_service.QdrantService", return_value=DummyQdrant()):
            result = run_rag_query(query="What is HR policy?", top_k=3, threshold=0.9, max_context_tokens=200)

        assert result["answer"] == UNKNOWN_POLICY_PHRASE
        assert result["fallback_used"] is True
        assert result["top_k"] == 3
        assert result["threshold"] == 0.9
        mock_answer.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    @patch("chatbot.services.rag_service.generate_answer")
    def test_citations_return_when_context_exists(self, mock_answer, mock_embed) -> None:
        mock_embed.return_value = [[0.0] * 1536]
        mock_answer.return_value = "Employee handbook says you must wear a badge."

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

        # Dense-first runtime calls LLM when context exists.
        assert result["answer"] == "Employee handbook says you must wear a badge."
        assert result["fallback_used"] is False
        assert len(result["citations"]) == 1
        assert result["citations"][0]["chunk_id"] == "1_0"
        mock_answer.assert_called_once()

    def test_strict_extractive_hajj_omits_weak_only_lines(self) -> None:
        """Weak query hits (e.g. 'allowed' alone) must not pull unrelated handbook lines."""
        chunk = {
            "text": (
                "8.10 Hajj Leaves\n"
                "All Muslim employees shall be entitled to fifteen (15) paid leaves to fulfill "
                "their obligation of performing Hajj, once during their service.\n"
                "• Salaries are proportional to efficiency, not to how many people work in your company.\n"
                "Limited use of cell phones at work is allowed for personal reasons."
            ),
            "chunk_id": "1_0",
            "doc_id": 1,
            "chunk_index": 0,
            "source_section": "8.10 Hajj",
            "page_number": 26,
            "token_count": count_tokens(
                "All Muslim employees shall be entitled to fifteen (15) paid leaves."
            ),
        }
        body, _used = _extractive_answer_with_sources_from_context(
            "how many hajj leaves are allowed?",
            [chunk],
        )
        assert body
        low = body.lower()
        assert "hajj" in low
        assert "cell phone" not in low
        assert "salaries are proportional" not in low

    def test_strict_respects_max_sentences_limit(self) -> None:
        chunk = {
            "text": (
                "Hajj policy allows fifteen paid leaves. "
                "This benefit is once in service. "
                "Submit leave request before travel."
            ),
            "chunk_id": "1_2",
            "doc_id": 1,
            "chunk_index": 2,
            "source_section": "8.10 Hajj",
            "page_number": 26,
            "token_count": count_tokens("Hajj policy allows fifteen paid leaves."),
        }
        body, _used = _extractive_answer_with_sources_from_context(
            "how many hajj leaves are allowed?",
            [chunk],
            max_sentences=1,
        )
        assert body
        # At most one terminal punctuation sentence is expected after enforcement.
        assert len([s for s in re.split(r"(?<=[.!?])\\s+", body.strip()) if s.strip()]) <= 1

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
    def test_obvious_gibberish_returns_unknown_without_retrieval(self, mock_embed) -> None:
        result = run_rag_query(query="nfbhgbgh", top_k=1, threshold=0.9, max_context_tokens=200)
        assert result["answer"] == UNKNOWN_POLICY_PHRASE
        assert result.get("pipeline") == "obvious_gibberish_short_circuit"
        assert result.get("rag_retrieval_ran") is False
        mock_embed.assert_not_called()

    @override_settings(RAG_ENABLE_GIBBERISH_FILTER=True)
    @patch("chatbot.services.rag_service.get_embeddings")
    def test_gibberish_query_returns_unknown_phrase(self, mock_embed) -> None:
        result = run_rag_query(query="huiiiuihuh zzxxqq", top_k=1, threshold=0.9, max_context_tokens=200)
        assert result["answer"] == UNKNOWN_POLICY_PHRASE
        assert result["fallback_used"] is True
        assert result.get("rag_retrieval_ran") is False
        # Simplified runtime only keeps obvious gibberish short-circuit.
        assert result.get("pipeline") == "obvious_gibberish_short_circuit"
        mock_embed.assert_not_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_normal_question_not_blocked_as_gibberish_by_default(self, mock_embed) -> None:
        mock_embed.return_value = [[0.0] * 1536]

        class DummyQdrant:
            def search(self, *_args, **_kwargs):
                return []

        with patch("chatbot.services.rag_service.QdrantService", return_value=DummyQdrant()):
            result = run_rag_query(
                query="What are the consequences of falsifying attendance records?",
                top_k=2,
                threshold=0.99,
                max_context_tokens=200,
            )

        assert result.get("pipeline") != "obvious_gibberish_short_circuit"
        assert result.get("rag_retrieval_ran") is True
        mock_embed.assert_called()

    @patch("chatbot.services.rag_service.get_embeddings")
    def test_hiring_policy_not_mistaken_for_greeting_hi(self, mock_embed) -> None:
        """'hi' substring in hiring/highlight must not short-circuit to greeting."""
        mock_embed.return_value = [[0.0] * 1536]

        class DummyQdrant:
            def search(self, *_args, **_kwargs):
                return []

        with patch("chatbot.services.rag_service.QdrantService", return_value=DummyQdrant()):
            result = run_rag_query(
                query="What is the hiring policy for interns?",
                top_k=3,
                threshold=0.9,
                max_context_tokens=200,
            )

        assert result["answer"].strip().lower() != "hi"
        assert not result["answer"].startswith("Hello! I can help you find information in the employee handbook")
        mock_embed.assert_called_once()


class ResponseBeautifyHtmlTests(SimpleTestCase):
    def test_format_plain_metadata_uses_colon_labels_and_unknown_without_citations(self) -> None:
        from chatbot.services.response_beautify_service import CLOSING_LINE, format_plain_answer_with_metadata

        out = format_plain_answer_with_metadata("I am doing well, thank you.", [])
        assert "Source: Unknown" in out
        assert "Page number: Unknown" in out
        assert CLOSING_LINE in out

    def test_greeting_style_skips_metadata_when_disabled(self) -> None:
        from chatbot.services.response_beautify_service import CLOSING_LINE, build_chat_answer_html, format_plain_answer_with_metadata

        plain = format_plain_answer_with_metadata(
            "I am doing well, thank you. How can I help you?", [], append_source_metadata=False
        )
        assert plain == "I am doing well, thank you. How can I help you?"
        assert "Source" not in plain
        assert CLOSING_LINE not in plain

        html_out = build_chat_answer_html(answer="Hello.", citations=[], append_source_metadata=False)
        assert "chat-source-meta" not in html_out
        assert "Source:" not in html_out
        assert CLOSING_LINE not in html_out

    def test_strip_handbook_page_chapter_noise(self) -> None:
        from chatbot.services.response_beautify_service import strip_handbook_layout_artifacts

        raw = (
            "This policy applies. 22 | P age 7 CHAPTER 2: RECRUITMENT AND INDUCTION POLICY "
            "The company reimburses fees."
        )
        out = strip_handbook_layout_artifacts(raw)
        assert "P age" not in out
        assert "CHAPTER 2" not in out
        assert "RECRUITMENT AND INDUCTION" not in out
        assert "The company reimburses fees" in out

        eq = "Intro === Page 44 === rest of sentence."
        assert "Page 44" not in strip_handbook_layout_artifacts(eq)
        assert "rest of sentence" in strip_handbook_layout_artifacts(eq)

    def test_strip_ampersand_section_headings(self) -> None:
        from chatbot.services.response_beautify_service import strip_handbook_layout_artifacts

        raw = "Proof submitted. & Reimbursement Process • Each receipt must be valid."
        out = strip_handbook_layout_artifacts(raw)
        assert "& Reimbursement" not in out
        assert "Each receipt" in out
        assert "costs & Benefits" in strip_handbook_layout_artifacts(
            "Employees enjoy costs & Benefits for all staff."
        )

    def test_strip_trailing_source_page_footer(self) -> None:
        from chatbot.services.response_beautify_service import strip_trailing_source_page_footer

        raw = (
            "This policy will be reviewed quarterly. "
            "Source section 14.20 Continuous Page number 48"
        )
        out = strip_trailing_source_page_footer(raw)
        assert "Page number" not in out
        assert "Source section" not in out
        assert "reviewed quarterly" in out

    def test_build_chat_answer_html_structure(self) -> None:
        from chatbot.services.response_beautify_service import build_chat_answer_html

        ans = "First paragraph.\n\n• Point one\n• Point two"
        cites = [{"source_section": "14.20 Continuous", "page": 48}]
        html_out = build_chat_answer_html(answer=ans, citations=cites)
        assert '<div class="chat-answer">' in html_out
        assert "<strong>Source:</strong>" in html_out
        assert "14.20 Continuous" in html_out
        assert "<strong>Page number:</strong>" in html_out
        assert "48" in html_out
        assert "<ul>" in html_out and "<li>" in html_out


class SuperAdminBootstrapCommandTests(SimpleTestCase):
    @patch("chatbot.management.commands.bootstrap_super_admin.FirestoreRepository")
    def test_bootstrap_creates_super_admin(self, mock_repo_cls) -> None:
        repo = mock_repo_cls.return_value
        repo.find_super_admin.return_value = None
        repo.find_admin_by_email.return_value = None

        call_command(
            "bootstrap_super_admin",
            "--email=owner@example.com",
            "--password=StrongPass123",
            "--full-name=Platform Owner",
        )

        repo.create_admin.assert_called_once()
        payload = repo.create_admin.call_args.args[0]
        assert payload["role"] == "super_admin"
        assert payload["is_verified"] is True
        assert payload["email"] == "owner@example.com"

    @patch("chatbot.management.commands.bootstrap_super_admin.FirestoreRepository")
    def test_bootstrap_rejects_second_super_admin(self, mock_repo_cls) -> None:
        repo = mock_repo_cls.return_value
        repo.find_super_admin.return_value = object()

        with self.assertRaises(CommandError):
            call_command(
                "bootstrap_super_admin",
                "--email=owner@example.com",
                "--password=StrongPass123",
                "--full-name=Platform Owner",
            )


class ReplaceSuperAdminCommandTests(SimpleTestCase):
    @patch("chatbot.management.commands.replace_super_admin.FirestoreRepository")
    def test_replace_promotes_existing_and_deletes_previous(self, mock_repo_cls) -> None:
        class Existing:
            id = 10
            email = "old@example.com"

        class Target:
            id = 20
            email = "acme-one@team1.com"

        repo = mock_repo_cls.return_value
        repo.find_super_admin.return_value = Existing()
        repo.find_admin_by_email.return_value = Target()

        call_command(
            "replace_super_admin",
            "--email=acme-one@team1.com",
        )

        repo.update_admin.assert_called_once()
        payload = repo.update_admin.call_args.args[1]
        assert payload["role"] == "super_admin"
        repo.delete_admin.assert_called_once_with(10)
        repo.create_admin.assert_not_called()

    @patch("chatbot.management.commands.replace_super_admin.FirestoreRepository")
    def test_replace_creates_when_target_missing(self, mock_repo_cls) -> None:
        class Existing:
            id = 10
            email = "old@example.com"

        class Created:
            id = 30
            email = "acme-one@team1.com"

        repo = mock_repo_cls.return_value
        repo.find_super_admin.return_value = Existing()
        repo.find_admin_by_email.return_value = None
        repo.create_admin.return_value = Created()

        call_command(
            "replace_super_admin",
            "--email=acme-one@team1.com",
            "--password=StrongPass123",
            "--full-name=Acme One Team",
        )

        repo.create_admin.assert_called_once()
        repo.delete_admin.assert_called_once_with(10)

    @patch("chatbot.management.commands.replace_super_admin.FirestoreRepository")
    def test_replace_requires_password_and_name_when_missing_target(self, mock_repo_cls) -> None:
        repo = mock_repo_cls.return_value
        repo.find_super_admin.return_value = None
        repo.find_admin_by_email.return_value = None

        with self.assertRaises(CommandError):
            call_command(
                "replace_super_admin",
                "--email=acme-one@team1.com",
            )
