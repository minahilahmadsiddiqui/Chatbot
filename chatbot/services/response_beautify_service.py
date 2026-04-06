"""
Format LLM chat answers for display: markdown headings for sections, clean paragraphs, lists.

Keeps wording; only adds structure. Must match gemini_service.UNKNOWN_POLICY_PHRASE exactly.

Also builds safe HTML for chat UIs: body first, then Source / Page number headings from citations.
"""

from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Optional, Tuple

# Keep in sync with chatbot.services.gemini_service.UNKNOWN_POLICY_PHRASE
UNKNOWN_POLICY_PHRASE = "The document doesn't mention it, contact the HR department."

# Appended after source/page metadata on successful answers (plain + HTML).
CLOSING_LINE = "If you have any further questions, do let me know."

# Trailing junk sometimes appended by the model on one line, e.g.
# "... policy. Source section 14.20 Continuous Page number 48"
_TAIL_SOURCE_PAGE = re.compile(
    r"(?is)\s+Source\s+section\b.+?\bPage\s+number\b\s*(\d{1,5})\s*\Z"
)
# Looser: no word-boundary between section title and "Page" (model typos)
_TAIL_SOURCE_PAGE_LOOSE = re.compile(
    r"(?is)\s+Source\s+section\s+.+\s+Page\s+number\s+\d{1,5}\s*\Z"
)
# Slightly looser: "Page number 48" at very end
_TAIL_PAGE_ONLY = re.compile(r"(?is)\s+Page\s+number\s+(\d{1,5})\s*\Z")
# Model sometimes omits "section": "... text Source 14.20 Continuous Page number 48"
_TAIL_SOURCE_SHORT = re.compile(
    r"(?is)\s+Source\s+(?!section\b).+?\bPage\s+number\b\s*\d{1,5}\s*\Z"
)

# PDF/export running headers pasted into chunk text, e.g. "22 | P age 7 CHAPTER 2: TITLE..."
# ("P age" = broken "Page"). Removed anywhere in the answer, not only line-start.
# Lookahead must require a real word (2+ letters), not a single char — otherwise "… POLICY The"
# wrongly ends the match at " T" and deletes the rest of the sentence.
_HAND_BOOK_LOOKAHEAD = r"(?=\s+(?:[A-Za-z]{2,})\b|\s+•|\Z)"
# Title words must be ASCII A–Z only; do not use (?i) on them or "The company" is eaten as "title".
_HAND_BOOK_PAGE_PIPE = re.compile(
    r"(?i)\d{1,3}\s*\|\s*P\s+age\s*\d{1,3}\s+"
    r"(?:CHAPTER\s+\d{1,2}\s*:\s*)?"
    r"(?-i:[A-Z]{2,}(?:\s+[A-Z]{2,})*)?"
    + _HAND_BOOK_LOOKAHEAD
)
_HAND_BOOK_PAGE_PIPE_NORMAL = re.compile(
    r"(?i)\d{1,3}\s*\|\s*Page\s+\d{1,3}\s+"
    r"(?:CHAPTER\s+\d{1,2}\s*:\s*)?"
    r"(?-i:[A-Z]{2,}(?:\s+[A-Z]{2,})*)?"
    + _HAND_BOOK_LOOKAHEAD
)
_HAND_BOOK_PAGE_EQUALS = re.compile(r"(?i)\s*=+\s*Page\s+\d{1,5}\s*=+\s*")
# Orphan "CHAPTER N: ALL CAPS TITLE" after a sentence boundary (no pipe cluster)
_HAND_BOOK_CHAPTER_INLINE = re.compile(
    r"(?i)(?<=[.!?•\n])\s+CHAPTER\s+\d{1,2}\s*:\s*"
    r"(?-i:[A-Z]{2,}(?:\s+[A-Z]{2,})*)"
    + _HAND_BOOK_LOOKAHEAD
)
# Export/handbook lines like "& Reimbursement Process" (not sentence "A & B").
_AMP_HEADING_LA = (
    r"(?=\s+(?:[A-Za-z]{2,})\b|\s+•|\Z|\.\s+(?:[A-Za-z]{2,})\b|[,;](?:\s|$))"
)
# Only after line start or sentence punctuation — avoids stripping "costs & Benefits".
# Use [ \t]+ between words only (\s would span newlines and swallow the next line).
_AMP_SECTION_HEADING = re.compile(
    r"(?m)(?:^|(?<=[.!?•\n]))[ \t]*&[ \t]+"
    r"(?:[A-Z][a-z]+|[A-Z]{2,}\b)(?:[ \t]+(?:[A-Z][a-z]+|[A-Z]{2,}\b))*"
    + _AMP_HEADING_LA
)
_AMP_HEADING_LINE = re.compile(
    r"(?im)^\s*&\s+"
    r"(?:[A-Z][a-z]+|[A-Z]{2,}\b)(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}\b))*\s*$"
)


def strip_handbook_layout_artifacts(text: str) -> str:
    """
    Remove handbook PDF running headers / page lines that appear inside answers
    (verbatim retrieval or model copying chunk banners).
    """
    t = (text or "").strip()
    if not t:
        return t
    t = _HAND_BOOK_PAGE_EQUALS.sub(" ", t)
    prev = None
    while prev != t:
        prev = t
        t = _HAND_BOOK_PAGE_PIPE.sub(" ", t)
        t = _HAND_BOOK_PAGE_PIPE_NORMAL.sub(" ", t)
        t = _HAND_BOOK_CHAPTER_INLINE.sub(" ", t)
        t = _AMP_SECTION_HEADING.sub(" ", t)
        t = _AMP_HEADING_LINE.sub("", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    return t.strip()


def _strip_inline_markdown_noise(text: str) -> str:
    t = text.replace("**", "").replace("*", "")
    return t


def _heading_display_text(line: str) -> str:
    s = line.strip()
    m = re.match(r"^#{1,6}\s+(.*)$", s)
    if m:
        return m.group(1).strip().rstrip(":").strip()
    return s.rstrip(":").strip()


def _is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.match(r"^#{1,6}\s+\S", s):
        return True
    low = s.lower()
    if low == "sources:" or low.startswith("sources:"):
        return False
    if re.match(r"^\d+[\.\)]\s", s):
        return False
    if len(s) > 120:
        return False
    if s.endswith(":"):
        return True
    if len(s) <= 42 and s.isupper() and any(c.isalpha() for c in s):
        return True
    return False


def _format_sources_block(block: str) -> str:
    """Turn a trailing 'Sources:' section into a compact markdown subsection."""
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if not lines:
        return ""
    first = lines[0]
    rest = lines[1:] if len(lines) > 1 else []
    low = first.lower()
    if not low.startswith("sources"):
        rest = lines
    bullets: List[str] = []
    for ln in rest:
        if re.match(r"^\d+[\.\)]\s", ln):
            body = re.sub(r"^\d+[\.\)]\s+", "", ln).strip()
            if body:
                bullets.append(f"- {body}")
        elif re.match(r"^[-*]\s+", ln):
            bullet_body = re.sub(r"^[-*]\s+", "", ln).strip()
            bullets.append(f"- {bullet_body}")
        else:
            bullets.append(f"- {ln}")
    if not bullets:
        return "### Sources\n"
    return "### Sources\n\n" + "\n".join(bullets)


def _format_main_body(text: str) -> str:
    lines = text.splitlines()
    out: List[str] = []
    para: List[str] = []

    def flush_para() -> None:
        nonlocal para
        if not para:
            return
        p = " ".join(x.strip() for x in para if x.strip())
        if p:
            out.append(p)
            out.append("")
        para = []

    for raw in lines:
        line = raw.strip()
        if not line:
            flush_para()
            continue
        if _is_heading_line(line):
            flush_para()
            out.append(f"## {_heading_display_text(line)}")
            out.append("")
            continue
        if re.match(r"^\d+[\.\)]\s+", line):
            flush_para()
            m = re.match(r"^(\d+)[\.\)]\s+(.*)$", line)
            if m:
                out.append(f"{m.group(1)}. {m.group(2).strip()}")
                out.append("")
            continue
        if re.match(r"^[-*]\s+", line):
            flush_para()
            body = re.sub(r"^[-*]\s+", "", line).strip()
            out.append(f"- {body}")
            out.append("")
            continue
        para.append(line)

    flush_para()
    return "\n".join(out).rstrip()


def beautify_llm_response(text: str) -> str:
    """
    Produce display-oriented markdown: ## section headings, paragraphs, numbered/bullet lists,
    and a ### Sources subsection when present.
    """
    t = (text or "").strip()
    if not t or t == UNKNOWN_POLICY_PHRASE:
        return t

    t = _strip_inline_markdown_noise(t)
    t = strip_handbook_layout_artifacts(t)

    m = re.search(r"(?im)^Sources:\s*$", t)
    if m:
        main = t[: m.start()].strip()
        sources_raw = t[m.start() :].strip()
    else:
        main = t.strip()
        sources_raw = ""

    main_fmt = _format_main_body(main)
    if sources_raw:
        src_fmt = _format_sources_block(sources_raw)
        return f"{main_fmt}\n\n{src_fmt}".strip()
    return main_fmt


_MD_SOURCES_TAIL = re.compile(r"(?ms)^###\s+Sources\s*\n.*\Z", re.IGNORECASE)
_PLAIN_SOURCES_TAIL = re.compile(r"(?is)\n\s*Sources:\s*\n[\s\S]*\Z")


def strip_trailing_source_page_footer(text: str) -> str:
    """Remove model-appended 'Source section … Page number N' at end of answer body."""
    t = (text or "").rstrip()
    m = _TAIL_SOURCE_PAGE.search(t)
    if m:
        return t[: m.start()].rstrip()
    m_loose = _TAIL_SOURCE_PAGE_LOOSE.search(t)
    if m_loose:
        return t[: m_loose.start()].rstrip()
    # Before _TAIL_PAGE_ONLY: that pattern only removes "Page number N", which would leave a
    # dangling "Source …" when the model omits the word "section".
    m_short = _TAIL_SOURCE_SHORT.search(t)
    if m_short:
        return t[: m_short.start()].rstrip()
    m2 = _TAIL_PAGE_ONLY.search(t)
    if m2:
        return t[: m2.start()].rstrip()
    # Last resort: last " Source section " in string whose tail ends with Page number <digits>
    for m3 in reversed(list(re.finditer(r"(?is)\s+Source\s+section\s+", t))):
        tail = t[m3.start() :]
        if re.search(r"(?is)Page\s+number\s+\d{1,5}\s*\Z", tail):
            return t[: m3.start()].rstrip()
    return t


def is_fallback_answer(t: str) -> bool:
    """True when the model returned a policy fallback (no grounded answer)."""
    s = (t or "").strip()
    if not s:
        return True
    if s == UNKNOWN_POLICY_PHRASE:
        return True
    low = s.lower().rstrip(".")
    return low == "contact the hr department"


# Backwards compatibility for internal checks
_is_fallback_answer = is_fallback_answer


def _break_inline_bullet_chunks(body: str) -> str:
    """
    Models often emit one long line with ' • ' between list items. Split into real lines so
    plain text and HTML lists render readably (not as a single paragraph).
    """
    b = (body or "").strip()
    if " • " not in b:
        return b
    chunks = re.split(r"\s+•\s+", b)
    if len(chunks) <= 1:
        return b
    head = chunks[0].strip()
    bullets = [f"• {c.strip()}" for c in chunks[1:] if c.strip()]
    if not bullets:
        return b
    return head + "\n\n" + "\n".join(bullets)


def _preprocess_body_for_structure(body: str) -> str:
    """
    When the model returns one long line, break before • bullets and '& Topic' so
    markdownish HTML can emit <ul> and separate paragraphs.
    """
    t = (body or "").strip()
    if not t:
        return t
    t = _break_inline_bullet_chunks(t)
    t = re.sub(r"([.!?])\s+•\s+", r"\1\n• ", t)
    t = re.sub(r"([a-zA-Z0-9)%])\s+•\s+", r"\1\n• ", t)
    t = re.sub(r"\.\s+&\s+", ".\n& ", t)
    return t


def clean_answer_body_only(answer: str, citations: List[Dict[str, Any]]) -> str:
    """
    Raw answer text only: strips trailing model footer, markdown Sources, formats bullets.
    Does NOT append Source / Page / closing line (use format_plain_answer_with_metadata for that).
    """
    t = (answer or "").strip()
    if not t or is_fallback_answer(t):
        return t
    t = strip_handbook_layout_artifacts(t)
    body = strip_trailing_source_page_footer(t)
    if citations:
        body = _MD_SOURCES_TAIL.sub("", body).rstrip()
        body = _PLAIN_SOURCES_TAIL.sub("", body).rstrip()
    body = _break_inline_bullet_chunks(body)
    body = re.sub(r"([.!?])\s+•\s+", r"\1\n\n• ", body)
    body = re.sub(r"([a-zA-Z0-9)%])\s+•\s+", r"\1\n• ", body)
    body = re.sub(r"\.\s+&\s+", ".\n\n& ", body)
    return body.strip()


def format_plain_answer_with_metadata(
    body: str,
    citations: List[Dict[str, Any]],
    *,
    append_source_metadata: bool = True,
) -> str:
    """Plain text with optional Source / Page number / closing line (for JSON `answer` field)."""
    body = (body or "").strip()
    if not body:
        return body
    if is_fallback_answer(body):
        return body
    if not append_source_metadata:
        return body
    src, page = _primary_source_from_citations(citations)
    return f"{body}\n\nSource: {src}\n\nPage number: {page}\n\n{CLOSING_LINE}"


def clean_answer_plain_for_client(
    answer: str,
    citations: List[Dict[str, Any]],
    *,
    append_source_metadata: bool = True,
) -> str:
    """
    Full plain-text answer for APIs: cleaned body + optional Source + Page number + closing line.
    """
    body = clean_answer_body_only(answer, citations)
    if not body or is_fallback_answer(body):
        return body
    return format_plain_answer_with_metadata(
        body, citations, append_source_metadata=append_source_metadata
    )


def _markdownish_to_html(body: str) -> str:
    """Convert beautifier-style markdown to safe HTML (headings, lists, paragraphs)."""
    body = (body or "").strip()
    if not body:
        return ""
    lines = body.splitlines()
    parts: List[str] = []
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i]
        line = raw.rstrip()
        if not line.strip():
            i += 1
            continue
        s = line.strip()
        if s.startswith("## ") and not s.startswith("### "):
            parts.append(f"<h2>{html.escape(s[3:].strip())}</h2>")
            i += 1
            continue
        if s.startswith("### "):
            parts.append(f"<h3>{html.escape(s[4:].strip())}</h3>")
            i += 1
            continue
        if re.match(r"^[-*]\s+", s) or s.startswith("•"):
            items: List[str] = []
            while i < n:
                ln = lines[i].strip()
                if re.match(r"^[-*]\s+", ln):
                    b = re.sub(r"^[-*]\s+", "", ln).strip()
                    items.append(f"<li>{html.escape(b)}</li>")
                    i += 1
                elif ln.startswith("•"):
                    b = ln.lstrip("•").strip()
                    items.append(f"<li>{html.escape(b)}</li>")
                    i += 1
                elif not ln.strip():
                    i += 1
                    break
                else:
                    break
            parts.append("<ul>" + "".join(items) + "</ul>")
            continue
        if re.match(r"^\d+[\.\)]\s+", s):
            items = []
            while i < n:
                ln = lines[i].strip()
                mnum = re.match(r"^(\d+)[\.\)]\s+(.*)$", ln)
                if mnum:
                    items.append(f"<li>{html.escape(mnum.group(2).strip())}</li>")
                    i += 1
                elif not ln:
                    i += 1
                    break
                else:
                    break
            parts.append("<ol>" + "".join(items) + "</ol>")
            continue
        para_lines = [s]
        i += 1
        while i < n:
            nxt = lines[i].strip()
            if not nxt:
                i += 1
                break
            if (
                nxt.startswith("#")
                or re.match(r"^[-*]\s+", nxt)
                or nxt.startswith("•")
                or re.match(r"^\d+[\.\)]\s+", nxt)
            ):
                break
            para_lines.append(nxt)
            i += 1
        ptext = " ".join(x.strip() for x in para_lines if x.strip())
        if ptext:
            parts.append(f"<p>{html.escape(ptext)}</p>")

    return "\n".join(parts)


def _primary_source_from_citations(citations: List[Dict[str, Any]]) -> Tuple[str, str]:
    """(source_label, page_label) for display; uses first citation with any metadata."""
    if not citations:
        return "Unknown", "Unknown"
    for c in citations:
        sec = (c.get("source_section") or "").strip()
        pg = c.get("page")
        if sec or pg is not None:
            src = sec if sec else "Unknown"
            page_str = str(pg) if pg is not None else "Unknown"
            return src, page_str
    c0 = citations[0]
    sec = (c0.get("source_section") or "").strip() or "Unknown"
    pg = c0.get("page")
    page_str = str(pg) if pg is not None else "Unknown"
    return sec, page_str


def build_chat_answer_html(
    *,
    answer: str,
    citations: List[Dict[str, Any]],
    append_source_metadata: bool = True,
) -> str:
    """
    HTML for chat UI. Pass **body-only** text from `clean_answer_body_only` (not the full plain
    string that already includes Source / Page / closing line), so metadata is not duplicated.

    Layout: answer paragraph(s) first, then optionally Source: / Page number: lines and closing.
    """
    t = (answer or "").strip()
    if not t:
        return ""

    if _is_fallback_answer(t):
        return f'<div class="chat-answer"><p class="chat-answer-fallback">{html.escape(t)}</p></div>'

    body = strip_trailing_source_page_footer(t)
    if citations:
        body = _MD_SOURCES_TAIL.sub("", body).rstrip()
        body = _PLAIN_SOURCES_TAIL.sub("", body).rstrip()

    body = _preprocess_body_for_structure(body)
    inner = _markdownish_to_html(body)
    if not inner:
        inner = f"<p>{html.escape(t)}</p>"

    if not append_source_metadata:
        return f'<div class="chat-answer">\n{inner}\n</div>'

    src_label, page_label = _primary_source_from_citations(citations)

    meta = (
        '<div class="chat-source-meta">'
        f'<p class="chat-source-line"><strong>Source:</strong> {html.escape(src_label)}</p>'
        f'<p class="chat-source-line"><strong>Page number:</strong> {html.escape(page_label)}</p>'
        f'<p class="chat-answer-closing">{html.escape(CLOSING_LINE)}</p>'
        "</div>"
    )

    extra = ""
    if len(citations) > 1:
        bits: List[str] = []
        for c in citations[1:8]:
            s = (c.get("source_section") or "").strip() or "—"
            pg = c.get("page")
            ps = str(pg) if pg is not None else "—"
            bits.append(f"{html.escape(s)} (page {html.escape(ps)})")
        if bits:
            extra = (
                '<p class="chat-additional-sources"><em>Also cited: '
                + "; ".join(bits)
                + "</em></p>"
            )

    return f'<div class="chat-answer">\n{inner}\n{meta}{extra}\n</div>'
