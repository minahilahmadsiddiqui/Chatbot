from __future__ import annotations

import re
from typing import List, Optional, Tuple

import tiktoken
_ENCODING = None

# Chapter headings (line-based), e.g. "Chapter 5: Benefits", "CHAPTER IV – Policy", "Ch. 3 Leave"
_CHAPTER_HEADING_RE = re.compile(
    r"(?i)^\s*(?:chapter|ch\.)\s+"
    r"(?:(\d{1,3})\b|([ivxlcdm]{1,12})\b|"
    r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b)"
    r"\s*[\.:\-\u2013\u2014]?\s*"
    r"(.*)$"
)
# "Chapter: Introduction" or "CHAPTER — Overview"
_CHAPTER_UNNUMBERED_RE = re.compile(
    r"(?i)^\s*chapter\s*[:\-\u2013\u2014]\s*(.{3,120})\s*$"
)

_SECTION_LINE_RE = re.compile(
    r"(?i)^\s*(?:SOURCE[_\s]*SECTION|SECTION)\s*[:=\-]\s*(.+?)\s*$"
)
def parse_standalone_page_line(ln: str) -> Optional[int]:
    """
    Recognize common page header lines only (whole line), e.g.:
    PAGE_NUMBER: 18, PAGE NUMBER = 18, PAGE: 18, Page 48, === Page 9 ===
    """
    s = (ln or "").strip()
    patterns = [
        # Exported PDFs / Word: "=== Page 9 ===" or "==== Page 12 ===="
        re.compile(r"(?i)^=+\s*page\s+(\d{1,5})\s*=+\s*$"),
        re.compile(r"(?i)^page[\s_]*number\s*[:=\-]+\s*(\d{1,5})\s*$"),
        re.compile(r"(?i)^page\s*[:=\-]+\s*(\d{1,5})\s*$"),
        re.compile(r"(?i)^page\s+(\d{1,5})\s*$"),
    ]
    for p in patterns:
        m = p.match(s)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                return None
    return None
# Inline / decorative lines often found in exported handbooks, e.g.:
# "43 | P age === Page 44 === Gym Fee Reimbursement 14.7 Purpose ..."
_PAGE_EMBEDDED_RE = re.compile(
    r"(?i)^.{0,120}?Page\s+(\d{1,5})\s*(?:={2,}\s*)(.+)$"
)
# Numbered policy headings on their own line, e.g. "14.7 Purpose" or "3.6 Gym policy"
_IMPLICIT_SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+\.\d+(?:\.\d+)?)\s+([A-Z][^\n]{0,120})\s*$"
)


def line_looks_like_toc_leader(ln: str) -> bool:
    """Table-of-contents lines use dot leaders (....) or many periods."""
    if not ln:
        return False
    if re.search(r"\.{4,}", ln):
        return True
    if ln.count(".") >= 8 and re.search(r"\.{2,}", ln):
        return True
    return False


def text_contains_toc_dot_run(text: str) -> bool:
    return bool(text and re.search(r"\.{4,}", text))


def normalize_handbook_text(text: str) -> str:
    """
    Normalize upload text but keep newlines so SOURCE_SECTION / PAGE_NUMBER blocks parse.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]
    out: List[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 2:
                out.append("")
            continue
        blank_run = 0
        out.append(ln.strip())
    return "\n".join(out).strip()


def handbook_has_section_markers(text: str) -> bool:
    for ln in text.split("\n"):
        if _SECTION_LINE_RE.match(ln.strip()):
            return True
    return False


def handbook_has_chapter_markers(text: str) -> bool:
    """True when lines look like 'Chapter N: Title' style headings."""
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln or len(ln) > 200:
            continue
        if _CHAPTER_HEADING_RE.match(ln) or _CHAPTER_UNNUMBERED_RE.match(ln):
            return True
    return False


def handbook_has_auto_structure(text: str) -> bool:
    """
    True when the file looks like an exported handbook (pages / numbered sections / chapters)
    but does not use SOURCE_SECTION: lines. Enables line-aware chunking + metadata.
    """
    if handbook_has_chapter_markers(text):
        return True
    for raw in text.split("\n"):
        ln = raw.strip()
        if not ln:
            continue
        if parse_standalone_page_line(ln) is not None or _PAGE_EMBEDDED_RE.match(ln):
            return True
        if _IMPLICIT_SECTION_HEADING_RE.match(ln):
            return True
    return False


def _chapter_label_from_line(ln: str) -> str:
    """Normalize a matched chapter line to a stable title string."""
    m = _CHAPTER_HEADING_RE.match(ln.strip())
    if m:
        arabic = (m.group(1) or "").strip()
        roman = (m.group(2) or "").strip()
        word = (m.group(3) or "").strip()
        tail = (m.group(4) or "").strip()
        if arabic:
            core = f"Chapter {arabic}"
        elif roman:
            core = f"Chapter {roman.upper()}"
        elif word:
            core = f"Chapter {word.title()}"
        else:
            core = "Chapter"
        return f"{core}: {tail}".strip() if tail else core
    m2 = _CHAPTER_UNNUMBERED_RE.match(ln.strip())
    if m2:
        t = (m2.group(1) or "").strip()
        return f"Chapter: {t}" if t else ""
    return ""


def infer_page_from_chunk_text(text: str) -> Optional[int]:
    """Best-effort page from chunk text (used for legacy flat uploads)."""
    if not text:
        return None
    try:
        m = re.search(r"={2,}\s*page\s+(\d{1,5})\b", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"\bpage\s*[:#]?\s*(\d{1,5})\b", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        m = re.search(r"\bp\s+age\s+(\d{1,5})\b", text, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
    except (TypeError, ValueError):
        return None
    return None


def infer_section_from_chunk_text(text: str) -> Optional[str]:
    """Best-effort section title from chunk text (used for legacy flat uploads)."""
    if not text:
        return None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    has_toc = text_contains_toc_dot_run(text)

    page_idx = None
    page_re = re.compile(r"\bpage\s*[:#]?\s*(\d{1,5})\b", flags=re.IGNORECASE)
    for i, ln in enumerate(lines):
        if page_re.search(ln) or re.search(r"={2,}\s*page\s+\d{1,5}\b", ln, flags=re.IGNORECASE):
            page_idx = i
            break

    if page_idx is not None and page_idx > 0:
        candidate = lines[page_idx - 1]
        if line_looks_like_toc_leader(candidate):
            candidate = ""
        if (
            candidate
            and 3 <= len(candidate) <= 120
            and not candidate.lower().startswith("page")
            and not candidate.endswith(".")
        ):
            return candidate

    # With TOC / dot leaders in the chunk, do not grab the first "18.5 Foo" match — that is
    # almost always a table-of-contents row, not the policy body.
    if not has_toc:
        m = re.search(
            r"\b(\d+(?:\.\d+)+)\s+([A-Za-z][^\n]{0,80})",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            num = m.group(1).strip()
            title_tail = m.group(2).strip()
            title_tail = re.sub(r"[,:;\.].*$", "", title_tail).strip()
            if title_tail:
                return f"{num} {title_tail}".strip()

    m = re.search(
        r"\b(?:section|policy|chapter)\s*[:\-]\s*([^\n]{3,120})",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r"\.{3,}.*$", "", candidate).strip()
        if candidate and 3 <= len(candidate) <= 120 and not line_looks_like_toc_leader(candidate):
            return candidate

    return None


def infer_chapter_from_chunk_text(text: str) -> str:
    """Best-effort chapter line from chunk text (legacy flat uploads)."""
    for ln in (text or "").splitlines():
        lab = _chapter_label_from_line(ln.strip())
        if lab:
            return lab
    return ""


def infer_legacy_chunk_metadata(text: str) -> Tuple[str, Optional[int], str]:
    """Metadata for unstructured uploads: section title, page, chapter label."""
    sec = infer_section_from_chunk_text(text)
    return (sec or "", infer_page_from_chunk_text(text), infer_chapter_from_chunk_text(text))


def split_auto_structured_into_embedding_chunks(
    text: str,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[str, int, str, Optional[int], str]]:
    """
    Line-aware chunking: chapters (Chapter N: Title), SOURCE_SECTION, numbered
    headings (14.7 Purpose), and page markers. Each embedding row carries
    source_section (immediate heading), page_number, and chapter_name (parent chapter).
    """
    results: List[Tuple[str, int, str, Optional[int], str]] = []
    current_section = ""
    current_chapter = ""
    current_page: Optional[int] = None
    body_lines: List[str] = []

    def flush_current_block() -> None:
        if not current_section:
            return
        local_lines = list(body_lines)
        while local_lines and not local_lines[0].strip():
            local_lines.pop(0)
        while local_lines and not local_lines[-1].strip():
            local_lines.pop()
        body = "\n".join(local_lines).strip()
        if not body:
            return
        sub_chunks, sub_counts = split_text_into_token_chunks(
            body,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        ch_out = current_chapter or ""
        for chunk_text, tc in zip(sub_chunks, sub_counts):
            results.append((chunk_text, tc, current_section, current_page, ch_out))

    for raw_ln in text.split("\n"):
        ln = raw_ln.strip()
        sec_m = _SECTION_LINE_RE.match(ln)
        if sec_m:
            flush_current_block()
            current_section = (sec_m.group(1) or "").strip()
            current_page = None
            body_lines = []
            continue

        if line_looks_like_toc_leader(ln):
            continue

        ch_label = _chapter_label_from_line(ln)
        if ch_label:
            flush_current_block()
            current_chapter = ch_label
            current_section = ch_label
            body_lines = []
            continue

        imp = _IMPLICIT_SECTION_HEADING_RE.match(ln)
        if imp:
            flush_current_block()
            num = (imp.group(1) or "").strip()
            tail = (imp.group(2) or "").strip()
            # One line may combine heading + body, e.g. "14.8 Scope This SOP applies..."
            parts = tail.split(None, 1)
            if len(parts) >= 2:
                current_section = f"{num} {parts[0]}".strip()
                body_lines = [parts[1]] if parts[1] else []
            else:
                current_section = ln.strip()
                body_lines = []
            continue

        if not current_section:
            pv = parse_standalone_page_line(ln)
            if pv is not None:
                current_page = pv
                current_section = "Handbook"
                current_chapter = ""
                continue
            emb = _PAGE_EMBEDDED_RE.match(ln)
            if emb:
                try:
                    current_page = int(emb.group(1))
                except Exception:
                    current_page = None
                rest = (emb.group(2) or "").strip()
                title_guess = _title_before_numbered_clause(rest)
                current_section = title_guess or "Handbook"
                current_chapter = ""
                if rest:
                    body_lines.append(rest)
                continue
            continue

        pv = parse_standalone_page_line(ln)
        if pv is not None:
            current_page = pv
            continue

        emb = _PAGE_EMBEDDED_RE.match(ln)
        if emb:
            try:
                current_page = int(emb.group(1))
            except Exception:
                current_page = None
            rest = (emb.group(2) or "").strip()
            if rest:
                body_lines.append(rest)
            continue

        body_lines.append(raw_ln.rstrip())

    flush_current_block()
    return results


def _title_before_numbered_clause(rest: str) -> str:
    """
    From e.g. 'Gym Fee Reimbursement 14.7 Purpose ...' return 'Gym Fee Reimbursement'.
    """
    if not rest or len(rest) < 5:
        return ""
    m = re.match(
        r"^(.{3,90}?)\s+(\d+\.\d+(?:\.\d+)?)\s+[A-Za-z]",
        rest.strip(),
    )
    if not m:
        return ""
    cand = (m.group(1) or "").strip()
    if len(cand) >= 3:
        return cand
    return ""


def split_handbook_into_embedding_chunks(
    text: str,
    *,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[str, int, str, Optional[int], str]]:
    """
    Parse handbook .txt with blocks:

        SOURCE_SECTION: 3.6 Small title
        PAGE_NUMBER: 18
        Body text (multi-line). Token-chunked; each chunk keeps same section + page.

    Also accepts: SOURCE SECTION: ...  and  PAGE: 18  (case-insensitive).
    Returns [] if no SOURCE_SECTION lines found (caller should use legacy chunking).
    """
    if not handbook_has_section_markers(text):
        return []

    results: List[Tuple[str, int, str, Optional[int], str]] = []
    current_section = ""
    current_page: Optional[int] = None
    body_lines: List[str] = []

    def flush_current_block() -> None:
        if not current_section:
            return
        local_lines = list(body_lines)
        while local_lines and not local_lines[0].strip():
            local_lines.pop(0)
        while local_lines and not local_lines[-1].strip():
            local_lines.pop()
        body = "\n".join(local_lines).strip()
        if not body:
            return
        sub_chunks, sub_counts = split_text_into_token_chunks(
            body,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
        ch_out = _chapter_label_from_line(current_section) or ""
        for chunk_text, tc in zip(sub_chunks, sub_counts):
            results.append((chunk_text, tc, current_section, current_page, ch_out))

    for raw_ln in text.split("\n"):
        ln = raw_ln.strip()
        sec_m = _SECTION_LINE_RE.match(ln)
        if sec_m:
            # New section starts; flush previous body first.
            flush_current_block()
            current_section = (sec_m.group(1) or "").strip()
            current_page = None
            body_lines = []
            continue

        if not current_section:
            # Ignore any preamble before the first section marker.
            continue

        if line_looks_like_toc_leader(ln):
            continue

        pv = parse_standalone_page_line(ln)
        if pv is not None:
            current_page = pv
            continue

        emb = _PAGE_EMBEDDED_RE.match(ln)
        if emb:
            try:
                current_page = int(emb.group(1))
            except Exception:
                current_page = None
            rest = (emb.group(2) or "").strip()
            if rest:
                body_lines.append(rest)
            continue

        body_lines.append(raw_ln.rstrip())

    # Flush final section block
    flush_current_block()

    return results


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


def truncate_text_to_token_budget(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens (tiktoken) from the start."""
    if max_tokens <= 0:
        return ""
    enc = _encoding()
    ids = enc.encode(text or "")
    if len(ids) <= max_tokens:
        return text or ""
    return enc.decode(ids[:max_tokens])


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

