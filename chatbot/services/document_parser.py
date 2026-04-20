from __future__ import annotations

import os
from io import BytesIO
from typing import Optional

from docx import Document as DocxDocument
from pypdf import PdfReader


def extract_text_from_upload(file_obj) -> Optional[str]:
    if not file_obj:
        return None

    filename = str(getattr(file_obj, "name", "") or "").lower()
    ext = os.path.splitext(filename)[1].strip().lower()
    raw = file_obj.read()
    if not raw:
        return None

    if ext == ".txt":
        try:
            return raw.decode("utf-8")
        except Exception:
            return raw.decode("latin-1", errors="ignore")

    if ext == ".pdf":
        try:
            reader = PdfReader(BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            out = "\n".join(pages).strip()
            return out or None
        except Exception:
            return None

    if ext == ".docx":
        try:
            doc = DocxDocument(BytesIO(raw))
            out = "\n".join(p.text for p in doc.paragraphs).strip()
            return out or None
        except Exception:
            return None

    return None

