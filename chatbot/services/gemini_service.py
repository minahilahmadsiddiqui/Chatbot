from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

from django.conf import settings
from openai import OpenAI

from chatbot.services.response_beautify_service import beautify_llm_response

UNKNOWN_POLICY_PHRASE = "The document doesn't mention it. Contact the HR"

_raw_llm_log_lock = threading.Lock()


def _openrouter_raw_log_file_path() -> Optional[Path]:
    """
    Log file for OpenRouter chat completion text only (exact API message.content),
    before strip, summarize, or beautify_llm_response.
    Override: settings.OPENROUTER_RAW_LOG_PATH (full path to file).
    """
    if not bool(getattr(settings, "LOG_RAW_LLM_RESPONSE", True)):
        return None
    base_dir = getattr(settings, "BASE_DIR", None)
    if base_dir is None:
        return None
    configured = getattr(settings, "OPENROUTER_RAW_LOG_PATH", None)
    if configured:
        return Path(configured)
    return Path(base_dir) / "logs" / "openrouter_raw.log"


def _openrouter_prompt_log_file_path() -> Optional[Path]:
    """
    Log file: exact system + user messages for the *initial* main chat completion only.
    Override: settings.OPENROUTER_PROMPT_LOG_PATH.
    """
    if not bool(getattr(settings, "LOG_RAW_LLM_RESPONSE", True)):
        return None
    base_dir = getattr(settings, "BASE_DIR", None)
    if base_dir is None:
        return None
    configured = getattr(settings, "OPENROUTER_PROMPT_LOG_PATH", None)
    if configured:
        return Path(configured)
    return Path(base_dir) / "logs" / "openrouter_prompt.log"


def _append_openrouter_initial_prompt(*, model: str, system_prompt: str, user_prompt: str) -> None:
    """Append one record: model id + system + user content as sent to OpenRouter (initial call only)."""
    path = _openrouter_prompt_log_file_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = (
        f"model={model}\n"
        "----- system -----\n"
        f"{system_prompt}\n"
        "----- user -----\n"
        f"{user_prompt}\n"
        "----- end request -----\n\n"
    )
    with _raw_llm_log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(record)


def _append_openrouter_raw_response(text: str) -> None:
    """Append exact OpenRouter assistant text; no strip (only ensures trailing newline in file)."""
    path = _openrouter_raw_log_file_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _raw_llm_log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


def _raw_answer_log_file_path() -> Optional[Path]:
    """
    Path to the raw answer log file (no console).
    settings.LLM_RAW_LOG_PATH overrides; default BASE_DIR/logs/answer_raw_before_ui.log
    """
    if not bool(getattr(settings, "LOG_RAW_LLM_RESPONSE", True)):
        return None
    base_dir = getattr(settings, "BASE_DIR", None)
    if base_dir is None:
        return None
    configured = getattr(settings, "LLM_RAW_LOG_PATH", None)
    if configured:
        return Path(configured)
    return Path(base_dir) / "logs" / "answer_raw_before_ui.log"


_OPENROUTER_PROMPT_STUB = (
    "# OpenRouter prompt log — exact system + user for the main CHAT completion append below.\n"
    "# With RAG_STRICT_NO_HALLUCINATE (default), answers usually come from extractive retrieval only;\n"
    "# generate_answer is not called, so this file stays as this notice until an LLM path runs\n"
    "# (e.g. empty extractive + RAG_STRICT_FALLBACK_TO_LLM=1, or strict mode off).\n\n"
)
_OPENROUTER_RAW_STUB = (
    "# OpenRouter raw assistant replies (before summarize + beautify in gemini_service) append below.\n"
    "# Same as openrouter_prompt.log: no chat call → no entries here.\n\n"
)


def ensure_raw_answer_log_dir() -> None:
    """
    Create log dirs; seed openrouter_*.log with a header if missing (extractive-only runs never
    call generate_answer, so those files would not exist otherwise).
    """
    for getter in (_openrouter_prompt_log_file_path, _openrouter_raw_log_file_path, _raw_answer_log_file_path):
        path = getter()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    pp = _openrouter_prompt_log_file_path()
    rp = _openrouter_raw_log_file_path()
    if pp is None and rp is None:
        return
    with _raw_llm_log_lock:
        if pp is not None and not pp.exists():
            pp.write_text(_OPENROUTER_PROMPT_STUB, encoding="utf-8")
        if rp is not None and not rp.exists():
            rp.write_text(_OPENROUTER_RAW_STUB, encoding="utf-8")


def append_raw_answer_before_ui_processing(text: str) -> None:
    """
    Append extractive RAG answer text (no OpenRouter call) for audit.
    OpenRouter chat output is logged separately to openrouter_raw.log inside generate_answer.
    """
    path = _raw_answer_log_file_path()
    if path is None:
        return
    raw = (text or "").strip()
    if not raw:
        return
    ensure_raw_answer_log_dir()
    with _raw_llm_log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(raw)
            if not raw.endswith("\n"):
                f.write("\n")


def _openrouter_client() -> OpenAI:
    api_key = getattr(settings, "OPENROUTER_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    timeout = float(getattr(settings, "OPENROUTER_HTTP_TIMEOUT_SEC", 120))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _strip_trailing_sources_block(text: str) -> str:
    """Remove model-written Sources section so summarization targets the answer body only."""
    t = (text or "").rstrip()
    m = re.search(r"\n(?i)Sources:\s*\n[\s\S]*\Z", t)
    if m:
        return t[: m.start()].rstrip()
    return t


def summarize_llm_answer_for_display(*, question: str, draft_body: str) -> str:
    """
    Second LLM call: produce a shorter version of the answer while keeping the same facts.
    """
    if not draft_body.strip() or draft_body.strip() == UNKNOWN_POLICY_PHRASE:
        return draft_body

    _sm = getattr(settings, "OPENROUTER_SUMMARY_MODEL", None)
    model = (str(_sm).strip() if _sm else "") or getattr(
        settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash"
    )
    max_out = int(getattr(settings, "OPENROUTER_SUMMARY_MAX_OUTPUT_TOKENS", 384))
    client = _openrouter_client()
    extra_headers = {
        "HTTP-Referer": str(getattr(settings, "OPENROUTER_REFERER", "http://localhost:8000")),
        "X-Title": str(getattr(settings, "OPENROUTER_TITLE", "chatbot-backend")),
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You shorten handbook assistant replies for a chat UI.\n"
                        "Target roughly 25–40% fewer words than the draft when possible, without losing "
                        "any policy facts, numbers, dates, thresholds, steps, or exceptions.\n"
                        "Merge redundant sentences; drop filler and repetition. Do not add information.\n"
                        "Every sentence must be grammatically complete: do not start or end with a "
                        "mid-sentence fragment (no orphaned clauses).\n"
                        "Use plain text; optional ## headings for major sections when the draft has clear topics.\n"
                        "No * or **. Numbered points are fine when they aid scanning.\n"
                        "Output only the shortened answer — no preamble, title, or 'Sources:' section."
                    ),
                },
                {
                    "role": "user",
                    "content": f"User question:\n{question}\n\nDraft answer:\n{draft_body}",
                },
            ],
            temperature=0.08,
            max_tokens=max_out,
            extra_headers=extra_headers,
        )
        shorter = (response.choices[0].message.content or "").strip()
        if len(shorter) < 12:
            return draft_body
        return shorter
    except Exception:
        return draft_body


def generate_answer(
    *,
    question: str,
    context_chunks: List[Dict[str, Any]],
    fallback_phrase: str = "Contact the HR department",
    model: str = "google/gemini-2.5-flash",
    max_output_tokens: int = 512,
    temperature: float = 0.2,
    prefer_answer_from_context: bool = False,
) -> str:
    """
    Calls OpenRouter Chat Completions API.
    Response is grounded by prompt rules; fallback_phrase is kept only
    for compatibility with callers and should not normally be returned
    when context is present.
    """
    ensure_raw_answer_log_dir()
    client = _openrouter_client()

    # Keep the prompt tight and explicit; prefer grounded answers over fallback
    # when there is any relevant information in the context.
    context_text = "\n\n".join(
        [
            f"[Chunk {i + 1}] {c.get('text', '')}".strip()
            for i, c in enumerate(context_chunks)
            if c.get("text")
        ]
    )

    concise = (
        "Be concise: answer only what the question asks; omit unrelated policies, examples, "
        "and background that does not directly address the question.\n"
        "Prefer a tight summary over long quotations; include full detail only where the question requires it.\n"
    )

    if prefer_answer_from_context:
        system_prompt = (
            "You are an employee handbook assistant.\n"
            + concise
            + "The user question should be answered using ONLY the provided Context excerpts.\n"
            "If the excerpts contain ANY information that reasonably addresses the question "
            "(even partial, or spread across chunks), summarize that information clearly.\n"
            "Do not invent facts that are not supported by the Context.\n"
            f"Use the exact phrase \"{UNKNOWN_POLICY_PHRASE}\" ONLY when the excerpts do not "
            "contain relevant information for the question at all.\n"
            "Prefer quoting or paraphrasing closely from the excerpts; cite [Chunk N] in Sources.\n"
            "Every sentence must be grammatically complete: do not paste mid-sentence fragments.\n"
        )
    else:
        system_prompt = (
            "You are an employee handbook assistant.\n"
            + concise
            + "Answer ONLY from the provided handbook excerpts.\n"
            "Never invent or infer company policy.\n"
            f"If the handbook does not explicitly mention the policy, respond with exactly: \"{UNKNOWN_POLICY_PHRASE}\".\n"
            "Do not answer using loosely related policies.\n"
            "Prefer exact section wording over paraphrased assumptions.\n"
            "Quote or cite section titles when possible.\n"
            "If you provide any specific detail, it must appear verbatim in the provided Context.\n"
            "Every sentence must be grammatically complete: do not paste mid-sentence fragments.\n"
        )
    prompt = (
        "Context (authoritative):\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Formatting rules:\n"
        "1) Use numbered points (1., 2., 3.) for all key items.\n"
        "2) Put each point on a new line, and leave one blank line between points.\n"
        "3) Do NOT use markdown symbols such as * or **.\n"
        "4) Do not include unrelated policies or sections — only what answers the question.\n"
        "5) Every point must be full, grammatically complete sentences — no mid-sentence fragments.\n"
        "6) After the answer, add a `Sources:` section listing which [Chunk N] excerpts you relied on.\n"
    )

    extra_headers = {
        "HTTP-Referer": str(getattr(settings, "OPENROUTER_REFERER", "http://localhost:8000")),
        "X-Title": str(getattr(settings, "OPENROUTER_TITLE", "chatbot-backend")),
    }
    _append_openrouter_initial_prompt(model=model, system_prompt=system_prompt, user_prompt=prompt)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
        extra_headers=extra_headers,
    )
    first_msg = response.choices[0].message.content
    raw_pieces: List[str] = [first_msg if first_msg is not None else ""]
    content = (first_msg or "").strip()
    finish_reason = getattr(response.choices[0], "finish_reason", None)

    # Continuation rounds multiply latency; keep low (raise max_output_tokens instead).
    max_continuations = min(int(getattr(settings, "OPENROUTER_MAX_CONTINUATIONS", 1)), 3)
    continuation_count = 0
    while finish_reason == "length" and content and continuation_count < max_continuations:
        follow_up = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "Continue exactly from where you stopped. Do not repeat prior lines.",
                },
            ],
            temperature=temperature,
            max_tokens=max_output_tokens,
            extra_headers=extra_headers,
        )
        cont_msg = follow_up.choices[0].message.content
        raw_pieces.append(cont_msg if cont_msg is not None else "")
        continuation = (cont_msg or "").strip()
        if not continuation:
            break
        content = f"{content}\n{continuation}".strip()
        finish_reason = getattr(follow_up.choices[0], "finish_reason", None)
        continuation_count += 1

    if not content:
        return UNKNOWN_POLICY_PHRASE
    # Exact OpenRouter assistant text (joined), before strip/summarize/beautify in this function.
    openrouter_raw = "\n".join(raw_pieces)
    if openrouter_raw.strip():
        _append_openrouter_raw_response(openrouter_raw)
    if getattr(settings, "RAG_LLM_POST_SUMMARY", True):
        min_chars = int(getattr(settings, "RAG_SUMMARIZE_MIN_INPUT_CHARS", 200))
        if len(content) >= min_chars:
            body = _strip_trailing_sources_block(content)
            if len(body.strip()) < 20:
                body = content
            summarized = summarize_llm_answer_for_display(question=question, draft_body=body)
            if summarized.strip():
                content = summarized.strip()
    out = beautify_llm_response(content)
    return out if out.strip() else content


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
        prefer_answer_from_context=False,
    )

