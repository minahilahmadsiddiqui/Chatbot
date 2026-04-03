from __future__ import annotations

import re
from typing import List, Dict, Any

from django.conf import settings
from openai import OpenAI

UNKNOWN_POLICY_PHRASE = "The document doesn't mention it, contact the HR department."


def _openrouter_client() -> OpenAI:
    api_key = getattr(settings, "OPENROUTER_API_KEY", None)
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    base_url = getattr(settings, "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    timeout = float(getattr(settings, "OPENROUTER_HTTP_TIMEOUT_SEC", 120))
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def _beautify_answer(text: str) -> str:
    """
    Normalize model output into readable markdown-style sections and bullets.
    Keeps content unchanged as much as possible while fixing jumbled formatting.
    """
    t = (text or "").strip()
    if not t:
        return t
    if t == UNKNOWN_POLICY_PHRASE:
        return t

    # Force plain-text output (no markdown symbols) and clear line-by-line points.
    t = t.replace("**", "")
    t = t.replace("*", "")

    # Break inline list markers into separate lines.
    t = re.sub(r"([^\n])\s+(\d+[\.\)])\s+", r"\1\n\2 ", t)
    t = re.sub(r"([^\n])\s+-\s+", r"\1\n- ", t)

    # Normalize common bullet markers to numbered points.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    out: List[str] = []
    num = 1
    for ln in lines:
        # Keep section-like labels ending with ':'
        if ln.endswith(":"):
            out.append(ln)
            out.append("")
            num = 1
            continue

        # Remove existing leading markers.
        ln = re.sub(r"^(?:-|\d+[\.\)])\s*", "", ln).strip()
        if not ln:
            continue

        out.append(f"{num}. {ln}")
        out.append("")  # one blank line after each point
        num += 1

    return "\n".join(out).strip()


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
                        "Use plain text only (no * or **). Numbered points are fine when they aid scanning.\n"
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
    content = (response.choices[0].message.content or "").strip()
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
        continuation = (follow_up.choices[0].message.content or "").strip()
        if not continuation:
            break
        content = f"{content}\n{continuation}".strip()
        finish_reason = getattr(follow_up.choices[0], "finish_reason", None)
        continuation_count += 1

    if not content:
        return UNKNOWN_POLICY_PHRASE
    out = _beautify_answer(content)
    if out.strip() == UNKNOWN_POLICY_PHRASE:
        return out
    if getattr(settings, "RAG_LLM_POST_SUMMARY", True):
        min_chars = int(getattr(settings, "RAG_SUMMARIZE_MIN_INPUT_CHARS", 200))
        if len(out) >= min_chars:
            body = _strip_trailing_sources_block(out)
            if len(body.strip()) < 20:
                body = out
            summarized = summarize_llm_answer_for_display(question=question, draft_body=body)
            if summarized.strip():
                out = summarized.strip()
    return out


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

