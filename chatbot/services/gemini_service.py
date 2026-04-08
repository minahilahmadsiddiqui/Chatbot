from __future__ import annotations

import re
import time
from typing import List, Dict, Any, Optional

from django.conf import settings
from openai import OpenAI

from chatbot.services.response_beautify_service import beautify_llm_response

UNKNOWN_POLICY_PHRASE = "The document doesn't mention it. Contact the HR"

def _append_openrouter_initial_prompt(*, model: str, system_prompt: str, user_prompt: str) -> None:
    del model, system_prompt, user_prompt
    return


def _append_rag_llm_prompt_payload(
    *,
    model: str,
    question: str,
    context_chunks: List[Dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
) -> None:
    del model, question, context_chunks, system_prompt, user_prompt
    return


def _append_openrouter_raw_response(text: str) -> None:
    del text
    return


def ensure_raw_answer_log_dir() -> None:
    return


def append_raw_answer_before_ui_processing(text: str) -> None:
    del text
    return


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


def _strip_inline_chunk_tags(text: str) -> str:
    """
    Remove inline grounding tags like [Chunk 2] from model output before UI display.
    """
    t = text or ""
    t = re.sub(r"\s*\[Chunk\s+\d+\]\s*", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _kw(text: str) -> set[str]:
    toks = {t.lower() for t in re.findall(r"[a-zA-Z0-9]+", text or "")}
    stop = {
        "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
        "be", "this", "that", "it", "as", "at", "by", "from", "what", "how", "when", "where",
        "which", "who", "why", "does", "do", "did", "can", "could", "should", "would", "about",
    }
    return {t for t in toks if len(t) >= 3 and t not in stop}


def _topic_focused_chunks(question: str, chunks: List[Dict[str, Any]], *, max_chunks: int = 5) -> List[Dict[str, Any]]:
    q = _kw(question)
    if not q:
        return chunks[:max_chunks]
    # For handbook domains, these anchors are critical for topic routing and should
    # always contribute to overlap scoring when present in the query.
    domain_anchors = {
        t
        for t in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if t in {
            "policy", "policies", "employee", "employees", "reimbursement",
            "certification", "certifications", "leave", "benefits", "scope", "purpose",
        }
    }
    q = q.union(domain_anchors)
    scored = []
    for c in chunks:
        text = str(c.get("text") or "")
        sec = str(c.get("source_section") or "")
        terms = _kw(f"{sec} {text}")
        # Prefer section-title alignment strongly so nearby unrelated policies are less likely.
        sec_terms = _kw(sec)
        ov = len(q.intersection(terms))
        sec_ov = len(q.intersection(sec_terms))
        if ov <= 0:
            continue
        scored.append((sec_ov, ov, c))
    if not scored:
        return chunks[:max_chunks]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [c for _sec_ov, _ov, c in scored[:max(1, max_chunks)]]


def summarize_llm_answer_for_display(*, question: str, draft_body: str) -> str:
    """
    Second LLM call: produce a shorter version of the answer while keeping the same facts.
    """
    if not draft_body.strip() or draft_body.strip() == UNKNOWN_POLICY_PHRASE:
        return draft_body

    _sm = getattr(settings, "OPENROUTER_SUMMARY_MODEL", None)
    model = (str(_sm).strip() if _sm else "") or getattr(
        settings, "OPENROUTER_CHAT_MODEL", "arcee-ai/trinity-large-preview:free"
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
                        "You are a policy-summary editor for an employee handbook chatbot.\n"
                        "Rewrite the draft into a shorter employee-facing answer while preserving policy meaning exactly.\n"
                        "Target about 25-40% fewer words when possible.\n"
                        "Do not remove or alter any policy-critical detail: eligibility, scope, required approvals, "
                        "required documents, timelines/deadlines, amounts/percentages, limits/caps, conditions, "
                        "exceptions, and role/location-specific rules.\n"
                        "Do not add new facts, assumptions, interpretations, or legal advice.\n"
                        "Merge repetition and remove filler, but keep actionable steps in the original order when the draft is procedural.\n"
                        "If the draft contains uncertainty or conditional wording (e.g., may, must, only if, unless), preserve it.\n"
                        "Every sentence must be grammatically complete and standalone; no fragments.\n"
                        "Output only the shortened answer body. Do not include preamble, title, or Sources section."
                    )
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
    focused_chunks = _topic_focused_chunks(
        question,
        context_chunks,
        max_chunks=int(getattr(settings, "RAG_LLM_FOCUSED_CONTEXT_CHUNKS", 5)),
    )
    context_text = "\n\n".join(
        [
            f"[Chunk {i + 1} | Section: {str(c.get('source_section') or '').strip()} | Page: {str(c.get('page_number') or '').strip()}] {c.get('text', '')}".strip()
            for i, c in enumerate(focused_chunks)
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
            + "The user question must be answered using ONLY the provided Context excerpts.\n"
            "Follow this decision rule strictly:\n"
            "Step 1) Determine whether the Context contains direct evidence for the exact question.\n"
            "Direct evidence means the same policy topic and explicit matching facts (eligibility, limits, amounts, timelines, approvals, conditions, exceptions).\n"
            "Do NOT treat generic HR text, nearby sections, or loosely related topics as evidence.\n"
            "If evidence is missing, partial-but-not-on-topic, or ambiguous, respond with EXACTLY this sentence and nothing else:\n"
            f"\"{UNKNOWN_POLICY_PHRASE}\"\n"
            "Step 2) Only if direct evidence exists, answer using only those relevant lines/chunks.\n"
            "Never infer, guess, generalize, or combine unrelated chunks to fabricate an answer.\n"
            "Never use prior knowledge beyond the Context.\n"
            "If any requested detail is not explicitly present in Context, omit that detail.\n"
            "Prefer short, faithful paraphrase close to the original wording.\n"
            "Every sentence must be grammatically complete: do not paste mid-sentence fragments.\n"
        )
    else:
        system_prompt = (
            "You are an employee handbook assistant.\n"
            + concise
            + "Answer ONLY from the provided handbook excerpts.\n"
            "Use ONLY excerpts that match the question topic; ignore unrelated policies.\n"
            "Do NOT stitch text from different policy topics.\n"
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
        "6) Ground every point strictly in the provided context, but DO NOT print any chunk tags such as [Chunk N].\n"
        "7) Do not print internal evidence markers, citations, or references in the final answer text.\n"
        "8) If a point cannot be grounded in context, do not include that point.\n"
        "9) Output only the answer body in clean user-facing language. Do not add a Sources section.\n"
    )

    extra_headers = {
        "HTTP-Referer": str(getattr(settings, "OPENROUTER_REFERER", "http://localhost:8000")),
        "X-Title": str(getattr(settings, "OPENROUTER_TITLE", "chatbot-backend")),
    }
    _append_rag_llm_prompt_payload(
        model=model,
        question=question,
        context_chunks=focused_chunks,
        system_prompt=system_prompt,
        user_prompt=prompt,
    )
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
    content = _strip_inline_chunk_tags(content)
    if getattr(settings, "RAG_LLM_POST_SUMMARY", True):
        min_chars = int(getattr(settings, "RAG_SUMMARIZE_MIN_INPUT_CHARS", 200))
        if len(content) >= min_chars:
            body = _strip_trailing_sources_block(content)
            if len(body.strip()) < 20:
                body = content
            summarized = summarize_llm_answer_for_display(question=question, draft_body=body)
            if summarized.strip():
                content = summarized.strip()
    content = _strip_inline_chunk_tags(content)
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

