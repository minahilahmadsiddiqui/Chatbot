# Chatbot backend

## End-to-end flow (user query → UI)

### 1. HTTP entry

- **`POST /api/chat/query/`** — JSON response (`chatbot/views.py` → `chat_query`).
- **`POST /api/chat/query/stream/`** — SSE (`chat_query_stream`).
- **`GET /api/chat/demo/`** — minimal browser page that POSTs to `/api/chat/query/` and renders `answer_html`.

### 2. RAG orchestration (`chatbot/services/rag_service.py`)

`run_rag_query(query, top_k, threshold, …)`:

1. **Short-circuits** — obvious gibberish → policy phrase; greetings → canned reply.
2. **Embed query** — `embeddings_service.get_embeddings` (OpenRouter embeddings).
3. **Retrieve** — `QdrantService` hybrid semantic + lexical merge, filter by threshold, rerank, pack **context chunks** (text + metadata).
4. **Answer text** (main branch, strict mode on by default):
   - **Extractive** — sentences chosen verbatim from chunks (`_extractive_answer_with_sources_from_context`). No chat LLM.  
     - Logged to **`logs/answer_raw_before_ui.log`** (via `append_raw_answer_before_ui_processing`).
   - **LLM fallback** — only if extractive is empty and `RAG_STRICT_FALLBACK_TO_LLM` is on: **`gemini_service.generate_answer`** (OpenRouter chat).
5. **Non-strict mode** (if `RAG_STRICT_NO_HALLUCINATE` is off) — chat LLM first; optional extractive fallback.

LangGraph is used when installed; otherwise the same logic runs in `_manual_pipeline`.

### 3. OpenRouter chat (`chatbot/services/gemini_service.py`)

When `generate_answer` runs:

1. Builds **context** string: `[Chunk N] …` lines from chunk texts.
2. **Messages sent to the model** (initial request only; see `generate_answer`):
   - **`role: system`** — Either **strict-from-context** copy (`prefer_answer_from_context=True`, usual LLM fallback) or **handbook-only** copy (`prefer_answer_from_context=False`). Both include: role as employee handbook assistant, concision rules, grounding rules, and the exact unknown-policy phrase `UNKNOWN_POLICY_PHRASE` where relevant.
   - **`role: user`** — Fixed skeleton:
     - `Context (authoritative):` + joined chunk lines (`[Chunk 1] …`, blank line between chunks).
     - `Question:` + user question.
     - `Formatting rules:` (numbered points, blank lines between points, no `*`/`**`, no unrelated sections, full sentences, trailing `Sources:` listing `[Chunk N]`).
3. **`chat.completions.create`** (OpenRouter) with `temperature` / `max_tokens` from the caller (settings-driven for RAG).
4. **Logged** — full initial prompt → **`logs/openrouter_prompt.log`** (same toggle as raw response). Assistant reply → **`logs/openrouter_raw.log`** (before summarize + `beautify_llm_response`).
5. Optional **continuation** rounds if `finish_reason == "length"` (different message list; not re-logged to `openrouter_prompt.log`).
6. Optional **second LLM** — `summarize_llm_answer_for_display` (shorten) if `RAG_LLM_POST_SUMMARY` and length threshold met.
7. **`beautify_llm_response`** (`response_beautify_service`) — markdown-ish structure for display.

Returned string from `run_rag_query` is the **final answer text** (extractive or post–LLM beautify).

### 4. API response shaping (`chatbot/views.py`)

After `run_rag_query`:

1. **`clean_answer_body_only(ans_raw, citations)`** — strip handbook junk, inline bullets, etc.; body only.
2. **`format_plain_answer_with_metadata(ans_body, citations)`** — plain `answer` with Source / Page / closing line.
3. **`build_chat_answer_html(ans_body, citations)`** — HTML for rich UI (`answer_html`).
4. **`ChatMessage`** row persisted with plain `response_text`.

**Client usage:** render **`answer_html`** as HTML (e.g. `innerHTML`); use **`answer`** for plain/multiline copy. SSE sends `answer` and `answer_html` events.

### 5. Ingestion (separate path)

Upload endpoints use **`text_splitter`**, **`embeddings_service`**, **`qdrant_service`** to chunk, embed, and upsert — not on every chat query.

---

## Log files (optional)

| File | Content |
|------|--------|
| `logs/openrouter_prompt.log` | Initial main chat request: `model=…`, then **system** and **user** strings exactly as sent to OpenRouter. If the file only shows `#` comment lines, your traffic is **extractive-only** (no chat LLM). |
| `logs/openrouter_raw.log` | Exact assistant `message.content` from that call (plus continuations joined), before summarize + `beautify_llm_response`. Same: stubs only until the chat model runs. |
| `logs/answer_raw_before_ui.log` | Extractive RAG answers only (no LLM). |

On each `/api/chat/query` (and stream), `ensure_raw_answer_log_dir()` creates `logs/` and seeds `openrouter_*.log` when missing.

Toggle with **`LOG_RAW_LLM_RESPONSE`**. Override paths: **`OPENROUTER_PROMPT_LOG_PATH`**, **`OPENROUTER_RAW_LOG_PATH`**, **`LLM_RAW_LOG_PATH`**.