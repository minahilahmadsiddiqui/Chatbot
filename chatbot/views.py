import hashlib
import json
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
from django.conf import settings
from django.contrib.auth.hashers import check_password, make_password
from django.core.mail import send_mail
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from openai import APIStatusError, AuthenticationError
from rest_framework import serializers, status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.response import Response
from drf_spectacular.utils import (
    OpenApiParameter,
    OpenApiExample,
    OpenApiResponse,
    OpenApiTypes,
    extend_schema,
    inline_serializer,
)

from chatbot.repositories.firestore_repository import FirestoreRepository
from chatbot.services.embeddings_service import get_embeddings
from chatbot.services.qdrant_service import QdrantService
from chatbot.services.response_beautify_service import (
    build_chat_answer_html,
    clean_answer_body_only,
    format_plain_answer_with_metadata,
)
from chatbot.services.gemini_service import ensure_raw_answer_log_dir
from chatbot.services.document_parser import extract_text_from_upload
from chatbot.services.payment_gateway_service import (
    create_provider_payment_intent,
    fetch_provider_payment_intent_for,
    parse_provider_webhook,
)
from chatbot.services.rag_service import FALLBACK_PHRASE, run_rag_query
from chatbot.services.telemetry_service import append_rag_telemetry
from chatbot.services.text_splitter import (
    handbook_has_auto_structure,
    handbook_has_section_markers,
    infer_legacy_chunk_metadata,
    normalize_handbook_text,
    normalize_text,
    split_auto_structured_into_embedding_chunks,
    split_handbook_into_embedding_chunks,
    split_text_into_token_chunks,
)
from chatbot.auth.jwt_auth import JWTAuthentication, IsAdmin, IsSuperAdmin, generate_access_token, generate_refresh_token

repo = FirestoreRepository()
PAID_BOT_AMOUNT_PKR = 1680
PAID_BOT_AMOUNT_USD = 6.0


def _current_admin_and_company_id(request) -> Tuple[Optional[Any], Optional[int], Optional[Response]]:
    admin_id = int(getattr(request.user, "id"))
    admin = repo.get_admin(admin_id)
    if not admin:
        return None, None, Response({"error": "Admin not found."}, status=status.HTTP_404_NOT_FOUND)
    if admin.company_id is None:
        return (
            admin,
            None,
            Response(
                {"error": "Admin is not linked to a company. Create company first."},
                status=status.HTTP_400_BAD_REQUEST,
            ),
        )
    return admin, int(admin.company_id), None


def _parse_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _current_admin_company_and_bot(
    request, *, bot_id_raw: Any = None
) -> Tuple[Optional[Any], Optional[int], Optional[Any], Optional[Response]]:
    admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return None, None, None, err
    if bot_id_raw is None:
        bot_id_raw = request.data.get("bot_id") or request.query_params.get("bot_id")
    bot_id = _parse_int(bot_id_raw)
    if bot_id is None:
        return (
            admin,
            company_id,
            None,
            Response({"error": "bot_id is required."}, status=status.HTTP_400_BAD_REQUEST),
        )
    bot = repo.get_bot(bot_id)
    if not bot or int(bot.company_id) != int(company_id):
        return (
            admin,
            company_id,
            None,
            Response({"error": "Bot not found."}, status=status.HTTP_404_NOT_FOUND),
        )
    return admin, company_id, bot, None


def _company_plan_max_bots(company_id: int) -> int:
    company = repo.get_company(int(company_id))
    if not company:
        return 1
    snap = repo.companies.document(str(company.id)).get()
    data = snap.to_dict() or {}
    plan_type = str(data.get("plan_type", "free")).strip().lower() or "free"
    if plan_type == "paid":
        return int(getattr(settings, "PAID_PLAN_MAX_BOTS", 10))
    return int(getattr(settings, "FREE_PLAN_MAX_BOTS", 1))


def _build_widget_script(*, bot_widget_key: str, request) -> str:
    base = request.build_absolute_uri("/").rstrip("/")
    script_src = f"{base}/api/public/widget.js"
    return (
        "<script>\n"
        f"  window.AcmeChatbotConfig = {{ botKey: '{bot_widget_key}' }};\n"
        "</script>\n"
        f"<script async src=\"{script_src}\"></script>"
    )


@extend_schema(exclude=True)
@api_view(["GET"])
@csrf_exempt
def public_widget_config(request):
    widget_key = str(request.query_params.get("widget_key") or "").strip()
    if not widget_key:
        return Response({"error": "widget_key is required."}, status=400)
    bot = repo.find_bot_by_widget_key(widget_key)
    if not bot:
        return Response({"error": "Invalid widget key."}, status=404)
    docs = repo.list_documents_for_bot(company_id=int(bot.company_id), bot_id=int(bot.id))
    prompt = str(bot.system_prompt or "").strip()
    first_line = ""
    if prompt:
        for line in prompt.splitlines():
            cleaned = line.strip()
            if cleaned:
                first_line = cleaned
                break
    if len(first_line) > 220:
        first_line = first_line[:220].rstrip() + "..."

    # Convert instruction-like system prompt lines into user-facing welcome copy.
    prompt_preview = first_line
    lowered = prompt_preview.lower()
    if lowered.startswith("you are "):
        prompt_preview = prompt_preview[8:].strip()
    if lowered.startswith("you are an "):
        prompt_preview = prompt_preview[11:].strip()
    if lowered.startswith("you are a "):
        prompt_preview = prompt_preview[10:].strip()
    if prompt_preview.lower().startswith("the "):
        prompt_preview = prompt_preview[4:].strip()
    if prompt_preview.lower().endswith("for this company."):
        prompt_preview = prompt_preview[: -len("for this company.")].strip()
    if prompt_preview.lower().endswith("for this company"):
        prompt_preview = prompt_preview[: -len("for this company")].strip()
    if prompt_preview and not prompt_preview.endswith("."):
        prompt_preview = f"{prompt_preview}."

    welcome_message = (
        f"Hi! I'm {bot.name}. I can help with {prompt_preview}"
        if prompt_preview
        else f"Hi! I'm {bot.name}. Ask me anything related to your company's uploaded knowledge base."
    )
    return Response(
        {
            "bot_id": int(bot.id),
            "bot_name": bot.name,
            "document_count": len(docs),
            "welcome_message": welcome_message,
            "theme": {
                "navy_deep": "hsl(226,55%,10%)",
                "navy": "hsl(226,45%,20%)",
                "navy_light": "hsl(226,35%,30%)",
                "mint": "hsl(160,62%,55%)",
                "mint_light": "hsl(160,55%,92%)",
            },
        }
    )


@extend_schema(exclude=True)
@api_view(["GET"])
@csrf_exempt
def public_widget_script(request):
    base = request.build_absolute_uri("/").rstrip("/")
    chat_url = f"{base}/api/public/chat/query/"
    config_url = f"{base}/api/public/widget/config/"
    js = f"""
(function () {{
  if (window.__acmeWidgetLoaded) return;
  window.__acmeWidgetLoaded = true;

  var cfg = window.AcmeChatbotConfig || {{}};
  var botKey = cfg.botKey || cfg.widgetKey || cfg.key;
  if (!botKey) {{
    console.error("Acme widget: missing botKey in window.AcmeChatbotConfig");
    return;
  }}

  var CHAT_URL = "{chat_url}";
  var CONFIG_URL = "{config_url}";
  var STORAGE_KEY = "acme_widget_history_" + botKey;
  var SESSION_KEY = "acme_widget_session_" + botKey;
  var WELCOME_KEY = "acme_widget_welcome_" + botKey;
  var dynamicWelcome = localStorage.getItem(WELCOME_KEY) || "Hello! How can I help you today?";

  function getSessionId() {{
    var sid = localStorage.getItem(SESSION_KEY);
    if (!sid) {{
      sid = "ws_" + Math.random().toString(36).slice(2) + Date.now().toString(36);
      localStorage.setItem(SESSION_KEY, sid);
    }}
    return sid;
  }}

  function getHistory() {{
    try {{
      var raw = localStorage.getItem(STORAGE_KEY);
      var rows = raw ? JSON.parse(raw) : [];
      return Array.isArray(rows) ? rows : [];
    }} catch (_e) {{
      return [];
    }}
  }}

  function setHistory(rows) {{
    try {{
      localStorage.setItem(STORAGE_KEY, JSON.stringify(rows || []));
    }} catch (_e) {{}}
  }}

  var panelOpen = false;
  var theme = {{
    navyDeep: "hsl(226,55%,10%)",
    navy: "hsl(226,45%,20%)",
    navyLight: "hsl(226,35%,30%)",
    mint: "hsl(160,62%,55%)",
    mintLight: "hsl(160,55%,92%)",
    textMain: "hsl(222,47%,11%)",
    textMuted: "hsl(220,9%,46%)",
    card: "#ffffff",
    border: "hsl(220,13%,91%)",
    bgSoft: "hsl(220,20%,97%)"
  }};
  if (cfg.theme && typeof cfg.theme === "object") {{
    theme = Object.assign(theme, {{
      navyDeep: cfg.theme.navyDeep || cfg.theme.navy_deep || theme.navyDeep,
      navy: cfg.theme.navy || theme.navy,
      navyLight: cfg.theme.navyLight || cfg.theme.navy_light || theme.navyLight,
      mint: cfg.theme.mint || theme.mint,
      mintLight: cfg.theme.mintLight || cfg.theme.mint_light || theme.mintLight
    }});
  }}

  var root = document.createElement("div");
  root.style.position = "fixed";
  root.style.right = "24px";
  root.style.bottom = "24px";
  root.style.zIndex = "2147483000";
  root.style.fontFamily = "Inter, Arial, sans-serif";

  var toggleBtn = document.createElement("button");
  toggleBtn.innerHTML = '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true" focusable="false" style="display:block"><path fill="currentColor" d="M12 3C6.48 3 2 6.94 2 11.8c0 2.57 1.3 4.88 3.37 6.5V22l3.46-1.9c1 .27 2.07.4 3.17.4 5.52 0 10-3.94 10-8.8S17.52 3 12 3Zm-4 9.3a1.3 1.3 0 1 1 0-2.6 1.3 1.3 0 0 1 0 2.6Zm4 0a1.3 1.3 0 1 1 0-2.6 1.3 1.3 0 0 1 0 2.6Zm4 0a1.3 1.3 0 1 1 0-2.6 1.3 1.3 0 0 1 0 2.6Z"/></svg>';
  toggleBtn.style.width = "56px";
  toggleBtn.style.height = "56px";
  toggleBtn.style.borderRadius = "50%";
  toggleBtn.style.border = "none";
  toggleBtn.style.padding = "0";
  toggleBtn.style.display = "flex";
  toggleBtn.style.alignItems = "center";
  toggleBtn.style.justifyContent = "center";
  toggleBtn.style.lineHeight = "0";
  toggleBtn.style.cursor = "pointer";
  toggleBtn.style.background = "linear-gradient(135deg, " + theme.navyDeep + ", " + theme.navy + ")";
  toggleBtn.style.color = "#fff";
  toggleBtn.style.boxShadow = "0 10px 24px hsl(226 45% 20% / 0.35), 0 3px 12px hsl(0 0% 0% / 0.12)";

  var panel = document.createElement("div");
  panel.style.display = "none";
  panel.style.width = "360px";
  panel.style.maxWidth = "calc(100vw - 24px)";
  panel.style.height = "540px";
  panel.style.maxHeight = "calc(100vh - 100px)";
  panel.style.borderRadius = "14px";
  panel.style.overflow = "hidden";
  panel.style.background = theme.card;
  panel.style.boxShadow = "0 16px 40px hsl(226 45% 20% / 0.22), 0 8px 20px hsl(226 45% 20% / 0.1)";
  panel.style.border = "1px solid " + theme.border;
  panel.style.marginTop = "10px";

  var header = document.createElement("div");
  header.style.background = "linear-gradient(90deg, " + theme.navyDeep + ", " + theme.navy + ", hsl(166,54%,20%))";
  header.style.color = "#fff";
  header.style.padding = "10px 12px";
  header.style.display = "flex";
  header.style.alignItems = "center";
  header.style.justifyContent = "space-between";

  var title = document.createElement("div");
  title.textContent = cfg.title || "ACME Assistant";
  title.style.fontWeight = "600";
  title.style.fontSize = "14px";

  var clearBtn = document.createElement("button");
  clearBtn.textContent = "Clear";
  clearBtn.style.background = "transparent";
  clearBtn.style.color = "hsl(210,20%,90%)";
  clearBtn.style.border = "1px solid hsl(226,35%,25%)";
  clearBtn.style.borderRadius = "8px";
  clearBtn.style.padding = "4px 8px";
  clearBtn.style.cursor = "pointer";
  clearBtn.style.fontSize = "12px";

  var body = document.createElement("div");
  body.style.height = "420px";
  body.style.overflowY = "auto";
  body.style.background = theme.bgSoft;
  body.style.padding = "10px";
  body.style.display = "flex";
  body.style.flexDirection = "column";
  body.style.gap = "8px";
  body.style.scrollBehavior = "smooth";

  var composer = document.createElement("div");
  composer.style.display = "flex";
  composer.style.gap = "8px";
  composer.style.padding = "10px";
  composer.style.borderTop = "1px solid " + theme.border;
  composer.style.background = "#fff";

  var input = document.createElement("input");
  input.type = "text";
  input.placeholder = "Ask a question...";
  input.style.flex = "1";
  input.style.border = "1px solid " + theme.border;
  input.style.borderRadius = "10px";
  input.style.padding = "10px";
  input.style.fontSize = "14px";
  input.style.color = theme.textMain;
  input.style.background = "#fff";
  input.style.outline = "none";

  var sendBtn = document.createElement("button");
  sendBtn.textContent = "Send";
  sendBtn.style.border = "none";
  sendBtn.style.borderRadius = "10px";
  sendBtn.style.padding = "10px 14px";
  sendBtn.style.background = "linear-gradient(135deg, " + theme.mint + ", hsl(160,70%,45%))";
  sendBtn.style.color = "#fff";
  sendBtn.style.cursor = "pointer";
  sendBtn.style.fontWeight = "600";

  var introScreen = document.createElement("div");
  introScreen.style.display = "flex";
  introScreen.style.flexDirection = "column";
  introScreen.style.height = "100%";
  introScreen.style.background = "linear-gradient(180deg, hsl(220 12% 22%), hsl(220 8% 18%))";

  var introTop = document.createElement("div");
  introTop.style.flex = "1";
  introTop.style.display = "flex";
  introTop.style.flexDirection = "column";
  introTop.style.justifyContent = "center";
  introTop.style.padding = "28px 24px 20px";
  introTop.style.color = "#fff";

  var introTitle = document.createElement("div");
  introTitle.textContent = "Hi there 👋";
  introTitle.style.fontSize = "36px";
  introTitle.style.lineHeight = "1.05";
  introTitle.style.fontWeight = "700";

  var introSubtitle = document.createElement("div");
  introSubtitle.textContent = "How can I help you?";
  introSubtitle.style.fontSize = "36px";
  introSubtitle.style.lineHeight = "1.05";
  introSubtitle.style.fontWeight = "700";
  introSubtitle.style.marginTop = "2px";

  var introCtaWrap = document.createElement("div");
  introCtaWrap.style.padding = "0 16px 16px";

  var introCta = document.createElement("button");
  introCta.type = "button";
  introCta.style.width = "100%";
  introCta.style.background = "#ffffff";
  introCta.style.border = "1px solid hsl(220 13% 91%)";
  introCta.style.borderRadius = "16px";
  introCta.style.padding = "18px 18px";
  introCta.style.display = "flex";
  introCta.style.alignItems = "center";
  introCta.style.justifyContent = "space-between";
  introCta.style.cursor = "pointer";
  introCta.style.boxShadow = "0 10px 24px hsl(226 45% 20% / 0.2)";
  introCta.style.textAlign = "left";

  var introCtaText = document.createElement("div");

  var introCtaHeading = document.createElement("div");
  introCtaHeading.textContent = "Send us a message";
  introCtaHeading.style.fontSize = "18px";
  introCtaHeading.style.fontWeight = "700";
  introCtaHeading.style.color = "hsl(222,47%,11%)";

  var introCtaHint = document.createElement("div");
  introCtaHint.textContent = "I'm here for your assisstance";
  introCtaHint.style.marginTop = "4px";
  introCtaHint.style.fontSize = "14px";
  introCtaHint.style.color = "hsl(220,9%,46%)";

  var introCtaArrow = document.createElement("div");
  introCtaArrow.innerHTML = "&#10148;";
  introCtaArrow.style.fontSize = "28px";
  introCtaArrow.style.lineHeight = "1";
  introCtaArrow.style.color = "hsl(220,10%,22%)";

  var chatShell = document.createElement("div");
  chatShell.style.display = "none";
  chatShell.style.height = "100%";
  chatShell.style.flexDirection = "column";

  var introEmojis = ["🤖"];
  function setIntroGreeting() {{
    var emoji = introEmojis[Math.floor(Math.random() * introEmojis.length)];
    introTitle.textContent = "Hi there " + emoji;
    introSubtitle.textContent = "How can I help you?";
  }}

  var headerLeft = document.createElement("div");
  headerLeft.style.display = "flex";
  headerLeft.style.alignItems = "center";
  headerLeft.style.gap = "8px";

  var backBtn = document.createElement("button");
  backBtn.type = "button";
  backBtn.setAttribute("aria-label", "Back");
  backBtn.innerHTML = "&#8592;";
  backBtn.style.width = "30px";
  backBtn.style.height = "30px";
  backBtn.style.borderRadius = "8px";
  backBtn.style.border = "1px solid hsl(226,35%,25%)";
  backBtn.style.background = "hsl(226 35% 25% / 0.45)";
  backBtn.style.color = "#fff";
  backBtn.style.cursor = "pointer";
  backBtn.style.fontSize = "16px";
  backBtn.style.lineHeight = "1";

  function applyThemeStyles() {{
    toggleBtn.style.background = "linear-gradient(135deg, " + theme.navyDeep + ", " + theme.navy + ")";
    header.style.background = "linear-gradient(90deg, " + theme.navyDeep + ", " + theme.navy + ", " + theme.navyLight + ")";
    panel.style.border = "1px solid " + theme.border;
    panel.style.background = theme.card;
    body.style.background = theme.bgSoft;
    input.style.border = "1px solid " + theme.border;
    input.style.color = theme.textMain;
    sendBtn.style.background = "linear-gradient(135deg, " + theme.navy + ", " + theme.navyLight + ")";
  }}

  function applyResponsiveLayout() {{
    var isMobile = window.innerWidth <= 640;
    root.style.right = isMobile ? "8px" : "24px";
    root.style.bottom = isMobile ? "8px" : "24px";
    panel.style.width = isMobile ? "calc(100vw - 16px)" : "380px";
    panel.style.height = isMobile ? "min(76vh, 560px)" : "560px";
    panel.style.maxWidth = isMobile ? "calc(100vw - 16px)" : "calc(100vw - 24px)";
    panel.style.maxHeight = isMobile ? "calc(100vh - 90px)" : "calc(100vh - 100px)";
    body.style.height = isMobile ? "calc(100% - 122px)" : "438px";
    introTitle.style.fontSize = isMobile ? "30px" : "36px";
    introSubtitle.style.fontSize = isMobile ? "30px" : "36px";
  }}

  function openIntro() {{
    setIntroGreeting();
    introScreen.style.display = "flex";
    chatShell.style.display = "none";
  }}

  function openChat() {{
    introScreen.style.display = "none";
    chatShell.style.display = "flex";
    renderHistory();
    setTimeout(function () {{ input.focus(); }}, 60);
  }}

  function bubble(text, role) {{
    var row = document.createElement("div");
    row.style.display = "flex";
    row.style.flexDirection = "column";
    row.style.justifyContent = role === "user" ? "flex-end" : "flex-start";
    row.style.alignItems = role === "user" ? "flex-end" : "flex-start";
    var b = document.createElement("div");
    b.textContent = text;
    b.style.maxWidth = "80%";
    b.style.padding = "10px 12px";
    b.style.borderRadius = "14px";
    b.style.fontSize = "13px";
    b.style.lineHeight = "1.45";
    b.style.boxShadow = "0 1px 3px hsl(226 45% 20% / 0.06)";
    b.style.whiteSpace = "pre-wrap";
    if (role === "user") {{
      b.style.background = "linear-gradient(135deg, " + theme.navy + ", " + theme.navyLight + ")";
      b.style.color = "#ffffff";
      b.style.borderTopRightRadius = "8px";
    }} else {{
      b.style.background = "hsl(220 14% 94%)";
      b.style.color = theme.textMain;
      b.style.border = "1px solid " + theme.border;
      b.style.borderTopLeftRadius = "8px";
    }}
    row.appendChild(b);
    if (role === "user") {{
      var copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false" style="display:block"><path fill="currentColor" d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1Zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm0 16H10V7h9v14Z"/></svg>';
      copyBtn.style.marginTop = "4px";
      copyBtn.style.width = "22px";
      copyBtn.style.height = "22px";
      copyBtn.style.padding = "0";
      copyBtn.style.border = "1px solid " + theme.border;
      copyBtn.style.borderRadius = "6px";
      copyBtn.style.background = "transparent";
      copyBtn.style.color = "hsl(220,9%,46%)";
      copyBtn.style.display = "inline-flex";
      copyBtn.style.alignItems = "center";
      copyBtn.style.justifyContent = "center";
      copyBtn.style.cursor = "pointer";
      copyBtn.title = "Copy message";
      copyBtn.addEventListener("click", function () {{
        var content = String(text || "");
        if (!content) return;
        if (navigator.clipboard && navigator.clipboard.writeText) {{
          navigator.clipboard.writeText(content).then(function () {{
            copyBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false" style="display:block"><path fill="currentColor" d="M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z"/></svg>';
            copyBtn.style.color = "hsl(142,72%,29%)";
            setTimeout(function () {{
              copyBtn.innerHTML = '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" focusable="false" style="display:block"><path fill="currentColor" d="M16 1H6a2 2 0 0 0-2 2v12h2V3h10V1Zm3 4H10a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h9a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Zm0 16H10V7h9v14Z"/></svg>';
              copyBtn.style.color = "hsl(220,9%,46%)";
            }}, 1200);
          }}).catch(function () {{}});
        }}
      }});
      row.appendChild(copyBtn);
    }}
    body.appendChild(row);
    body.scrollTop = body.scrollHeight;
  }}

  function renderHistory() {{
    body.innerHTML = "";
    var rows = getHistory();
    if (!rows.length) {{
      bubble(dynamicWelcome, "assistant");
      return;
    }}
    rows.forEach(function (m) {{ bubble(m.content || "", m.role || "assistant"); }});
  }}

  function addHistory(role, content) {{
    var rows = getHistory();
    rows.push({{ role: role, content: content, ts: Date.now() }});
    setHistory(rows);
  }}

  function removeTypingIndicator() {{
    var existing = document.getElementById("acme-widget-typing");
    if (existing && existing.parentNode) {{
      existing.parentNode.removeChild(existing);
    }}
  }}

  function showTypingIndicator() {{
    removeTypingIndicator();
    var row = document.createElement("div");
    row.id = "acme-widget-typing";
    row.style.display = "flex";
    row.style.justifyContent = "flex-start";
    row.style.alignItems = "center";

    var bubbleWrap = document.createElement("div");
    bubbleWrap.style.display = "flex";
    bubbleWrap.style.alignItems = "center";
    bubbleWrap.style.gap = "8px";
    bubbleWrap.style.padding = "10px 12px";
    bubbleWrap.style.borderRadius = "14px";
    bubbleWrap.style.borderTopLeftRadius = "8px";
    bubbleWrap.style.background = "hsl(220 14% 94%)";
    bubbleWrap.style.border = "1px solid " + theme.border;
    bubbleWrap.style.boxShadow = "0 1px 3px hsl(226 45% 20% / 0.06)";

    var dots = document.createElement("span");
    dots.textContent = "...";
    dots.style.fontSize = "16px";
    dots.style.lineHeight = "1";
    dots.style.color = theme.mint;
    dots.style.letterSpacing = "1px";
    dots.style.animation = "acmeTypingPulse 1s ease-in-out infinite";

    bubbleWrap.appendChild(dots);
    row.appendChild(bubbleWrap);
    body.appendChild(row);
    body.scrollTop = body.scrollHeight;
  }}

  function setLoading(flag) {{
    sendBtn.disabled = !!flag;
    sendBtn.textContent = flag ? "..." : "Send";
  }}

  async function loadDynamicConfig() {{
    try {{
      var res = await fetch(CONFIG_URL + "?widget_key=" + encodeURIComponent(botKey));
      if (!res.ok) return;
      var data = await res.json();
      if (data && data.bot_name && !cfg.title) {{
        title.textContent = String(data.bot_name);
      }}
      if (data && data.welcome_message) {{
        dynamicWelcome = String(data.welcome_message);
        localStorage.setItem(WELCOME_KEY, dynamicWelcome);
      }}
      if (data && data.theme && typeof data.theme === "object") {{
        theme = Object.assign(theme, {{
          navyDeep: data.theme.navy_deep || data.theme.navyDeep || theme.navyDeep,
          navy: data.theme.navy || theme.navy,
          navyLight: data.theme.navy_light || data.theme.navyLight || theme.navyLight,
          mint: data.theme.mint || theme.mint,
          mintLight: data.theme.mint_light || data.theme.mintLight || theme.mintLight
        }});
        applyThemeStyles();
      }}
    }} catch (_e) {{
      // keep defaults
    }}
  }}

  async function ask() {{
    var q = (input.value || "").trim();
    if (!q) return;
    input.value = "";
    addHistory("user", q);
    bubble(q, "user");
    setLoading(true);
    showTypingIndicator();
    try {{
      var res = await fetch(CHAT_URL, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify({{
          widget_key: botKey,
          query: q,
          session_id: getSessionId()
        }})
      }});
      var data = await res.json();
      var answer = (data && data.answer) ? String(data.answer) : "I could not generate a response.";
      removeTypingIndicator();
      addHistory("assistant", answer);
      bubble(answer, "assistant");
    }} catch (_e) {{
      var msg = "Request failed. Please try again.";
      removeTypingIndicator();
      addHistory("assistant", msg);
      bubble(msg, "assistant");
    }} finally {{
      removeTypingIndicator();
      setLoading(false);
    }}
  }}

  clearBtn.addEventListener("click", function () {{
    setHistory([]);
    localStorage.removeItem(SESSION_KEY);
    renderHistory();
  }});
  sendBtn.addEventListener("click", ask);
  input.addEventListener("keydown", function (e) {{
    if (e.key === "Enter") ask();
  }});
  input.addEventListener("focus", function () {{
    input.style.boxShadow = "0 0 0 2px hsl(160 62% 55% / 0.25)";
    input.style.borderColor = "hsl(160 62% 55%)";
  }});
  input.addEventListener("blur", function () {{
    input.style.boxShadow = "none";
    input.style.borderColor = theme.border;
  }});
  introCta.addEventListener("click", function () {{
    openChat();
  }});
  backBtn.addEventListener("click", function () {{
    openIntro();
  }});
  toggleBtn.addEventListener("click", function () {{
    panelOpen = !panelOpen;
    panel.style.display = panelOpen ? "block" : "none";
    if (panelOpen) {{
      openIntro();
    }}
  }});
  window.addEventListener("resize", applyResponsiveLayout);

  introTop.appendChild(introTitle);
  introTop.appendChild(introSubtitle);
  introCtaText.appendChild(introCtaHeading);
  introCtaText.appendChild(introCtaHint);
  introCta.appendChild(introCtaText);
  introCta.appendChild(introCtaArrow);
  introCtaWrap.appendChild(introCta);
  introScreen.appendChild(introTop);
  introScreen.appendChild(introCtaWrap);

  headerLeft.appendChild(backBtn);
  headerLeft.appendChild(title);
  header.appendChild(headerLeft);
  header.appendChild(clearBtn);
  composer.appendChild(input);
  composer.appendChild(sendBtn);
  chatShell.appendChild(header);
  chatShell.appendChild(body);
  chatShell.appendChild(composer);
  panel.appendChild(introScreen);
  panel.appendChild(chatShell);
  root.appendChild(toggleBtn);
  root.appendChild(panel);
  document.body.appendChild(root);
  if (!document.getElementById("acme-widget-typing-style")) {{
    var typingStyle = document.createElement("style");
    typingStyle.id = "acme-widget-typing-style";
    typingStyle.textContent = "@keyframes acmeTypingPulse {{0%{{opacity:.35}}50%{{opacity:1}}100%{{opacity:.35}}}}";
    document.head.appendChild(typingStyle);
  }}
  applyThemeStyles();
  applyResponsiveLayout();
  setIntroGreeting();
  loadDynamicConfig();
}})();
"""
    return HttpResponse(js, content_type="application/javascript; charset=utf-8")


def _send_verification_email(email: str, code: str) -> None:
    subject = "ACME ONE Admin Email Verification Code"
    message = (
        "Welcome to ACME ONE.\n\n"
        "Use the verification code below to confirm your admin email address:\n\n"
        f"{code}\n\n"
        "This code expires in 30 minutes.\n"
        "If you did not request this, you can ignore this email."
    )
    sender_address = getattr(settings, "EMAIL_HOST_USER", "").strip() or "no-reply@example.com"
    sender_name = getattr(settings, "EMAIL_SENDER_NAME", "").strip() or "CHATBOT ADMIN Panel"
    from_email = f"{sender_name} <{sender_address}>"
    try:
        send_mail(subject, message, from_email, [email], fail_silently=True)
    except Exception:
        # In local/dev environments without email configured we ignore failures.
        pass


def _send_password_reset_email(email: str, code: str) -> None:
    subject = "ACME ONE Admin Password Reset Code"
    message = (
        "We received a request to reset your ACME ONE admin password.\n\n"
        "Use the reset code below to set a new password:\n\n"
        f"{code}\n\n"
        "This code expires in 30 minutes.\n"
        "If you did not request a password reset, you can ignore this email."
    )
    sender_address = getattr(settings, "EMAIL_HOST_USER", "").strip() or "no-reply@example.com"
    sender_name = getattr(settings, "EMAIL_SENDER_NAME", "").strip() or "CHATBOT ADMIN Panel"
    from_email = f"{sender_name} <{sender_address}>"
    try:
        send_mail(subject, message, from_email, [email], fail_silently=True)
    except Exception:
        # In local/dev environments without email configured we ignore failures.
        pass


def _fake_provider_intent_id() -> str:
    return f"pi_{secrets.token_hex(12)}"


@extend_schema(
    summary="Admin signup",
    request=inline_serializer(
        name="AdminSignupRequest",
        fields={
            "email": serializers.EmailField(),
            "password": serializers.CharField(),
            "full_name": serializers.CharField(),
        },
    ),
    responses={
        201: OpenApiResponse(
            response=inline_serializer(
                name="AdminSignupResponse",
                fields={
                    "message": serializers.CharField(),
                    "admin": inline_serializer(
                        name="AdminSignupAdmin",
                        fields={
                            "id": serializers.IntegerField(),
                            "email": serializers.EmailField(),
                            "full_name": serializers.CharField(),
                            "company_id": serializers.IntegerField(allow_null=True),
                            "is_verified": serializers.BooleanField(),
                        },
                    ),
                },
            )
        ),
        400: OpenApiResponse(description="Validation error"),
        409: OpenApiResponse(description="Admin already registered"),
    },
    examples=[
        OpenApiExample(
            "Signup example",
            value={"email": "admin@example.com", "password": "StrongPass123", "full_name": "Admin User"},
        )
    ],
)
@api_view(["POST"])
@csrf_exempt
def admin_signup(request):
    email = str(request.data.get("email") or "").strip().lower()
    password = str(request.data.get("password") or "")
    full_name = str(request.data.get("full_name") or "").strip()
    if not email or not password or not full_name:
        return Response(
            {"error": "email, password, and full_name are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(password) < 8:
        return Response(
            {"error": "Password must be at least 8 characters."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    existing = repo.find_admin_by_email(email)
    if existing:
        return Response({"error": "Admin already registered."}, status=status.HTTP_409_CONFLICT)
    verification_code = f"{uuid.uuid4().hex[:6]}".upper()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    admin = repo.create_admin(
        {
            "email": email,
            "password_hash": make_password(password),
            "full_name": full_name,
            "company_id": None,
            "role": "admin",
            "is_verified": False,
            "verification_code": verification_code,
            "verification_expires_at": expires_at,
        }
    )
    _send_verification_email(email, verification_code)
    return Response(
        {
            "message": "Signup successful.",
            "admin": {
                "id": admin.id,
                "email": admin.email,
                "full_name": admin.full_name,
                "company_id": admin.company_id,
                "is_verified": admin.is_verified,
            },
        },
        status=status.HTTP_201_CREATED,
    )


@extend_schema(
    summary="Admin login",
    request=inline_serializer(
        name="AdminLoginRequest",
        fields={
            "email": serializers.EmailField(),
            "password": serializers.CharField(),
        },
    ),
    responses={
        200: OpenApiResponse(
            response=inline_serializer(
                name="AdminLoginResponse",
                fields={
                    "message": serializers.CharField(),
                    "access_token": serializers.CharField(),
                    "refresh_token": serializers.CharField(),
                    "admin": inline_serializer(
                        name="AdminLoginAdmin",
                        fields={
                            "id": serializers.IntegerField(),
                            "email": serializers.EmailField(),
                            "full_name": serializers.CharField(),
                            "company_id": serializers.IntegerField(allow_null=True),
                            "role": serializers.CharField(),
                        },
                    ),
                },
            )
        ),
        400: OpenApiResponse(description="Missing credentials"),
        401: OpenApiResponse(description="Invalid email or password"),
        403: OpenApiResponse(description="Email not verified"),
    },
)
@api_view(["POST"])
@csrf_exempt
def admin_login(request):
    email = str(request.data.get("email") or "").strip().lower()
    password = str(request.data.get("password") or "")
    if not email or not password:
        return Response({"error": "email and password are required."}, status=status.HTTP_400_BAD_REQUEST)
    admin = repo.find_admin_by_email(email)
    if not admin or not check_password(password, admin.password_hash):
        return Response({"error": "Invalid email or password."}, status=status.HTTP_401_UNAUTHORIZED)
    if not admin.is_verified:
        return Response({"error": "Email not verified."}, status=status.HTTP_403_FORBIDDEN)
    access_token = generate_access_token(admin)
    refresh_jti = uuid.uuid4().hex
    refresh_token = generate_refresh_token(admin, jti=refresh_jti)
    refresh_expires_at = (
        datetime.now(timezone.utc) + timedelta(days=int(getattr(settings, "JWT_REFRESH_TOKEN_LIFETIME_DAYS", 7)))
    ).isoformat()
    repo.store_refresh_token(jti=refresh_jti, admin_id=admin.id, expires_at=refresh_expires_at)
    return Response(
        {
            "message": "Login successful.",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "admin": {
                "id": admin.id,
                "email": admin.email,
                "full_name": admin.full_name,
                "company_id": admin.company_id,
                "role": admin.role,
            },
        },
        status=status.HTTP_200_OK,
    )


@extend_schema(
    summary="Verify admin email",
    request=inline_serializer(
        name="VerifyEmailRequest",
        fields={
            "email": serializers.EmailField(),
            "code": serializers.CharField(),
        },
    ),
    responses={
        200: OpenApiResponse(description="Email verified"),
        400: OpenApiResponse(description="Invalid or expired code"),
        404: OpenApiResponse(description="Admin not found"),
    },
)
@api_view(["POST"])
@csrf_exempt
def verify_email(request):
    email = str(request.data.get("email") or "").strip().lower()
    code = str(request.data.get("code") or "").strip().upper()
    if not email or not code:
        return Response({"error": "email and code are required."}, status=status.HTTP_400_BAD_REQUEST)
    admin = repo.find_admin_by_email(email)
    if not admin:
        return Response({"error": "Admin not found."}, status=status.HTTP_404_NOT_FOUND)
    if admin.is_verified:
        return Response({"message": "Email already verified."}, status=status.HTTP_200_OK)
    if not admin.verification_code or admin.verification_code.upper() != code:
        return Response({"error": "Invalid verification code."}, status=status.HTTP_400_BAD_REQUEST)
    if admin.verification_expires_at:
        try:
            expires_dt = datetime.fromisoformat(admin.verification_expires_at)
            if expires_dt <= datetime.now(timezone.utc):
                return Response({"error": "Verification code expired."}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            pass
    repo.update_admin(
        admin.id,
        {
            "is_verified": True,
            "verification_code": None,
            "verification_expires_at": None,
        },
    )
    return Response({"message": "Email verified successfully."}, status=status.HTTP_200_OK)


@extend_schema(
    summary="Resend email verification code",
    request=inline_serializer(
        name="ResendVerificationCodeRequest",
        fields={"email": serializers.EmailField()},
    ),
    responses={
        200: OpenApiResponse(description="Verification code sent"),
        400: OpenApiResponse(description="Email is required"),
        404: OpenApiResponse(description="Admin not found"),
    },
)
@api_view(["POST"])
@csrf_exempt
def resend_verification_code(request):
    email = str(request.data.get("email") or "").strip().lower()
    if not email:
        return Response({"error": "email is required."}, status=status.HTTP_400_BAD_REQUEST)
    admin = repo.find_admin_by_email(email)
    if not admin:
        return Response({"error": "Admin not found."}, status=status.HTTP_404_NOT_FOUND)
    if admin.is_verified:
        return Response({"message": "Email is already verified."}, status=status.HTTP_200_OK)

    verification_code = f"{uuid.uuid4().hex[:6]}".upper()
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
    repo.update_admin(
        admin.id,
        {
            "verification_code": verification_code,
            "verification_expires_at": expires_at,
        },
    )
    _send_verification_email(email, verification_code)
    return Response(
        {"message": "Verification code sent successfully."},
        status=status.HTTP_200_OK,
    )


@extend_schema(
    summary="Request password reset code",
    request=inline_serializer(
        name="ForgotPasswordRequest",
        fields={"email": serializers.EmailField()},
    ),
    responses={
        200: OpenApiResponse(description="If account exists, reset code is sent"),
        400: OpenApiResponse(description="Email is required"),
    },
)
@api_view(["POST"])
@csrf_exempt
def forgot_password(request):
    email = str(request.data.get("email") or "").strip().lower()
    if not email:
        return Response({"error": "email is required."}, status=status.HTTP_400_BAD_REQUEST)
    admin = repo.find_admin_by_email(email)
    if admin:
        reset_code = f"{uuid.uuid4().hex[:6]}".upper()
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()
        repo.update_admin(
            admin.id,
            {
                "password_reset_code": reset_code,
                "password_reset_expires_at": expires_at,
            },
        )
        _send_password_reset_email(email, reset_code)
    return Response(
        {"message": "If this email is registered, a password reset code has been sent."},
        status=status.HTTP_200_OK,
    )


@extend_schema(
    summary="Reset admin password",
    request=inline_serializer(
        name="ResetPasswordRequest",
        fields={
            "email": serializers.EmailField(),
            "code": serializers.CharField(),
            "new_password": serializers.CharField(),
        },
    ),
    responses={
        200: OpenApiResponse(description="Password reset successful"),
        400: OpenApiResponse(description="Validation/code error"),
        404: OpenApiResponse(description="Admin not found"),
    },
)
@api_view(["POST"])
@csrf_exempt
def reset_password(request):
    email = str(request.data.get("email") or "").strip().lower()
    code = str(request.data.get("code") or "").strip().upper()
    new_password = str(request.data.get("new_password") or "")

    if not email or not code or not new_password:
        return Response(
            {"error": "email, code, and new_password are required."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    if len(new_password) < 8:
        return Response(
            {"error": "Password must be at least 8 characters."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    admin = repo.find_admin_by_email(email)
    if not admin:
        return Response({"error": "Admin not found."}, status=status.HTTP_404_NOT_FOUND)
    if not admin.password_reset_code or admin.password_reset_code.upper() != code:
        return Response({"error": "Invalid reset code."}, status=status.HTTP_400_BAD_REQUEST)
    if admin.password_reset_expires_at:
        try:
            expires_dt = datetime.fromisoformat(admin.password_reset_expires_at)
            if expires_dt <= datetime.now(timezone.utc):
                return Response({"error": "Reset code expired."}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            pass

    repo.update_admin(
        admin.id,
        {
            "password_hash": make_password(new_password),
            "password_reset_code": None,
            "password_reset_expires_at": None,
        },
    )
    return Response({"message": "Password reset successful."}, status=status.HTTP_200_OK)


@extend_schema(
    summary="Refresh access token",
    request=inline_serializer(
        name="RefreshTokenRequest",
        fields={"refresh_token": serializers.CharField()},
    ),
    responses={
        200: OpenApiResponse(
            response=inline_serializer(
                name="RefreshTokenResponse",
                fields={"access_token": serializers.CharField()},
            )
        ),
        401: OpenApiResponse(description="Invalid or expired refresh token"),
    },
)
@api_view(["POST"])
@csrf_exempt
def refresh_token(request):
    token = str(request.data.get("refresh_token") or "").strip()
    if not token:
        return Response({"error": "refresh_token is required."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return Response({"error": "Refresh token expired."}, status=status.HTTP_401_UNAUTHORIZED)
    except jwt.InvalidTokenError:
        return Response({"error": "Invalid refresh token."}, status=status.HTTP_401_UNAUTHORIZED)
    if payload.get("type") != "refresh":
        return Response({"error": "Invalid token type."}, status=status.HTTP_401_UNAUTHORIZED)
    jti = str(payload.get("jti") or "")
    stored = repo.get_refresh_token(jti)
    if not stored:
        return Response({"error": "Refresh token revoked."}, status=status.HTTP_401_UNAUTHORIZED)
    admin_id = int(payload.get("sub"))
    admin = repo.get_admin(admin_id)
    if not admin or not admin.is_verified:
        return Response({"error": "Admin not found or not verified."}, status=status.HTTP_401_UNAUTHORIZED)
    access = generate_access_token(admin)
    return Response({"access_token": access}, status=status.HTTP_200_OK)


@extend_schema(
    summary="Logout (revoke refresh token)",
    request=inline_serializer(
        name="LogoutRequest",
        fields={"refresh_token": serializers.CharField()},
    ),
    responses={
        200: OpenApiResponse(
            response=inline_serializer(
                name="LogoutResponse",
                fields={"message": serializers.CharField()},
            )
        ),
    },
)
@api_view(["POST"])
@csrf_exempt
def logout(request):
    token = str(request.data.get("refresh_token") or "").strip()
    if not token:
        return Response({"error": "refresh_token is required."}, status=status.HTTP_400_BAD_REQUEST)
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except jwt.InvalidTokenError:
        # If token is invalid we treat it as already logged out.
        return Response({"message": "Logged out."}, status=status.HTTP_200_OK)
    jti = str(payload.get("jti") or "")
    if jti:
        repo.revoke_refresh_token(jti)
    return Response({"message": "Logged out."}, status=status.HTTP_200_OK)


@extend_schema(
    summary="Super-admin only example endpoint",
    responses={200: OpenApiResponse(description="Super admin access OK")},
)
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsSuperAdmin])
@csrf_exempt
def super_admin_only_example(request):
    return Response({"message": "Super admin access OK."}, status=status.HTTP_200_OK)


def _super_admin_company_row(company) -> Dict[str, Any]:
    admins = repo.list_admins(company_id=int(company.id))
    bots = repo.list_bots(company_id=int(company.id))
    docs = repo.list_documents(company_id=int(company.id))
    bot_ids = [int(b.id) for b in bots]
    query_count = repo.count_chat_messages_for_bot_ids(bot_ids=bot_ids)
    return {
        "company_id": int(company.id),
        "name": company.name,
        "plan_type": company.plan_type,
        "admin_count": len(admins),
        "bot_count": len(bots),
        "document_count": len(docs),
        "query_count": int(query_count),
        "created_at": company.created_at,
    }


@extend_schema(
    operation_id="super_admin_overview",
    summary="Super-admin dashboard overview",
    description="Returns high-level platform metrics for the super-admin dashboard.",
    responses={
        200: OpenApiResponse(
            response=inline_serializer(
                name="SuperAdminOverviewResponse",
                fields={
                    "totals": inline_serializer(
                        name="SuperAdminTotals",
                        fields={
                            "companies": serializers.IntegerField(),
                            "admins": serializers.IntegerField(),
                            "bots": serializers.IntegerField(),
                            "documents": serializers.IntegerField(),
                            "chat_queries": serializers.IntegerField(),
                            "fallback_queries": serializers.IntegerField(),
                            "fallback_rate": serializers.FloatField(),
                        },
                    )
                },
            )
        )
    },
)
@csrf_exempt
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsSuperAdmin])
def super_admin_dashboard_overview(request):
    companies = repo.list_companies()
    admins = repo.list_admins()
    bots = repo.list_all_bots()
    docs = repo.list_documents()
    total_queries = repo.count_chat_messages()
    total_fallback = repo.count_fallback_chat_messages()
    fallback_rate = (float(total_fallback) / float(total_queries)) if total_queries > 0 else 0.0
    return Response(
        {
            "totals": {
                "companies": len(companies),
                "admins": len(admins),
                "bots": len(bots),
                "documents": len(docs),
                "chat_queries": int(total_queries),
                "fallback_queries": int(total_fallback),
                "fallback_rate": round(fallback_rate, 4),
            }
        }
    )


@extend_schema(
    operation_id="super_admin_companies_list",
    summary="List companies for super-admin",
    description="Returns all companies with lightweight analytics for the super-admin table.",
    responses={
        200: OpenApiResponse(
            response=inline_serializer(
                name="SuperAdminCompaniesResponse",
                fields={
                    "companies": serializers.ListField(
                        child=inline_serializer(
                            name="SuperAdminCompanyRow",
                            fields={
                                "company_id": serializers.IntegerField(),
                                "name": serializers.CharField(),
                                "plan_type": serializers.CharField(),
                                "admin_count": serializers.IntegerField(),
                                "bot_count": serializers.IntegerField(),
                                "document_count": serializers.IntegerField(),
                                "query_count": serializers.IntegerField(),
                                "created_at": serializers.CharField(),
                            },
                        )
                    )
                },
            )
        )
    },
)
@csrf_exempt
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsSuperAdmin])
def super_admin_companies(request):
    rows = [_super_admin_company_row(c) for c in repo.list_companies()]
    return Response({"companies": rows})


@extend_schema(
    operation_id="super_admin_company_detail",
    summary="Get company drill-down for super-admin",
    description="Returns one company details, summary counts, and related admins/bots/documents.",
    responses={
        200: OpenApiResponse(description="Company detail with aggregates"),
        404: OpenApiResponse(description="Company not found"),
    },
)
@csrf_exempt
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsSuperAdmin])
def super_admin_company_detail(request, company_id: int):
    company = repo.get_company(int(company_id))
    if not company:
        return Response({"error": "Company not found."}, status=status.HTTP_404_NOT_FOUND)
    admins = repo.list_admins(company_id=int(company.id))
    bots = repo.list_bots(company_id=int(company.id))
    docs = repo.list_documents(company_id=int(company.id))
    bot_ids = [int(b.id) for b in bots]
    query_count = repo.count_chat_messages_for_bot_ids(bot_ids=bot_ids)
    fallback_count = repo.count_fallback_chat_messages_for_bot_ids(bot_ids=bot_ids)
    fallback_rate = (float(fallback_count) / float(query_count)) if query_count > 0 else 0.0
    return Response(
        {
            "company": {
                "id": int(company.id),
                "name": company.name,
                "plan_type": company.plan_type,
                "admin_id": int(company.admin_id),
                "domain": company.domain,
                "created_at": company.created_at,
            },
            "summary": {
                "admin_count": len(admins),
                "bot_count": len(bots),
                "document_count": len(docs),
                "chat_queries": int(query_count),
                "fallback_rate": round(fallback_rate, 4),
            },
            "admins": [
                {
                    "id": int(a.id),
                    "email": a.email,
                    "full_name": a.full_name,
                    "role": a.role,
                    "is_verified": bool(a.is_verified),
                    "created_at": a.created_at,
                }
                for a in admins
            ],
            "bots": [
                {
                    "id": int(b.id),
                    "name": b.name,
                    "plan_type": b.plan_type,
                    "created_at": b.created_at,
                }
                for b in bots
            ],
            "documents": [
                {
                    "id": int(d.id),
                    "bot_id": d.bot_id,
                    "name": d.name,
                    "status": d.status,
                    "chunk_count": d.chunk_count,
                    "token_count": d.token_count,
                    "embedding_count": d.embedding_count,
                    "created_at": d.created_at,
                }
                for d in docs
            ],
        }
    )


@extend_schema(
    operation_id="super_admin_company_update_plan",
    summary="Update company plan (super-admin)",
    request=inline_serializer(
        name="SuperAdminUpdateCompanyPlanRequest",
        fields={"plan_type": serializers.ChoiceField(choices=["free", "paid"])},
    ),
    responses={
        200: OpenApiResponse(description="Company plan updated"),
        400: OpenApiResponse(description="Invalid plan_type"),
        404: OpenApiResponse(description="Company not found"),
    },
)
@csrf_exempt
@api_view(["PATCH"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsSuperAdmin])
def super_admin_update_company_plan(request, company_id: int):
    company = repo.get_company(int(company_id))
    if not company:
        return Response({"error": "Company not found."}, status=status.HTTP_404_NOT_FOUND)
    plan_type = str(request.data.get("plan_type") or "").strip().lower()
    if plan_type not in {"free", "paid"}:
        return Response({"error": "plan_type must be either 'free' or 'paid'."}, status=400)
    repo.update_company(int(company_id), {"plan_type": plan_type})
    fresh = repo.get_company(int(company_id))
    return Response(
        {
            "message": "Company plan updated.",
            "company": {
                "id": int(fresh.id),
                "name": fresh.name,
                "plan_type": fresh.plan_type,
            },
        }
    )


@extend_schema(
    methods=["GET"],
    operation_id="admin_get_company",
    summary="Get current admin company",
    description="Returns the company linked to the authenticated admin.",
    responses={
        200: OpenApiResponse(
            response=inline_serializer(
                name="AdminCompanyResponse",
                fields={
                    "company": inline_serializer(
                        name="AdminCompany",
                        fields={
                            "id": serializers.IntegerField(),
                            "name": serializers.CharField(),
                            "admin_id": serializers.IntegerField(),
                        },
                    )
                },
            )
        ),
        400: OpenApiResponse(description="Admin has no company"),
        401: OpenApiResponse(description="Unauthorized"),
        404: OpenApiResponse(description="Company not found"),
    },
)
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
@csrf_exempt
def get_admin_company(request):
    admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err
    company = repo.get_company(int(company_id))
    if not company:
        return Response({"error": "Company not found."}, status=status.HTTP_404_NOT_FOUND)
    return Response(
        {
            "company": {
                "id": int(company.id),
                "name": company.name,
                "admin_id": int(company.admin_id),
            }
        }
    )


@extend_schema(
    summary="Create company for current admin",
    description="Each admin can own exactly one company. Requires Bearer access token.",
    request=inline_serializer(
        name="AddCompanyRequest",
        fields={
            "name": serializers.CharField(),
            "plan_type": serializers.ChoiceField(choices=["free", "paid"], required=False),
        },
    ),
    responses={
        201: OpenApiResponse(
            response=inline_serializer(
                name="AddCompanyResponse",
                fields={
                    "message": serializers.CharField(),
                    "company": inline_serializer(
                        name="Company",
                        fields={
                            "id": serializers.IntegerField(),
                            "name": serializers.CharField(),
                            "admin_id": serializers.IntegerField(),
                        },
                    ),
                },
            )
        ),
        400: OpenApiResponse(description="Validation error"),
        401: OpenApiResponse(description="Unauthorized"),
        409: OpenApiResponse(description="Admin already has a company or domain exists"),
    },
)
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
@csrf_exempt
def add_company(request):
    admin_id = int(getattr(request.user, "id"))
    admin = repo.get_admin(admin_id)
    if admin.company_id is not None:
        return Response({"error": "Admin already has a company."}, status=status.HTTP_409_CONFLICT)
    name = str(request.data.get("name") or "").strip()
    plan_type = str(request.data.get("plan_type") or "free").strip().lower() or "free"
    if plan_type not in {"free", "paid"}:
        return Response({"error": "plan_type must be either 'free' or 'paid'."}, status=status.HTTP_400_BAD_REQUEST)
    if not name:
        return Response({"error": "name is required."}, status=status.HTTP_400_BAD_REQUEST)
    if repo.find_company_by_name(name):
        return Response({"error": "Company name already exists."}, status=status.HTTP_409_CONFLICT)
    company = repo.create_company({"name": name, "admin_id": admin.id, "plan_type": plan_type})
    repo.update_admin(admin.id, {"company_id": company.id})
    return Response(
        {
            "message": "Company created.",
            "company": {
                "id": company.id,
                "name": company.name,
                "admin_id": company.admin_id,
            },
        },
        status=status.HTTP_201_CREATED,
    )


@extend_schema(
    methods=["POST"],
    operation_id="admin_payment_create_intent",
    summary="Create payment intent for paid bot",
    description="Creates a sandbox payment transaction before paid bot creation.",
    request=inline_serializer(
        name="CreatePaymentIntentRequest",
        fields={
            "bot_name": serializers.CharField(),
            "payment_method": serializers.ChoiceField(
                choices=["card", "jazzcash", "bank", "easypaisa", "payfast", "paypro", "paypal"]
            ),
            "card_type": serializers.ChoiceField(choices=["credit", "debit"], required=False),
        },
    ),
    responses={
        201: OpenApiResponse(description="Payment intent created"),
        400: OpenApiResponse(description="Validation error"),
        401: OpenApiResponse(description="Unauthorized"),
    },
)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def create_payment_intent(request):
    admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err
    bot_name = str(request.data.get("bot_name") or "").strip()
    payment_method = str(request.data.get("payment_method") or "").strip().lower()
    card_type = str(request.data.get("card_type") or "").strip().lower()
    if not bot_name:
        return Response({"error": "bot_name is required."}, status=400)
    if payment_method not in {"card", "jazzcash", "bank", "easypaisa", "payfast", "paypro", "paypal"}:
        return Response(
            {
                "error": (
                    "payment_method must be one of: "
                    "card, jazzcash, bank, easypaisa, payfast, paypro, paypal."
                )
            },
            status=400,
        )
    if payment_method == "card" and card_type not in {"credit", "debit"}:
        return Response({"error": "card_type must be 'credit' or 'debit' for card payments."}, status=400)
    provider_mode = str(getattr(settings, "PAYMENT_PROVIDER", "sandbox")).strip().lower() or "sandbox"
    provider_intent_id = _fake_provider_intent_id()
    provider_status = "pending"
    client_secret = None
    checkout_url = None
    provider = "sandbox"
    if provider_mode != "sandbox":
        try:
            provider_currency = str(getattr(settings, "PAYMENT_CURRENCY", "pkr")).strip().lower() or "pkr"
            charge_amount = float(PAID_BOT_AMOUNT_PKR) if provider_currency == "pkr" else float(PAID_BOT_AMOUNT_USD)
            provider_result = create_provider_payment_intent(
                payment_method=payment_method,
                amount_major=charge_amount,
                currency=provider_currency,
                metadata={
                    "company_id": int(company_id),
                    "admin_id": int(admin.id),
                    "bot_name": bot_name,
                },
            )
        except Exception as exc:
            return Response({"error": f"Payment provider error: {str(exc)}"}, status=502)
        provider_intent_id = provider_result.provider_intent_id
        provider_status = provider_result.status
        client_secret = provider_result.client_secret
        checkout_url = provider_result.checkout_url
        provider = provider_result.provider
    payment = repo.create_payment(
        {
            "company_id": int(company_id),
            "admin_id": int(admin.id),
            "bot_name": bot_name,
            "amount_pkr": PAID_BOT_AMOUNT_PKR,
            "amount_usd": PAID_BOT_AMOUNT_USD,
            "currency": "PKR",
            "payment_method": payment_method,
            "card_type": card_type if payment_method == "card" else None,
            "provider": provider,
            "provider_intent_id": provider_intent_id,
            "status": provider_status,
            "metadata": {"source": "admin_bot_checkout"},
        }
    )
    return Response(
        {
            "message": "Payment intent created.",
            "payment": {
                "id": payment.id,
                "provider_intent_id": payment.provider_intent_id,
                "status": payment.status,
                "bot_name": payment.bot_name,
                "amount_pkr": payment.amount_pkr,
                "amount_usd": payment.amount_usd,
                "currency": payment.currency,
                "payment_method": payment.payment_method,
                "card_type": payment.card_type,
                "provider": payment.provider,
                "client_secret": client_secret,
                "checkout_url": checkout_url,
                "created_at": payment.created_at,
            },
        },
        status=status.HTTP_201_CREATED,
    )


@extend_schema(
    methods=["POST"],
    operation_id="admin_payment_confirm",
    summary="Confirm a payment intent",
    description="Marks a sandbox transaction as paid. In production this would be done by webhook verification.",
    request=inline_serializer(
        name="ConfirmPaymentRequest",
        fields={
            "payment_id": serializers.IntegerField(required=False),
            "provider_intent_id": serializers.CharField(required=False),
        },
    ),
    responses={
        200: OpenApiResponse(description="Payment confirmed"),
        400: OpenApiResponse(description="Validation error"),
        404: OpenApiResponse(description="Payment not found"),
    },
)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def confirm_payment(request):
    admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err
    payment_id = _parse_int(request.data.get("payment_id"))
    provider_intent_id = str(request.data.get("provider_intent_id") or "").strip()
    payment = None
    if payment_id is not None:
        payment = repo.get_payment(payment_id)
    elif provider_intent_id:
        payment = repo.find_payment_by_provider_intent_id(provider_intent_id)
    else:
        return Response({"error": "payment_id or provider_intent_id is required."}, status=400)
    if not payment:
        return Response({"error": "Payment not found."}, status=404)
    if int(payment.company_id) != int(company_id) or int(payment.admin_id) != int(admin.id):
        return Response({"error": "Payment not found."}, status=404)
    if str(payment.provider).lower() != "sandbox":
        try:
            provider_payment = fetch_provider_payment_intent_for(
                provider=str(payment.provider).lower(),
                provider_intent_id=payment.provider_intent_id,
            )
        except Exception as exc:
            return Response({"error": f"Payment provider error: {str(exc)}"}, status=502)
        provider_status = str(provider_payment.status).lower()
        updates: Dict[str, Any] = {"status": provider_status}
        if provider_status in {"succeeded", "paid", "completed", "captured", "success"}:
            updates["status"] = "paid"
            updates["paid_at"] = datetime.now(timezone.utc).isoformat()
        elif provider_status in {"failed", "cancelled", "canceled", "declined", "error"}:
            updates["status"] = "failed"
        repo.update_payment(int(payment.id), updates)
        fresh = repo.get_payment(int(payment.id))
        return Response(
            {
                "message": "Payment status synchronized.",
                "payment": {
                    "id": fresh.id,
                    "provider_intent_id": fresh.provider_intent_id,
                    "status": fresh.status,
                    "paid_at": fresh.paid_at,
                    "amount_pkr": fresh.amount_pkr,
                    "amount_usd": fresh.amount_usd,
                },
            }
        )
    now_iso = datetime.now(timezone.utc).isoformat()
    repo.update_payment(int(payment.id), {"status": "paid", "paid_at": now_iso})
    fresh = repo.get_payment(int(payment.id))
    return Response(
        {
            "message": "Payment confirmed.",
            "payment": {
                "id": fresh.id,
                "provider_intent_id": fresh.provider_intent_id,
                "status": fresh.status,
                "paid_at": fresh.paid_at,
                "amount_pkr": fresh.amount_pkr,
                "amount_usd": fresh.amount_usd,
            },
        }
    )


@extend_schema(
    methods=["GET"],
    operation_id="admin_payment_status",
    summary="Get payment status",
    parameters=[
        OpenApiParameter("payment_id", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False),
        OpenApiParameter(
            "provider_intent_id",
            OpenApiTypes.STR,
            OpenApiParameter.QUERY,
            required=False,
        ),
    ],
    responses={
        200: OpenApiResponse(description="Payment status"),
        400: OpenApiResponse(description="Validation error"),
        404: OpenApiResponse(description="Payment not found"),
    },
)
@csrf_exempt
@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def payment_status(request):
    admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err
    payment_id = _parse_int(request.query_params.get("payment_id"))
    provider_intent_id = str(request.query_params.get("provider_intent_id") or "").strip()
    payment = repo.get_payment(payment_id) if payment_id is not None else None
    if payment is None and provider_intent_id:
        payment = repo.find_payment_by_provider_intent_id(provider_intent_id)
    if not payment:
        return Response({"error": "Payment not found."}, status=404)
    if int(payment.company_id) != int(company_id) or int(payment.admin_id) != int(admin.id):
        return Response({"error": "Payment not found."}, status=404)
    if str(payment.provider).lower() != "sandbox":
        try:
            provider_payment = fetch_provider_payment_intent_for(
                provider=str(payment.provider).lower(),
                provider_intent_id=payment.provider_intent_id,
            )
            provider_status = str(provider_payment.status).lower()
            updates: Dict[str, Any] = {"status": provider_status}
            if provider_status in {"succeeded", "paid", "completed", "captured", "success"}:
                updates["status"] = "paid"
                if not payment.paid_at:
                    updates["paid_at"] = datetime.now(timezone.utc).isoformat()
            elif provider_status in {"failed", "cancelled", "canceled", "declined", "error"}:
                updates["status"] = "failed"
            repo.update_payment(int(payment.id), updates)
            payment = repo.get_payment(int(payment.id)) or payment
        except Exception:
            # Keep local status if provider lookup temporarily fails.
            pass
    return Response(
        {
            "payment": {
                "id": payment.id,
                "provider_intent_id": payment.provider_intent_id,
                "status": payment.status,
                "bot_name": payment.bot_name,
                "amount_pkr": payment.amount_pkr,
                "amount_usd": payment.amount_usd,
                "currency": payment.currency,
                "payment_method": payment.payment_method,
                "card_type": payment.card_type,
                "created_at": payment.created_at,
                "updated_at": payment.updated_at,
                "paid_at": payment.paid_at,
            }
        }
    )


@extend_schema(
    methods=["POST"],
    operation_id="payment_webhook",
    summary="Payment webhook callback",
    description="Provider callback endpoint to update transaction status.",
    parameters=[
        OpenApiParameter(
            "provider",
            OpenApiTypes.STR,
            OpenApiParameter.QUERY,
            required=False,
            description="Provider key e.g. stripe, paypal, jazzcash, easypaisa, payfast, paypro, bank_transfer.",
        )
    ],
    request=inline_serializer(
        name="PaymentWebhookRequest",
        fields={
            "provider_intent_id": serializers.CharField(),
            "event": serializers.ChoiceField(choices=["payment.succeeded", "payment.failed"]),
        },
    ),
    responses={200: OpenApiResponse(description="Webhook processed")},
)
@csrf_exempt
@api_view(["POST"])
@permission_classes([])
@authentication_classes([])
def payment_webhook(request):
    provider = str(request.query_params.get("provider") or "").strip().lower()
    if not provider:
        provider = str(getattr(settings, "PAYMENT_PROVIDER", "sandbox")).strip().lower() or "sandbox"
    signature = (
        str(request.headers.get("Stripe-Signature") or "")
        or str(request.headers.get("X-Signature") or "")
        or str(request.headers.get("Paypal-Transmission-Sig") or "")
    )
    if provider == "stripe" and not signature:
        return Response({"error": "Missing Stripe-Signature header."}, status=400)
    try:
        event = parse_provider_webhook(provider=provider, payload=request.body, signature=signature)
    except Exception as exc:
        return Response({"error": f"Webhook verification failed: {str(exc)}"}, status=400)
    if provider == "stripe":
        event_type = str(event.get("type") or "").strip().lower()
        obj = ((event.get("data") or {}).get("object") or {}) if isinstance(event, dict) else {}
        provider_intent_id = str(obj.get("id") or "").strip()
    elif provider == "paypal":
        event_type = str(event.get("event_type") or "").strip().lower()
        resource = event.get("resource") or {}
        provider_intent_id = str((resource or {}).get("id") or "").strip()
    else:
        event_type = str(event.get("event") or event.get("status") or "").strip().lower()
        provider_intent_id = str(event.get("provider_intent_id") or event.get("id") or "").strip()
    if not provider_intent_id:
        return Response({"message": "ignored"}, status=200)
    payment = repo.find_payment_by_provider_intent_id(provider_intent_id)
    if not payment:
        return Response({"message": "ignored"}, status=200)
    success_events = {
        "payment_intent.succeeded",
        "checkout.order.approved",
        "payment.succeeded",
        "paid",
        "completed",
        "success",
    }
    failed_events = {
        "payment_intent.payment_failed",
        "payment_intent.canceled",
        "payment.failed",
        "failed",
        "cancelled",
        "canceled",
        "declined",
    }
    if event_type in success_events:
        repo.update_payment(
            int(payment.id),
            {"status": "paid", "paid_at": datetime.now(timezone.utc).isoformat()},
        )
    elif event_type in failed_events:
        repo.update_payment(int(payment.id), {"status": "failed"})
    return Response({"message": "ok"})


@extend_schema(
    methods=["GET"],
    operation_id="bots_list",
    summary="List bots for current company",
    responses={200: OpenApiResponse(description="Bots listed")},
)
@extend_schema(
    methods=["POST"],
    operation_id="bots_create",
    summary="List or create bots for current company",
    description="GET lists company bots. POST creates one bot per company.",
    request=inline_serializer(
        name="CreateBotRequest",
        fields={
            "name": serializers.CharField(required=False),
            "system_prompt": serializers.CharField(required=False, allow_blank=True),
            "openrouter_api_key": serializers.CharField(required=False, allow_blank=True),
        },
    ),
    responses={
        200: OpenApiResponse(description="Bots listed"),
        201: OpenApiResponse(description="Bot created"),
        400: OpenApiResponse(description="Validation error"),
    },
)
@csrf_exempt
@api_view(["GET", "POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def bots(request):
    admin_id = int(getattr(request.user, "id"))
    admin = repo.get_admin(admin_id)
    if not admin:
        return Response({"error": "Admin not found."}, status=status.HTTP_404_NOT_FOUND)

    company_id: Optional[int] = None
    if admin.company_id is not None:
        company_id = int(admin.company_id)
    elif str(getattr(request.user, "role", "")).lower() == "super_admin":
        company_raw = request.query_params.get("company_id")
        if request.method == "POST":
            company_raw = request.data.get("company_id") or company_raw
        parsed_company_id = _parse_int(company_raw)
        if parsed_company_id is None:
            return Response(
                {"error": "company_id is required for super admin context."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        company = repo.get_company(parsed_company_id)
        if not company:
            return Response({"error": "Company not found."}, status=status.HTTP_404_NOT_FOUND)
        company_id = int(parsed_company_id)
    else:
        return Response(
            {"error": "Admin is not linked to a company. Create company first."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if request.method == "GET":
        rows = repo.list_bots(company_id=company_id)
        return Response(
            [
                {
                    "id": b.id,
                    "company_id": b.company_id,
                    "name": b.name,
                    "system_prompt": b.system_prompt,
                    "openrouter_api_key": b.openrouter_api_key,
                    "plan_type": b.plan_type,
                    "widget_key": b.widget_key,
                    "created_at": b.created_at,
                }
                for b in rows
            ]
        )

    name = str(request.data.get("name") or "").strip()
    system_prompt = str(request.data.get("system_prompt") or "").strip()
    openrouter_api_key = str(request.data.get("openrouter_api_key") or "").strip()
    if not name:
        return Response({"error": "name is required."}, status=400)

    existing = repo.list_bots(company_id=company_id)
    if len(existing) >= 1:
        return Response(
            {"error": "Only one bot is allowed per company."},
            status=status.HTTP_400_BAD_REQUEST,
        )
    bot = repo.create_bot(
        {
            "company_id": company_id,
            "name": name,
            "system_prompt": system_prompt,
            "openrouter_api_key": openrouter_api_key,
            "plan_type": "free",
            "widget_key": f"bot_{secrets.token_urlsafe(24)}",
        }
    )
    return Response(
        {
            "message": "Bot created.",
            "bot": {
                "id": bot.id,
                "company_id": bot.company_id,
                "name": bot.name,
                "system_prompt": bot.system_prompt,
                "openrouter_api_key": bot.openrouter_api_key,
                "plan_type": bot.plan_type,
                "widget_key": bot.widget_key,
                "created_at": bot.created_at,
            },
        },
        status=status.HTTP_201_CREATED,
    )


@extend_schema(
    methods=["GET"],
    operation_id="bot_detail_get",
    summary="Get bot details",
    responses={
        200: OpenApiResponse(description="Bot details"),
        404: OpenApiResponse(description="Bot not found"),
    },
)
@extend_schema(
    methods=["PATCH"],
    operation_id="bot_detail_update",
    summary="Get or update bot details",
    description="Fetch bot configuration or update name/system prompt for a specific bot.",
    request=inline_serializer(
        name="UpdateBotRequest",
        fields={
            "name": serializers.CharField(required=False),
            "system_prompt": serializers.CharField(required=False, allow_blank=True),
            "openrouter_api_key": serializers.CharField(required=False, allow_blank=True),
        },
    ),
    responses={
        200: OpenApiResponse(description="Bot details"),
        400: OpenApiResponse(description="Validation error"),
        404: OpenApiResponse(description="Bot not found"),
    },
)
@csrf_exempt
@api_view(["PATCH", "GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def bot_detail(request, bot_id: int):
    _admin, company_id, bot, err = _current_admin_company_and_bot(request, bot_id_raw=bot_id)
    if err:
        return err
    if request.method == "GET":
        return Response(
            {
                "id": bot.id,
                "company_id": company_id,
                "name": bot.name,
                "system_prompt": bot.system_prompt,
                "openrouter_api_key": bot.openrouter_api_key,
                "plan_type": bot.plan_type,
                "widget_key": bot.widget_key,
                "created_at": bot.created_at,
            }
        )

    updates: Dict[str, Any] = {}
    if "name" in request.data:
        name = str(request.data.get("name") or "").strip()
        if not name:
            return Response({"error": "name cannot be empty."}, status=400)
        updates["name"] = name
    if "system_prompt" in request.data:
        updates["system_prompt"] = str(request.data.get("system_prompt") or "").strip()
    if "openrouter_api_key" in request.data:
        updates["openrouter_api_key"] = str(request.data.get("openrouter_api_key") or "").strip()
    if not updates:
        return Response({"error": "No updates provided."}, status=400)
    repo.update_bot(int(bot.id), updates)
    fresh = repo.get_bot(int(bot.id))
    return Response(
        {
            "message": "Bot updated.",
            "bot": {
                "id": fresh.id,
                "company_id": fresh.company_id,
                "name": fresh.name,
                "system_prompt": fresh.system_prompt,
                "openrouter_api_key": fresh.openrouter_api_key,
                "plan_type": fresh.plan_type,
                "widget_key": fresh.widget_key,
                "created_at": fresh.created_at,
            },
        }
    )


@extend_schema(
    operation_id="bot_generate_widget_script",
    summary="Generate embeddable widget script for bot",
    request=None,
    parameters=[
        OpenApiParameter("bot_id", OpenApiTypes.INT, OpenApiParameter.PATH, required=True),
    ],
    responses={200: OpenApiResponse(description="Widget script generated")},
)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def generate_widget_script(request, bot_id: int):
    _admin, _company_id, bot, err = _current_admin_company_and_bot(request, bot_id_raw=bot_id)
    if err:
        return err
    return Response(
        {
            "bot_id": bot.id,
            "widget_key": bot.widget_key,
            "widget_script": _build_widget_script(bot_widget_key=bot.widget_key, request=request),
        }
    )


@extend_schema(
    operation_id="public_widget_chat_query",
    summary="Public widget chat query",
    request=inline_serializer(
        name="PublicChatQueryRequest",
        fields={
            "widget_key": serializers.CharField(),
            "query": serializers.CharField(),
            "top_k": serializers.IntegerField(required=False),
            "threshold": serializers.FloatField(required=False),
            "session_id": serializers.CharField(required=False),
        },
    ),
    responses={
        200: OpenApiResponse(description="Chat response"),
        400: OpenApiResponse(description="Validation error"),
        404: OpenApiResponse(description="Invalid widget key"),
    },
)
@csrf_exempt
@api_view(["POST"])
def public_chat_query(request):
    """
    Public widget endpoint.
    Body: {"widget_key":"...", "query":"...", "top_k":5, "threshold":0.3, "session_id":"optional"}
    """
    widget_key = str(request.data.get("widget_key") or "").strip()
    query = request.data.get("query") or ""
    top_k = request.data.get("top_k", getattr(settings, "RAG_TOP_K", 5))
    threshold = request.data.get("threshold", getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.3))
    session_id = request.data.get("session_id") or "widget-default"
    if not widget_key:
        return Response({"error": "widget_key is required."}, status=400)
    bot = repo.find_bot_by_widget_key(widget_key)
    if not bot:
        return Response({"error": "Invalid widget key."}, status=404)
    try:
        top_k_int = int(top_k)
        threshold_f = float(threshold)
    except Exception:
        return Response({"error": "Invalid 'top_k' or 'threshold' types."}, status=status.HTTP_400_BAD_REQUEST)

    ensure_raw_answer_log_dir()
    t0 = time.time()
    result = run_rag_query(
        query=query,
        top_k=top_k_int,
        threshold=threshold_f,
        session_id=str(session_id),
        company_id=int(bot.company_id),
        bot_id=int(bot.id),
        bot_system_prompt=str(bot.system_prompt or ""),
        bot_openrouter_api_key=str(bot.openrouter_api_key or ""),
    )
    latency_ms = int((time.time() - t0) * 1000)
    citations = result.get("citations") or []
    ans_raw = result.get("answer") or FALLBACK_PHRASE
    ans_body = clean_answer_body_only(ans_raw, citations)
    append_meta = result.get("pipeline") != "greeting_short_circuit"
    ans = format_plain_answer_with_metadata(
        ans_body, citations, append_source_metadata=append_meta
    )
    return Response(
        {
            "answer": ans,
            "answer_html": build_chat_answer_html(
                answer=ans_body, citations=citations, append_source_metadata=append_meta
            ),
            "fallback_used": bool(result.get("fallback_used")),
            "retrieved": result.get("retrieved") or [],
            "citations": citations,
            "retrieval_diagnostics": result.get("retrieval_diagnostics") or {},
            "sentence_evidence": result.get("sentence_evidence") or [],
            "ab_variant": result.get("ab_variant"),
            "latency_ms": latency_ms,
            "top_k": result.get("top_k"),
            "threshold": result.get("threshold"),
            "rag_retrieval_ran": result.get("rag_retrieval_ran"),
            "pipeline": result.get("pipeline"),
            "bot_id": bot.id,
        },
        status=status.HTTP_200_OK,
    )



@extend_schema(
    operation_id="documents_ingest",
    summary="Upload and ingest document for selected bot",
    description="Uploads text/file, chunks it, embeds it, and stores vectors in Qdrant scoped to bot/company.",
    request=inline_serializer(
        name="UploadDocumentRequest",
        fields={
            "bot_id": serializers.IntegerField(),
            "name": serializers.CharField(required=False),
            "text": serializers.CharField(required=False),
            "file": serializers.FileField(required=False),
            "force_reindex": serializers.BooleanField(required=False),
        },
    ),
    responses={
        200: OpenApiResponse(description="Document already exists"),
        201: OpenApiResponse(description="Document ingested"),
        400: OpenApiResponse(description="Validation or ingest error"),
        401: OpenApiResponse(description="Embedding provider auth failure"),
    },
)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def upload_document(request):
    """
    Backwards-compatible ingestion endpoint.

    Accepts either:
      - JSON: {"text": "...", "name": "...optional"}
      - multipart: file upload under key "file" (.txt, .pdf, .docx)

    Optional handbook layout (preserves newlines in the file). Each chunk in Qdrant
    stores: text, source_section (subsection or heading), page_number, chapter_name.

    Explicit blocks:

        SOURCE_SECTION: 3.6 Gym reimbursement
        PAGE_NUMBER: 18
        Body text for this section...

    Chapter-style headings (auto-detected, line must start with Chapter/Ch.):

        Chapter 2: Reimbursement
        Page 48
        14.7 Purpose
        Body text...

    Aliases: PAGE: 18, SOURCE SECTION:, PAGE NUMBER:.

    If there are no SOURCE_SECTION lines but the file has chapters, page markers,
    or numbered headings (e.g. "14.7 Purpose", "=== Page 44 ==="), the line-aware
    parser fills source_section, page_number, and chapter_name.

    Plain prose with none of the above is chunked as one stream; metadata is
    inferred per chunk when possible.

    No file storage is used; only text is embedded into Qdrant.
    """
    raw_text = request.data.get("text")
    file_obj = request.FILES.get("file")

    if raw_text and file_obj:
        return Response({"error": "Provide either 'text' or 'file', not both."}, status=status.HTTP_400_BAD_REQUEST)
    _admin, company_id, bot, err = _current_admin_company_and_bot(request)
    if err:
        return err

    content: str
    doc_name: str
    if file_obj:
        content = extract_text_from_upload(file_obj) or ""
        if not content:
            return Response(
                {"error": "Unsupported or unreadable file. Supported formats: .txt, .pdf, .docx"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        doc_name = request.data.get("name") or getattr(file_obj, "name", "uploaded_text")
    else:
        content = str(raw_text or "")
        if not content.strip():
            return Response({"error": "No text provided."}, status=status.HTTP_400_BAD_REQUEST)
        doc_name = request.data.get("name") or "raw_text"

    handbook_norm = normalize_handbook_text(content)
    use_explicit_sections = handbook_has_section_markers(handbook_norm)
    use_auto_structure = (not use_explicit_sections) and handbook_has_auto_structure(handbook_norm)

    chunk_size_tokens = int(getattr(settings, "RAG_INGEST_CHUNK_SIZE_TOKENS", 500))
    overlap_tokens = int(getattr(settings, "RAG_INGEST_CHUNK_OVERLAP_TOKENS", 75))
    embedding_batch_size = int(getattr(settings, "EMBEDDING_BATCH_SIZE", 64))

    handbook_rows: List[Tuple[str, int, str, Optional[int], str]] = []
    if use_explicit_sections:
        handbook_rows = split_handbook_into_embedding_chunks(
            handbook_norm,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )
    elif use_auto_structure:
        handbook_rows = split_auto_structured_into_embedding_chunks(
            handbook_norm,
            chunk_size_tokens=chunk_size_tokens,
            overlap_tokens=overlap_tokens,
        )

    if handbook_rows:
        normalized = handbook_norm
        chunks = [r[0] for r in handbook_rows]
        token_counts = [r[1] for r in handbook_rows]
        chunk_meta = [(r[2], r[3], r[4]) for r in handbook_rows]
    else:
        normalized = normalize_text(content)
        if not normalized:
            return Response({"error": "Text is empty after normalization."}, status=status.HTTP_400_BAD_REQUEST)
        chunks, token_counts = split_text_into_token_chunks(
            normalized, chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
        )
        chunk_meta = [infer_legacy_chunk_metadata(c) for c in chunks]

    # Include ingest settings in the dedupe fingerprint so re-uploads after
    # chunking/config changes are re-indexed instead of silently reusing stale vectors.
    fingerprint = json.dumps(
        {
            "text": normalized,
            "chunk_size_tokens": chunk_size_tokens,
            "overlap_tokens": overlap_tokens,
            "embedding_model": str(getattr(settings, "OPENROUTER_EMBEDDING_MODEL", "")),
            "ingest_version": "v2",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    content_hash = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    existing = repo.find_document_by_content_hash(
        content_hash, company_id=company_id, bot_id=int(bot.id)
    )
    force_reindex_raw = request.data.get("force_reindex")
    force_reindex = str(force_reindex_raw or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if existing:
        if force_reindex:
            qdrant = QdrantService()
            qdrant.delete_by_doc_id(existing.id)
            repo.delete_document(existing.id)
        else:
            return Response(
                {
                    "message": "Document already exists",
                    "document_id": existing.id,
                    "chunk_count": existing.chunk_count,
                    "token_count": existing.token_count,
                    "embedding_count": existing.embedding_count,
                    "status": existing.status,
                },
                status=status.HTTP_200_OK,
            )

    if not chunks:
        return Response({"error": "Unable to chunk provided text."}, status=status.HTTP_400_BAD_REQUEST)

    chunk_count = len(chunks)
    token_count = sum(token_counts)
    embedding_count = chunk_count
    bot_openrouter_key = str(bot.openrouter_api_key or "").strip()
    if not bot_openrouter_key:
        return Response(
            {"error": "OpenRouter API key is required for this bot. Save it in bot settings first."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    doc = repo.create_document(
        {
            "name": doc_name,
            "company_id": company_id,
            "bot_id": int(bot.id),
            "content_hash": content_hash,
            "content_length": len(normalized),
            "chunk_count": chunk_count,
            "token_count": token_count,
            "embedding_count": embedding_count,
            "status": "processing",
        }
    )

    try:
        embeddings = get_embeddings(
            chunks,
            batch_size=embedding_batch_size,
            openrouter_api_key=bot_openrouter_key,
        )
        if len(embeddings) != chunk_count:
            raise RuntimeError("Embeddings returned unexpected count.")

        points: List[Dict[str, Any]] = []
        for i, (chunk_text, embedding, tok_count, meta) in enumerate(
            zip(chunks, embeddings, token_counts, chunk_meta)
        ):
            point_id = doc.id * 1_000_000 + i
            _section_title, _page_num, _chapter_name = meta
            payload: Dict[str, Any] = {
                "doc_id": doc.id,
                "company_id": company_id,
                "bot_id": int(bot.id),
                "chunk_index": i,
                "chunk_id": f"{doc.id}_{i}",
                "text": chunk_text,
                "token_count": tok_count,
            }
            points.append(
                {
                    "id": point_id,
                    "vector": embedding,
                    "payload": payload,
                }
            )

        qdrant = QdrantService()
        qdrant.add_embeddings(points)

        repo.update_document(doc.id, {"status": "ready", "error_message": ""})
        doc.status = "ready"
        doc.error_message = ""
    except Exception as e:
        doc.status = "error"
        doc.error_message = str(e)
        repo.update_document(doc.id, {"status": "error", "error_message": str(e)})
        msg = str(e)
        # Surface vector-size issues as a user-actionable 400 instead of an opaque traceback.
        if "vector dimension mismatch" in msg.lower() or "vector dimension error" in msg.lower():
            return Response(
                {
                    "error": msg,
                    "hint": (
                        "Embedding dimension and Qdrant collection size must match. "
                        "Set QDRANT_VECTOR_SIZE to your model dimension and use a fresh "
                        "QDRANT_COLLECTION_NAME (or recreate the collection)."
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        if isinstance(e, AuthenticationError) or (
            isinstance(e, APIStatusError) and getattr(e, "status_code", None) == 401
        ):
            return Response(
                {
                    "error": "Embedding API authentication failed (401).",
                    "details": msg,
                    "hint": (
                        "Set a valid OpenRouter key in this bot's OpenRouter field "
                        "(OpenRouter: https://openrouter.ai/keys). Invalid, expired, or revoked keys "
                        'return "User not found" from the provider.'
                    ),
                },
                status=status.HTTP_401_UNAUTHORIZED,
            )
        raise

    return Response(
        {
            "message": "Ingested successfully",
            "document_id": doc.id,
            "chunk_count": doc.chunk_count,
            "token_count": doc.token_count,
            "embedding_count": doc.embedding_count,
            "status": doc.status,
        },
        status=status.HTTP_201_CREATED,
    )


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def upload_document_alias(request):
    return upload_document(request)

@extend_schema(
    operation_id="documents_delete",
    summary="Delete a document and its vectors",
    parameters=[
        OpenApiParameter("bot_id", OpenApiTypes.INT, OpenApiParameter.QUERY, required=True),
    ],
    responses={
        200: OpenApiResponse(description="Deleted successfully"),
        400: OpenApiResponse(description="Qdrant deletion failure"),
        404: OpenApiResponse(description="Document not found"),
    },
)
@csrf_exempt
@api_view(['DELETE'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def delete_document(request, doc_id):
    _admin, company_id, bot, err = _current_admin_company_and_bot(request)
    if err:
        return err
    doc = repo.get_document(int(doc_id))

    if not doc:
        return Response({"error": "Document not found"}, status=404)
    if int(doc.company_id or -1) != int(company_id):
        return Response({"error": "Document not found"}, status=404)
    if int(doc.bot_id or -1) != int(bot.id):
        return Response({"error": "Document not found"}, status=404)

    try:
        # Deletion by payload filter does not depend on vector dimension.
        # Allow cleanup even if collection/model dimensions changed over time.
        qdrant = QdrantService(validate_vector_dimension=False)
        qdrant.delete_by_doc_id(int(doc_id))
    except Exception as e:
        return Response(
            {
                "error": "Failed to delete vectors from Qdrant.",
                "details": str(e),
            },
            status=status.HTTP_400_BAD_REQUEST,
        )

    repo.delete_document(int(doc_id))

    return Response({
        "message": "Deleted successfully"
    })

@extend_schema(
    operation_id="stats_get",
    summary="Get document/vector stats",
    parameters=[
        OpenApiParameter("bot_id", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False),
    ],
    responses={200: OpenApiResponse(description="Stats response")},
)
@csrf_exempt
@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def get_stats(request):
    _admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err
    bot_id = _parse_int(request.query_params.get("bot_id"))
    if bot_id is not None:
        bot = repo.get_bot(int(bot_id))
        if not bot or int(bot.company_id) != int(company_id):
            return Response({"error": "Bot not found."}, status=status.HTTP_404_NOT_FOUND)
        docs = repo.list_documents_for_bot(company_id=company_id, bot_id=int(bot_id))
        total_documents = len(docs)
        total_chunks = sum(int(d.chunk_count or 0) for d in docs)
        total_vectors = sum(int(d.embedding_count or 0) for d in docs)
        total_tokens = sum(int(d.token_count or 0) for d in docs)
        return Response(
            {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "total_vector_embeddings": total_vectors,
                "total_tokens": total_tokens,
            }
        )
    return Response(repo.get_stats(company_id=company_id))

@extend_schema(
    operation_id="documents_list",
    summary="List documents for company or bot",
    parameters=[
        OpenApiParameter("bot_id", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False),
    ],
    responses={200: OpenApiResponse(description="Document list")},
)
@csrf_exempt
@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def get_all_documents(request):
    _admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err
    bot_id = _parse_int(request.query_params.get("bot_id"))
    if bot_id is not None:
        bot = repo.get_bot(int(bot_id))
        if not bot or int(bot.company_id) != int(company_id):
            return Response({"error": "Bot not found."}, status=status.HTTP_404_NOT_FOUND)
        documents = repo.list_documents_for_bot(company_id=company_id, bot_id=int(bot_id))
    else:
        documents = repo.list_documents(company_id=company_id)

    data = []
    for doc in documents:
        data.append({
            "id": doc.id,
            "bot_id": doc.bot_id,
            "name": doc.name,
            "chunk_count": doc.chunk_count,
            "embedding_count": doc.embedding_count,
            "token_count": doc.token_count,
            "status": doc.status,
            "created_at": doc.created_at,
        })

    return Response(data)

@extend_schema(
    operation_id="chat_query",
    summary="Chat query (bot-scoped)",
    request=inline_serializer(
        name="ChatQueryRequest",
        fields={
            "bot_id": serializers.IntegerField(),
            "query": serializers.CharField(),
            "top_k": serializers.IntegerField(required=False),
            "threshold": serializers.FloatField(required=False),
            "session_id": serializers.CharField(required=False),
        },
    ),
    responses={
        200: OpenApiResponse(description="Chat answer"),
        400: OpenApiResponse(description="Validation error"),
        404: OpenApiResponse(description="Bot not found"),
    },
)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def chat_query(request):
    """
    POST /api/chat/query
    Body: {"query": "...", "top_k": 5, "threshold": 0.3, "session_id": "optional"}
    """
    query = request.data.get("query") or ""
    top_k = request.data.get("top_k", getattr(settings, "RAG_TOP_K", 5))
    threshold = request.data.get("threshold", getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.3))
    session_id = request.data.get("session_id") or "default"

    try:
        top_k_int = int(top_k)
        threshold_f = float(threshold)
    except Exception:
        return Response({"error": "Invalid 'top_k' or 'threshold' types."}, status=status.HTTP_400_BAD_REQUEST)
    _admin, company_id, bot, err = _current_admin_company_and_bot(request)
    if err:
        return err

    ensure_raw_answer_log_dir()
    t0 = time.time()
    result = run_rag_query(
        query=query,
        top_k=top_k_int,
        threshold=threshold_f,
        session_id=str(session_id),
        company_id=company_id,
        bot_id=int(bot.id),
        bot_system_prompt=str(bot.system_prompt or ""),
        bot_openrouter_api_key=str(bot.openrouter_api_key or ""),
    )
    latency_ms = int((time.time() - t0) * 1000)

    retrieved = result.get("retrieved") or []
    citations = result.get("citations") or []
    retrieved_chunk_ids: List[str] = []
    for c in citations:
        chunk_id = c.get("chunk_id")
        chunk_index = c.get("chunk_index")
        if chunk_id is not None:
            retrieved_chunk_ids.append(str(chunk_id))
        elif chunk_index is not None:
            retrieved_chunk_ids.append(str(chunk_index))

    ans_raw = result.get("answer") or FALLBACK_PHRASE
    ans_body = clean_answer_body_only(ans_raw, citations)
    append_meta = result.get("pipeline") != "greeting_short_circuit"
    ans = format_plain_answer_with_metadata(
        ans_body, citations, append_source_metadata=append_meta
    )

    repo.create_chat_message(
        {
            "session_id": str(session_id),
            "bot_id": int(bot.id),
            "query": str(query),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "model_used": str(getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")),
            "response_text": ans,
            "latency_ms": latency_ms,
            "fallback_used": bool(result.get("fallback_used")),
        }
    )
    append_rag_telemetry(
        {
            "session_id": str(session_id),
            "query": str(query),
            "ab_variant": result.get("ab_variant"),
            "pipeline": result.get("pipeline"),
            "fallback_used": bool(result.get("fallback_used")),
            "latency_ms": latency_ms,
            "retrieval_diagnostics": result.get("retrieval_diagnostics") or {},
        }
    )

    return Response(
        {
            "answer": ans,
            "answer_html": build_chat_answer_html(
                answer=ans_body, citations=citations, append_source_metadata=append_meta
            ),
            "fallback_used": bool(result.get("fallback_used")),
            "retrieved": retrieved,
            "citations": citations,
            "retrieval_diagnostics": result.get("retrieval_diagnostics") or {},
            "sentence_evidence": result.get("sentence_evidence") or [],
            "ab_variant": result.get("ab_variant"),
            "latency_ms": latency_ms,
            "top_k": result.get("top_k"),
            "threshold": result.get("threshold"),
            "rag_retrieval_ran": result.get("rag_retrieval_ran"),
            "pipeline": result.get("pipeline"),
        },
        status=status.HTTP_200_OK,
    )

@extend_schema(
    operation_id="chat_query_stream",
    summary="Streaming chat query (SSE, bot-scoped)",
    request=inline_serializer(
        name="ChatQueryStreamRequest",
        fields={
            "bot_id": serializers.IntegerField(),
            "query": serializers.CharField(),
            "top_k": serializers.IntegerField(required=False),
            "threshold": serializers.FloatField(required=False),
            "session_id": serializers.CharField(required=False),
        },
    ),
    responses={200: OpenApiResponse(description="SSE stream response")},
)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def chat_query_stream(request):
    """
    Server-Sent Events (SSE) variant of chat/query.
    Emits: start -> answer -> done (+ citations payload).

    Body: {"query": "...", "top_k": 5, "threshold": 0.3, "session_id": "optional"}
    """
    query = request.data.get("query") or ""
    top_k = request.data.get("top_k", getattr(settings, "RAG_TOP_K", 5))
    threshold = request.data.get("threshold", getattr(settings, "RAG_SIMILARITY_THRESHOLD", 0.3))
    session_id = request.data.get("session_id") or "default"

    try:
        top_k_int = int(top_k)
        threshold_f = float(threshold)
    except Exception:
        return Response({"error": "Invalid 'top_k' or 'threshold' types."}, status=status.HTTP_400_BAD_REQUEST)
    _admin, company_id, bot, err = _current_admin_company_and_bot(request)
    if err:
        return err

    ensure_raw_answer_log_dir()
    result = run_rag_query(
        query=query,
        top_k=top_k_int,
        threshold=threshold_f,
        session_id=str(session_id),
        company_id=company_id,
        bot_id=int(bot.id),
        bot_system_prompt=str(bot.system_prompt or ""),
        bot_openrouter_api_key=str(bot.openrouter_api_key or ""),
    )
    citations = result.get("citations") or []
    top_k_used = result.get("top_k")
    threshold_used = result.get("threshold")

    answer_raw = result.get("answer") or FALLBACK_PHRASE
    answer_body = clean_answer_body_only(answer_raw, citations)
    append_meta = result.get("pipeline") != "greeting_short_circuit"
    answer = format_plain_answer_with_metadata(
        answer_body, citations, append_source_metadata=append_meta
    )
    answer_html = build_chat_answer_html(
        answer=answer_body, citations=citations, append_source_metadata=append_meta
    )

    # Persist audit row (kept non-blocking for the client; still synchronous server-side).
    retrieved_chunk_ids: List[str] = []
    for c in citations:
        chunk_id = c.get("chunk_id")
        chunk_index = c.get("chunk_index")
        if chunk_id is not None:
            retrieved_chunk_ids.append(str(chunk_id))
        elif chunk_index is not None:
            retrieved_chunk_ids.append(str(chunk_index))
    repo.create_chat_message(
        {
            "session_id": str(session_id),
            "bot_id": int(bot.id),
            "query": str(query),
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "model_used": str(getattr(settings, "OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")),
            "response_text": answer,
            "latency_ms": int(result.get("latency_ms") or 0),
            "fallback_used": bool(result.get("fallback_used")),
        }
    )
    append_rag_telemetry(
        {
            "session_id": str(session_id),
            "query": str(query),
            "ab_variant": result.get("ab_variant"),
            "pipeline": result.get("pipeline"),
            "fallback_used": bool(result.get("fallback_used")),
            "latency_ms": int(result.get("latency_ms") or 0),
            "retrieval_diagnostics": result.get("retrieval_diagnostics") or {},
        }
    )

    def event_stream():
        # Minimal SSE envelope; frontend can render as it arrives.
        yield "event: start\ndata: {}\n\n"
        # JSON-encode to preserve newlines/spacing reliably in SSE clients.
        yield f"event: answer\ndata: {json.dumps(answer)}\n\n"
        yield f"event: answer_html\ndata: {json.dumps(answer_html)}\n\n"
        yield f"event: citations\ndata: {json.dumps(citations)}\n\n"
        yield f"event: done\ndata: {json.dumps({'top_k': top_k_used, 'threshold': threshold_used, 'rag_retrieval_ran': result.get('rag_retrieval_ran'), 'pipeline': result.get('pipeline'), 'retrieval_diagnostics': result.get('retrieval_diagnostics') or {}, 'sentence_evidence': result.get('sentence_evidence') or [], 'ab_variant': result.get('ab_variant')})}\n\n"

    response = StreamingHttpResponse(
        event_stream(),
        content_type="text/event-stream",
    )

    # Basic SSE-friendly headers; avoid hop-by-hop headers like 'Connection'
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"   # helps if you have nginx/reverse proxy

    return response


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def add_company_alias(request):
    return add_company(request)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["GET", "POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def bots_alias(request):
    return bots(request)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["PATCH", "GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def bot_detail_alias(request, bot_id: int):
    return bot_detail(request, bot_id)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def generate_widget_script_alias(request, bot_id: int):
    return generate_widget_script(request, bot_id)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def get_all_documents_alias(request):
    return get_all_documents(request)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(['GET'])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def get_stats_alias(request):
    return get_stats(request)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def documents_ingest_alias(request):
    return upload_document(request)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def chat_query_alias(request):
    return chat_query(request)


@extend_schema(exclude=True)
@csrf_exempt
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def chat_query_stream_alias(request):
    return chat_query_stream(request)