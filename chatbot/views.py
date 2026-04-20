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
from django.http import StreamingHttpResponse
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


def _send_verification_email(email: str, code: str) -> None:
    subject = "Verify your admin account"
    message = f"Your verification code is: {code}"
    from_email = getattr(settings, "DEFAULT_FROM_EMAIL", None) or getattr(
        settings, "EMAIL_HOST_USER", "no-reply@example.com"
    )
    try:
        send_mail(subject, message, from_email, [email], fail_silently=True)
    except Exception:
        # In local/dev environments without email configured we ignore failures.
        pass


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
    methods=["GET"],
    operation_id="bots_list",
    summary="List bots for current company",
    responses={200: OpenApiResponse(description="Bots listed")},
)
@extend_schema(
    methods=["POST"],
    operation_id="bots_create",
    summary="List or create bots for current company",
    description="GET lists company bots. POST creates a bot with system prompt and plan type.",
    request=inline_serializer(
        name="CreateBotRequest",
        fields={
            "name": serializers.CharField(required=False),
            "system_prompt": serializers.CharField(required=False, allow_blank=True),
            "plan_type": serializers.ChoiceField(choices=["free", "paid"], required=False),
        },
    ),
    responses={
        200: OpenApiResponse(description="Bots listed"),
        201: OpenApiResponse(description="Bot created"),
        400: OpenApiResponse(description="Validation error"),
        402: OpenApiResponse(description="Plan limit reached"),
    },
)
@csrf_exempt
@api_view(["GET", "POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAdmin])
def bots(request):
    _admin, company_id, err = _current_admin_and_company_id(request)
    if err:
        return err

    if request.method == "GET":
        rows = repo.list_bots(company_id=company_id)
        return Response(
            [
                {
                    "id": b.id,
                    "company_id": b.company_id,
                    "name": b.name,
                    "system_prompt": b.system_prompt,
                    "plan_type": b.plan_type,
                    "widget_key": b.widget_key,
                    "created_at": b.created_at,
                }
                for b in rows
            ]
        )

    name = str(request.data.get("name") or "").strip()
    system_prompt = str(request.data.get("system_prompt") or "").strip()
    requested_plan = str(request.data.get("plan_type") or "free").strip().lower() or "free"
    if requested_plan not in {"free", "paid"}:
        return Response({"error": "plan_type must be either 'free' or 'paid'."}, status=400)
    if not name:
        return Response({"error": "name is required."}, status=400)

    existing = repo.list_bots(company_id=company_id)
    max_bots = _company_plan_max_bots(company_id)
    if len(existing) >= max_bots:
        return Response(
            {
                "error": "Bot limit reached for current plan.",
                "current_bots": len(existing),
                "max_bots": max_bots,
                "hint": "Upgrade company plan to add more bots.",
            },
            status=status.HTTP_402_PAYMENT_REQUIRED,
        )

    # First bot can be free. Additional bots should be explicitly paid.
    if len(existing) >= 1 and requested_plan != "paid":
        return Response(
            {"error": "Additional bots require paid plan_type='paid'."},
            status=status.HTTP_402_PAYMENT_REQUIRED,
        )
    bot = repo.create_bot(
        {
            "company_id": company_id,
            "name": name,
            "system_prompt": system_prompt,
            "plan_type": requested_plan,
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
        embeddings = get_embeddings(chunks, batch_size=embedding_batch_size)
        if len(embeddings) != chunk_count:
            raise RuntimeError("Embeddings returned unexpected count.")

        points: List[Dict[str, Any]] = []
        for i, (chunk_text, embedding, tok_count, meta) in enumerate(
            zip(chunks, embeddings, token_counts, chunk_meta)
        ):
            point_id = doc.id * 1_000_000 + i
            section_title, page_num, chapter_name = meta
            payload: Dict[str, Any] = {
                "doc_id": doc.id,
                "company_id": company_id,
                "bot_id": int(bot.id),
                "chunk_index": i,
                "chunk_id": f"{doc.id}_{i}",
                "text": chunk_text,
                "token_count": tok_count,
                "source_section": section_title or "",
                "page_number": page_num,
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
                        "Set a valid OPENROUTER_API_KEY in .env (OpenRouter: "
                        "https://openrouter.ai/keys). Invalid, expired, or revoked keys "
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