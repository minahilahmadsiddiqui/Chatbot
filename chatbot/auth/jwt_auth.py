from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import jwt
from django.conf import settings
from django.utils.timezone import now
from rest_framework import authentication, exceptions

from chatbot.repositories.firestore_repository import FirestoreRepository, AdminRecord


def _jwt_secret() -> str:
    return getattr(settings, "JWT_SECRET_KEY", settings.SECRET_KEY)


def _jwt_alg() -> str:
    return getattr(settings, "JWT_ALGORITHM", "HS256")


def _access_lifetime() -> datetime.timedelta:
    minutes = int(getattr(settings, "JWT_ACCESS_TOKEN_LIFETIME_MIN", 15))
    return datetime.timedelta(minutes=minutes)


def _refresh_lifetime() -> datetime.timedelta:
    days = int(getattr(settings, "JWT_REFRESH_TOKEN_LIFETIME_DAYS", 7))
    return datetime.timedelta(days=days)


def generate_access_token(admin: AdminRecord) -> str:
    issued_at = now()
    payload = {
        "sub": str(admin.id),
        "email": admin.email,
        "role": admin.role,
        "company_id": admin.company_id,
        "type": "access",
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + _access_lifetime()).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_jwt_alg())


def generate_refresh_token(admin: AdminRecord, *, jti: str) -> str:
    issued_at = now()
    payload = {
        "sub": str(admin.id),
        "email": admin.email,
        "role": admin.role,
        "company_id": admin.company_id,
        "type": "refresh",
        "jti": jti,
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + _refresh_lifetime()).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_jwt_alg())


@dataclass
class AdminUser:
    id: int
    email: str
    role: str
    company_id: Optional[int]

    @property
    def is_authenticated(self) -> bool:  # DRF compatibility
        return True

    @property
    def pk(self) -> int:  # Django/DRF compatibility for throttling, permissions, etc.
        return self.id


class JWTAuthentication(authentication.BaseAuthentication):
    """
    DRF authentication class that validates a Bearer JWT access token and
    resolves the current admin from Firestore.
    """

    def authenticate(self, request) -> Optional[Tuple[AdminUser, Any]]:
        auth_header = request.headers.get("Authorization") or ""
        auth_header = str(auth_header).strip()
        if not auth_header:
            return None
        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        token = parts[1]
        try:
            payload = jwt.decode(token, _jwt_secret(), algorithms=[_jwt_alg()])
        except jwt.ExpiredSignatureError:
            raise exceptions.AuthenticationFailed("Access token expired.")
        except jwt.InvalidTokenError:
            raise exceptions.AuthenticationFailed("Invalid access token.")

        if payload.get("type") != "access":
            raise exceptions.AuthenticationFailed("Invalid token type.")

        admin_id_raw = payload.get("sub")
        try:
            admin_id = int(admin_id_raw)
        except Exception:
            raise exceptions.AuthenticationFailed("Invalid admin id in token.")

        repo = FirestoreRepository()
        admin = repo.get_admin(admin_id)
        if not admin:
            raise exceptions.AuthenticationFailed("Admin not found.")
        if not admin.is_verified:
            raise exceptions.AuthenticationFailed("Admin email not verified.")

        user = AdminUser(
            id=admin.id,
            email=admin.email,
            role=admin.role,
            company_id=admin.company_id,
        )
        return user, token


class IsAdmin:
    def has_permission(self, request, view) -> bool:
        user = getattr(request, "user", None)
        if not getattr(user, "is_authenticated", False):
            return False
        return getattr(user, "role", "") in {"admin", "super_admin"}


class IsSuperAdmin:
    def has_permission(self, request, view) -> bool:
        user = getattr(request, "user", None)
        if not getattr(user, "is_authenticated", False):
            return False
        return getattr(user, "role", "") == "super_admin"

