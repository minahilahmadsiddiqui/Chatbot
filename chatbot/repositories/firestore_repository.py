from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from firebase_admin import firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from chatbot.services.firebase_service import get_firestore_client


@dataclass
class DocumentRecord:
    id: int
    company_id: Optional[int]
    bot_id: Optional[int]
    name: str
    content_hash: Optional[str]
    content_length: Optional[int]
    chunk_count: Optional[int]
    token_count: Optional[int]
    embedding_count: Optional[int]
    status: str
    error_message: str
    created_at: str


@dataclass
class AdminRecord:
    id: int
    email: str
    password_hash: str
    full_name: str
    company_id: Optional[int]
    role: str
    is_verified: bool
    verification_code: Optional[str]
    verification_expires_at: Optional[str]
    created_at: str


@dataclass
class CompanyRecord:
    id: int
    name: str
    domain: str
    admin_id: int
    plan_type: str
    created_at: str


@dataclass
class BotRecord:
    id: int
    company_id: int
    name: str
    system_prompt: str
    plan_type: str
    widget_key: str
    created_at: str


class FirestoreRepository:
    def __init__(self) -> None:
        self.db = get_firestore_client()
        self.documents = self.db.collection("documents")
        self.chat_messages = self.db.collection("chat_messages")
        self.admins = self.db.collection("admins")
        self.companies = self.db.collection("companies")
        self.bots = self.db.collection("bots")
        self.auth_sessions = self.db.collection("auth_sessions")
        self.refresh_tokens = self.db.collection("refresh_tokens")
        self.counters = self.db.collection("_counters")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _sort_by_created_desc(self, rows: List[Any]) -> List[Any]:
        # created_at is stored in ISO-8601, so lexical sort is chronological.
        return sorted(rows, key=lambda r: str(getattr(r, "created_at", "")), reverse=True)

    def _document_from_snapshot(self, snap) -> Optional[DocumentRecord]:
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        return DocumentRecord(
            id=int(data.get("id", 0)),
            company_id=data.get("company_id"),
            bot_id=data.get("bot_id"),
            name=str(data.get("name", "")),
            content_hash=data.get("content_hash"),
            content_length=data.get("content_length"),
            chunk_count=data.get("chunk_count"),
            token_count=data.get("token_count"),
            embedding_count=data.get("embedding_count"),
            status=str(data.get("status", "ready")),
            error_message=str(data.get("error_message", "")),
            created_at=str(data.get("created_at", self._now_iso())),
        )

    def _admin_from_snapshot(self, snap) -> Optional[AdminRecord]:
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        return AdminRecord(
            id=int(data.get("id", 0)),
            email=str(data.get("email", "")).lower(),
            password_hash=str(data.get("password_hash", "")),
            full_name=str(data.get("full_name", "")),
            company_id=data.get("company_id"),
            role=str(data.get("role", "admin")),
            is_verified=bool(data.get("is_verified", False)),
            verification_code=data.get("verification_code"),
            verification_expires_at=data.get("verification_expires_at"),
            created_at=str(data.get("created_at", self._now_iso())),
        )

    def _company_from_snapshot(self, snap) -> Optional[CompanyRecord]:
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        return CompanyRecord(
            id=int(data.get("id", 0)),
            name=str(data.get("name", "")),
            domain=str(data.get("domain", "")).lower(),
            admin_id=int(data.get("admin_id", 0)),
            plan_type=str(data.get("plan_type", "free")).strip().lower() or "free",
            created_at=str(data.get("created_at", self._now_iso())),
        )

    def _bot_from_snapshot(self, snap) -> Optional[BotRecord]:
        if not snap.exists:
            return None
        data = snap.to_dict() or {}
        return BotRecord(
            id=int(data.get("id", 0)),
            company_id=int(data.get("company_id", 0)),
            name=str(data.get("name", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")),
            plan_type=str(data.get("plan_type", "free")).strip().lower() or "free",
            widget_key=str(data.get("widget_key", "")).strip(),
            created_at=str(data.get("created_at", self._now_iso())),
        )

    def _next_id(self, counter_name: str) -> int:
        counter_ref = self.counters.document(counter_name)
        tx = self.db.transaction()

        @firestore.transactional
        def update_counter(transaction):
            snap = counter_ref.get(transaction=transaction)
            current = int((snap.to_dict() or {}).get("value", 0)) if snap.exists else 0
            nxt = current + 1
            transaction.set(counter_ref, {"value": nxt}, merge=True)
            return nxt

        return int(update_counter(tx))

    def _next_document_id(self) -> int:
        return self._next_id("documents")

    def find_document_by_content_hash(
        self,
        content_hash: str,
        *,
        company_id: Optional[int] = None,
        bot_id: Optional[int] = None,
    ) -> Optional[DocumentRecord]:
        query_ref = self.documents.where(filter=FieldFilter("content_hash", "==", content_hash))
        if company_id is not None:
            query_ref = query_ref.where(filter=FieldFilter("company_id", "==", int(company_id)))
        if bot_id is not None:
            query_ref = query_ref.where(filter=FieldFilter("bot_id", "==", int(bot_id)))
        query = query_ref.limit(1).stream()
        for snap in query:
            return self._document_from_snapshot(snap)
        return None

    def get_document(self, document_id: int) -> Optional[DocumentRecord]:
        snap = self.documents.document(str(document_id)).get()
        return self._document_from_snapshot(snap)

    def create_document(self, payload: Dict[str, Any]) -> DocumentRecord:
        next_id = self._next_document_id()
        data = {
            "id": next_id,
            "company_id": payload.get("company_id"),
            "bot_id": payload.get("bot_id"),
            "name": payload.get("name", ""),
            "content_hash": payload.get("content_hash"),
            "content_length": payload.get("content_length"),
            "chunk_count": payload.get("chunk_count"),
            "token_count": payload.get("token_count"),
            "embedding_count": payload.get("embedding_count"),
            "status": payload.get("status", "processing"),
            "error_message": payload.get("error_message", ""),
            "created_at": payload.get("created_at") or self._now_iso(),
        }
        self.documents.document(str(next_id)).set(data)
        return DocumentRecord(**data)

    def update_document(self, document_id: int, updates: Dict[str, Any]) -> None:
        self.documents.document(str(document_id)).set(updates, merge=True)

    def delete_document(self, document_id: int) -> None:
        self.documents.document(str(document_id)).delete()

    def list_documents(self, *, company_id: Optional[int] = None) -> List[DocumentRecord]:
        docs = []
        query = self.documents
        if company_id is not None:
            query = query.where(filter=FieldFilter("company_id", "==", int(company_id)))
        for snap in query.stream():
            rec = self._document_from_snapshot(snap)
            if rec:
                docs.append(rec)
        return self._sort_by_created_desc(docs)

    def list_documents_for_bot(self, *, company_id: int, bot_id: int) -> List[DocumentRecord]:
        docs = []
        query = (
            self.documents.where(filter=FieldFilter("company_id", "==", int(company_id)))
            .where(filter=FieldFilter("bot_id", "==", int(bot_id)))
        )
        for snap in query.stream():
            rec = self._document_from_snapshot(snap)
            if rec:
                docs.append(rec)
        return self._sort_by_created_desc(docs)

    def get_stats(self, *, company_id: Optional[int] = None) -> Dict[str, int]:
        total_documents = 0
        total_chunks = 0
        total_vector_embeddings = 0
        total_tokens = 0
        query = self.documents
        if company_id is not None:
            query = query.where(filter=FieldFilter("company_id", "==", int(company_id)))
        for snap in query.stream():
            total_documents += 1
            data = snap.to_dict() or {}
            total_chunks += int(data.get("chunk_count") or 0)
            total_vector_embeddings += int(data.get("embedding_count") or 0)
            total_tokens += int(data.get("token_count") or 0)
        return {
            "total_documents": total_documents,
            "total_chunks": total_chunks,
            "total_vector_embeddings": total_vector_embeddings,
            "total_tokens": total_tokens,
        }

    def create_chat_message(self, payload: Dict[str, Any]) -> None:
        data = {
            "session_id": payload.get("session_id", "default"),
            "bot_id": payload.get("bot_id"),
            "query": payload.get("query", ""),
            "retrieved_chunk_ids": payload.get("retrieved_chunk_ids") or [],
            "model_used": payload.get("model_used", ""),
            "response_text": payload.get("response_text", ""),
            "latency_ms": int(payload.get("latency_ms") or 0),
            "fallback_used": bool(payload.get("fallback_used")),
            "created_at": self._now_iso(),
        }
        self.chat_messages.add(data)

    def find_admin_by_email(self, email: str) -> Optional[AdminRecord]:
        normalized = str(email or "").strip().lower()
        query = self.admins.where(filter=FieldFilter("email", "==", normalized)).limit(1).stream()
        for snap in query:
            return self._admin_from_snapshot(snap)
        return None

    def get_admin(self, admin_id: int) -> Optional[AdminRecord]:
        snap = self.admins.document(str(admin_id)).get()
        return self._admin_from_snapshot(snap)

    def find_super_admin(self) -> Optional[AdminRecord]:
        query = self.admins.where(filter=FieldFilter("role", "==", "super_admin")).limit(1).stream()
        for snap in query:
            return self._admin_from_snapshot(snap)
        return None

    def create_admin(self, payload: Dict[str, Any]) -> AdminRecord:
        next_id = self._next_id("admins")
        data = {
            "id": next_id,
            "email": str(payload.get("email", "")).strip().lower(),
            "password_hash": str(payload.get("password_hash", "")),
            "full_name": str(payload.get("full_name", "")).strip(),
            "company_id": payload.get("company_id"),
            "role": str(payload.get("role", "admin")),
            "is_verified": bool(payload.get("is_verified", False)),
            "verification_code": payload.get("verification_code"),
            "verification_expires_at": payload.get("verification_expires_at"),
            "created_at": self._now_iso(),
        }
        self.admins.document(str(next_id)).set(data)
        return AdminRecord(**data)

    def update_admin(self, admin_id: int, updates: Dict[str, Any]) -> None:
        self.admins.document(str(admin_id)).set(updates, merge=True)

    def delete_admin(self, admin_id: int) -> None:
        self.admins.document(str(admin_id)).delete()

    def list_admins(self, *, company_id: Optional[int] = None) -> List[AdminRecord]:
        out: List[AdminRecord] = []
        query = self.admins
        if company_id is not None:
            query = query.where(filter=FieldFilter("company_id", "==", int(company_id)))
        for snap in query.stream():
            admin = self._admin_from_snapshot(snap)
            if admin:
                out.append(admin)
        return out

    def create_session(self, *, admin_id: int, token: str, expires_at: str) -> None:
        data = {
            "token": token,
            "admin_id": int(admin_id),
            "created_at": self._now_iso(),
            "expires_at": expires_at,
        }
        self.auth_sessions.document(token).set(data)

    def get_session(self, token: str) -> Optional[Dict[str, Any]]:
        snap = self.auth_sessions.document(token).get()
        if not snap.exists:
            return None
        return snap.to_dict() or {}

    def store_refresh_token(self, *, jti: str, admin_id: int, expires_at: str) -> None:
        data = {
            "jti": jti,
            "admin_id": int(admin_id),
            "created_at": self._now_iso(),
            "expires_at": expires_at,
        }
        self.refresh_tokens.document(jti).set(data)

    def get_refresh_token(self, jti: str) -> Optional[Dict[str, Any]]:
        snap = self.refresh_tokens.document(jti).get()
        if not snap.exists:
            return None
        return snap.to_dict() or {}

    def revoke_refresh_token(self, jti: str) -> None:
        self.refresh_tokens.document(jti).delete()

    def find_company_by_name(self, name: str) -> Optional[CompanyRecord]:
        normalized = str(name or "").strip().lower()
        query = self.companies.where(filter=FieldFilter("name_normalized", "==", normalized)).limit(1).stream()
        for snap in query:
            return self._company_from_snapshot(snap)
        return None

    def get_company(self, company_id: int) -> Optional[CompanyRecord]:
        snap = self.companies.document(str(company_id)).get()
        return self._company_from_snapshot(snap)

    def update_company(self, company_id: int, updates: Dict[str, Any]) -> None:
        self.companies.document(str(company_id)).set(updates, merge=True)

    def list_companies(self) -> List[CompanyRecord]:
        out: List[CompanyRecord] = []
        for snap in self.companies.stream():
            company = self._company_from_snapshot(snap)
            if company:
                out.append(company)
        return self._sort_by_created_desc(out)

    def create_company(self, payload: Dict[str, Any]) -> CompanyRecord:
        next_id = self._next_id("companies")
        name = str(payload.get("name", "")).strip()
        data = {
            "id": next_id,
            "name": name,
            "name_normalized": name.lower(),
            # retained for backward compatibility with existing records/schema
            "domain": str(payload.get("domain", "")).strip().lower(),
            "admin_id": int(payload.get("admin_id")),
            "plan_type": str(payload.get("plan_type", "free")).strip().lower() or "free",
            "created_at": self._now_iso(),
        }
        self.companies.document(str(next_id)).set(data)
        return CompanyRecord(
            id=int(data["id"]),
            name=str(data["name"]),
            domain=str(data.get("domain", "")),
            admin_id=int(data["admin_id"]),
            plan_type=str(data.get("plan_type", "free")).strip().lower() or "free",
            created_at=str(data["created_at"]),
        )

    def create_bot(self, payload: Dict[str, Any]) -> BotRecord:
        next_id = self._next_id("bots")
        data = {
            "id": next_id,
            "company_id": int(payload.get("company_id")),
            "name": str(payload.get("name", "")).strip(),
            "system_prompt": str(payload.get("system_prompt", "")),
            "plan_type": str(payload.get("plan_type", "free")).strip().lower() or "free",
            "widget_key": str(payload.get("widget_key", "")).strip(),
            "created_at": self._now_iso(),
        }
        self.bots.document(str(next_id)).set(data)
        return BotRecord(**data)

    def get_bot(self, bot_id: int) -> Optional[BotRecord]:
        snap = self.bots.document(str(bot_id)).get()
        return self._bot_from_snapshot(snap)

    def list_bots(self, *, company_id: int) -> List[BotRecord]:
        out: List[BotRecord] = []
        query = self.bots.where(filter=FieldFilter("company_id", "==", int(company_id)))
        for snap in query.stream():
            bot = self._bot_from_snapshot(snap)
            if bot:
                out.append(bot)
        return self._sort_by_created_desc(out)

    def list_all_bots(self) -> List[BotRecord]:
        out: List[BotRecord] = []
        for snap in self.bots.stream():
            bot = self._bot_from_snapshot(snap)
            if bot:
                out.append(bot)
        return self._sort_by_created_desc(out)

    def update_bot(self, bot_id: int, updates: Dict[str, Any]) -> None:
        self.bots.document(str(bot_id)).set(updates, merge=True)

    def find_bot_by_widget_key(self, widget_key: str) -> Optional[BotRecord]:
        key = str(widget_key or "").strip()
        if not key:
            return None
        query = self.bots.where(filter=FieldFilter("widget_key", "==", key)).limit(1).stream()
        for snap in query:
            return self._bot_from_snapshot(snap)
        return None

    def count_chat_messages(self) -> int:
        total = 0
        for _ in self.chat_messages.stream():
            total += 1
        return total

    def count_chat_messages_for_bot_ids(self, *, bot_ids: List[int]) -> int:
        if not bot_ids:
            return 0
        total = 0
        values = [int(v) for v in bot_ids]
        # Firestore "in" supports up to 10 values; chunk for safety.
        for i in range(0, len(values), 10):
            chunk = values[i : i + 10]
            query = self.chat_messages.where(filter=FieldFilter("bot_id", "in", chunk))
            for _ in query.stream():
                total += 1
        return total

    def count_fallback_chat_messages(self) -> int:
        total = 0
        query = self.chat_messages.where(filter=FieldFilter("fallback_used", "==", True))
        for _ in query.stream():
            total += 1
        return total

    def count_fallback_chat_messages_for_bot_ids(self, *, bot_ids: List[int]) -> int:
        if not bot_ids:
            return 0
        total = 0
        values = [int(v) for v in bot_ids]
        for i in range(0, len(values), 10):
            chunk = values[i : i + 10]
            query = (
                self.chat_messages.where(filter=FieldFilter("bot_id", "in", chunk))
                .where(filter=FieldFilter("fallback_used", "==", True))
            )
            for _ in query.stream():
                total += 1
        return total
