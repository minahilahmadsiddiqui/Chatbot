import os
from functools import lru_cache
from pathlib import Path

import firebase_admin
from django.conf import settings
from firebase_admin import credentials, firestore


def _resolve_credentials_path() -> str:
    configured = (getattr(settings, "FIREBASE_CREDENTIALS_PATH", "") or "").strip()
    if configured:
        return configured
    env_value = (os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "") or "").strip()
    if env_value:
        return env_value
    return str(Path(settings.BASE_DIR).parent / "acme-one-chatbot-firebase-adminsdk-fbsvc-def9733648.json")


@lru_cache(maxsize=1)
def get_firestore_client():
    if not firebase_admin._apps:
        cred_path = _resolve_credentials_path()
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()
