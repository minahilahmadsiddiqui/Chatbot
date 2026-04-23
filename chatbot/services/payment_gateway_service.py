from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import hashlib
import hmac
import secrets

from django.conf import settings
import httpx


@dataclass
class PaymentIntentResult:
    provider_intent_id: str
    status: str
    client_secret: Optional[str]
    checkout_url: Optional[str]
    provider: str
    raw: Dict[str, Any]


def _provider_name() -> str:
    return str(getattr(settings, "PAYMENT_PROVIDER", "sandbox")).strip().lower() or "sandbox"


def _map_method_to_provider(payment_method: str) -> str:
    mapping = {
        "card": "stripe",
        "paypal": "paypal",
        "easypaisa": "easypaisa",
        "jazzcash": "jazzcash",
        "bank": "bank_transfer",
        "bank_transfer": "bank_transfer",
        "payfast": "payfast",
        "paypro": "paypro",
    }
    return mapping.get(str(payment_method or "").strip().lower(), "unknown")


def _norm_status(value: str) -> str:
    s = str(value or "").strip().lower()
    if s in {"succeeded", "paid", "completed", "captured", "success"}:
        return "paid"
    if s in {"failed", "canceled", "cancelled", "declined", "error"}:
        return "failed"
    return s or "pending"


def _generic_provider_conf(provider: str) -> Dict[str, str]:
    key = provider.upper()
    return {
        "base_url": str(getattr(settings, f"{key}_BASE_URL", "") or "").strip(),
        "create_path": str(getattr(settings, f"{key}_CREATE_PATH", "/payments/create") or "").strip(),
        "status_path": str(getattr(settings, f"{key}_STATUS_PATH", "/payments/{id}") or "").strip(),
        "api_key": str(getattr(settings, f"{key}_API_KEY", "") or "").strip(),
        "api_secret": str(getattr(settings, f"{key}_API_SECRET", "") or "").strip(),
        "webhook_secret": str(getattr(settings, f"{key}_WEBHOOK_SECRET", "") or "").strip(),
    }


def create_provider_payment_intent(
    *,
    payment_method: str,
    amount_major: float,
    currency: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> PaymentIntentResult:
    provider_mode = _provider_name()
    metadata = metadata or {}
    provider = _map_method_to_provider(payment_method)
    if provider_mode == "sandbox":
        return PaymentIntentResult(
            provider_intent_id=f"pi_{secrets.token_hex(12)}",
            status="pending",
            client_secret=None,
            checkout_url=None,
            provider="sandbox",
            raw={"provider_mode": "sandbox"},
        )
    if provider_mode == "stripe_fake":
        fake_intent = f"pi_test_{secrets.token_hex(12)}"
        fake_client_secret = f"{fake_intent}_secret_{secrets.token_hex(16)}"
        return PaymentIntentResult(
            provider_intent_id=fake_intent,
            status="requires_payment_method",
            client_secret=fake_client_secret,
            checkout_url=None,
            provider="stripe_fake",
            raw={"provider_mode": "stripe_fake", "payment_method": payment_method},
        )
    if provider == "stripe":
        try:
            import stripe
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Stripe SDK is not installed. Install 'stripe'.") from exc
        secret_key = str(getattr(settings, "STRIPE_SECRET_KEY", "") or "").strip()
        if not secret_key:
            raise RuntimeError("Missing STRIPE_SECRET_KEY.")
        stripe.api_key = secret_key
        amount_minor = int(round(float(amount_major) * 100))
        intent = stripe.PaymentIntent.create(
            amount=amount_minor,
            currency=str(currency or "pkr").lower(),
            automatic_payment_methods={"enabled": True},
            metadata=metadata,
        )
        return PaymentIntentResult(
            provider_intent_id=str(intent.get("id", "")),
            status=str(intent.get("status", "requires_payment_method")),
            client_secret=intent.get("client_secret"),
            checkout_url=None,
            provider="stripe",
            raw=dict(intent),
        )
    if provider == "paypal":
        client_id = str(getattr(settings, "PAYPAL_CLIENT_ID", "") or "").strip()
        client_secret = str(getattr(settings, "PAYPAL_CLIENT_SECRET", "") or "").strip()
        base_url = str(getattr(settings, "PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com") or "").strip()
        if not client_id or not client_secret:
            raise RuntimeError("Missing PAYPAL_CLIENT_ID or PAYPAL_CLIENT_SECRET.")
        with httpx.Client(timeout=30.0) as client:
            token_resp = client.post(
                f"{base_url}/v1/oauth2/token",
                auth=(client_id, client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
            )
            token_resp.raise_for_status()
            access_token = token_resp.json().get("access_token")
            if not access_token:
                raise RuntimeError("Unable to obtain PayPal access token.")
            order_resp = client.post(
                f"{base_url}/v2/checkout/orders",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {access_token}",
                },
                json={
                    "intent": "CAPTURE",
                    "purchase_units": [
                        {
                            "amount": {
                                "currency_code": str(currency or "usd").upper(),
                                "value": f"{float(amount_major):.2f}",
                            },
                            "custom_id": json.dumps(metadata)[:127],
                        }
                    ],
                    "application_context": {"user_action": "PAY_NOW"},
                },
            )
            order_resp.raise_for_status()
            order = order_resp.json()
        approve_url = None
        for link in order.get("links", []) or []:
            if str(link.get("rel", "")).lower() == "approve":
                approve_url = link.get("href")
                break
        return PaymentIntentResult(
            provider_intent_id=str(order.get("id", "")),
            status=str(order.get("status", "created")).lower(),
            client_secret=None,
            checkout_url=approve_url,
            provider="paypal",
            raw=order,
        )
    if provider in {"easypaisa", "jazzcash", "payfast", "paypro", "bank_transfer"}:
        conf = _generic_provider_conf(provider)
        if not conf["base_url"]:
            raise RuntimeError(f"Missing configuration for {provider}.")
        url = f"{conf['base_url'].rstrip('/')}/{conf['create_path'].lstrip('/')}"
        payload = {
            "amount": float(amount_major),
            "currency": str(currency or "pkr").upper(),
            "metadata": metadata,
            "method": payment_method,
        }
        headers = {"Content-Type": "application/json"}
        if conf["api_key"]:
            headers["Authorization"] = f"Bearer {conf['api_key']}"
        if conf["api_secret"]:
            headers["X-Api-Secret"] = conf["api_secret"]
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        return PaymentIntentResult(
            provider_intent_id=str(data.get("id") or data.get("payment_id") or data.get("transaction_id") or ""),
            status=_norm_status(str(data.get("status", "pending"))),
            client_secret=data.get("client_secret"),
            checkout_url=data.get("checkout_url") or data.get("redirect_url"),
            provider=provider,
            raw=data,
        )
    raise RuntimeError(f"Unsupported payment method: {payment_method}")


def fetch_provider_payment_intent(provider_intent_id: str) -> PaymentIntentResult:
    return fetch_provider_payment_intent_for(provider="stripe", provider_intent_id=provider_intent_id)


def fetch_provider_payment_intent_for(*, provider: str, provider_intent_id: str) -> PaymentIntentResult:
    provider = str(provider or "").strip().lower()
    if provider == "stripe_fake":
        fake_client_secret = f"{provider_intent_id}_secret_{secrets.token_hex(8)}"
        return PaymentIntentResult(
            provider_intent_id=provider_intent_id,
            status="succeeded",
            client_secret=fake_client_secret,
            checkout_url=None,
            provider="stripe_fake",
            raw={"provider_mode": "stripe_fake", "provider_intent_id": provider_intent_id},
        )
    if provider == "stripe":
        try:
            import stripe
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Stripe SDK is not installed. Install 'stripe'.") from exc
        secret_key = str(getattr(settings, "STRIPE_SECRET_KEY", "") or "").strip()
        if not secret_key:
            raise RuntimeError("Missing STRIPE_SECRET_KEY.")
        stripe.api_key = secret_key
        intent = stripe.PaymentIntent.retrieve(provider_intent_id)
        return PaymentIntentResult(
            provider_intent_id=str(intent.get("id", "")),
            status=str(intent.get("status", "")),
            client_secret=intent.get("client_secret"),
            checkout_url=None,
            provider="stripe",
            raw=dict(intent),
        )
    if provider == "paypal":
        client_id = str(getattr(settings, "PAYPAL_CLIENT_ID", "") or "").strip()
        client_secret = str(getattr(settings, "PAYPAL_CLIENT_SECRET", "") or "").strip()
        base_url = str(getattr(settings, "PAYPAL_BASE_URL", "https://api-m.sandbox.paypal.com") or "").strip()
        if not client_id or not client_secret:
            raise RuntimeError("Missing PAYPAL_CLIENT_ID or PAYPAL_CLIENT_SECRET.")
        with httpx.Client(timeout=30.0) as client:
            token_resp = client.post(
                f"{base_url}/v1/oauth2/token",
                auth=(client_id, client_secret),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data={"grant_type": "client_credentials"},
            )
            token_resp.raise_for_status()
            access_token = token_resp.json().get("access_token")
            detail_resp = client.get(
                f"{base_url}/v2/checkout/orders/{provider_intent_id}",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            detail_resp.raise_for_status()
            order = detail_resp.json()
        return PaymentIntentResult(
            provider_intent_id=str(order.get("id", "")),
            status=_norm_status(str(order.get("status", "pending"))),
            client_secret=None,
            checkout_url=None,
            provider="paypal",
            raw=order,
        )
    if provider in {"easypaisa", "jazzcash", "payfast", "paypro", "bank_transfer"}:
        conf = _generic_provider_conf(provider)
        if not conf["base_url"]:
            raise RuntimeError(f"Missing configuration for {provider}.")
        path = conf["status_path"].replace("{id}", provider_intent_id)
        url = f"{conf['base_url'].rstrip('/')}/{path.lstrip('/')}"
        headers = {}
        if conf["api_key"]:
            headers["Authorization"] = f"Bearer {conf['api_key']}"
        if conf["api_secret"]:
            headers["X-Api-Secret"] = conf["api_secret"]
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return PaymentIntentResult(
            provider_intent_id=provider_intent_id,
            status=_norm_status(str(data.get("status", "pending"))),
            client_secret=data.get("client_secret"),
            checkout_url=data.get("checkout_url") or data.get("redirect_url"),
            provider=provider,
            raw=data,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def parse_provider_webhook(*, provider: str, payload: bytes, signature: str) -> Dict[str, Any]:
    provider = str(provider or "").strip().lower()
    if provider == "stripe_fake":
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid stripe_fake webhook payload: {exc}") from exc
    if provider == "stripe":
        try:
            import stripe
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Stripe SDK is not installed. Install 'stripe'.") from exc
        secret_key = str(getattr(settings, "STRIPE_SECRET_KEY", "") or "").strip()
        webhook_secret = str(getattr(settings, "STRIPE_WEBHOOK_SECRET", "") or "").strip()
        if not secret_key or not webhook_secret:
            raise RuntimeError("Missing STRIPE_SECRET_KEY or STRIPE_WEBHOOK_SECRET.")
        stripe.api_key = secret_key
        event = stripe.Webhook.construct_event(payload=payload, sig_header=signature, secret=webhook_secret)
        return dict(event)
    if provider == "paypal":
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid PayPal webhook payload: {exc}") from exc
    if provider in {"easypaisa", "jazzcash", "payfast", "paypro", "bank_transfer"}:
        conf = _generic_provider_conf(provider)
        secret = conf["webhook_secret"]
        if secret:
            expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
            if not signature or not hmac.compare_digest(signature, expected):
                raise RuntimeError("Webhook signature verification failed.")
        try:
            return json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid webhook payload: {exc}") from exc
    raise RuntimeError(f"Unsupported provider for webhook parsing: {provider}")
