from __future__ import annotations

from django.contrib.auth.hashers import make_password
from django.core.management.base import BaseCommand, CommandError

from chatbot.repositories.firestore_repository import FirestoreRepository


class Command(BaseCommand):
    help = "Create the initial platform super admin (one-time bootstrap)."

    def add_arguments(self, parser):
        parser.add_argument("--email", required=True, help="Super admin email")
        parser.add_argument("--password", required=True, help="Super admin password (min 8 chars)")
        parser.add_argument("--full-name", required=True, help="Super admin full name")

    def handle(self, *args, **options):
        email = str(options.get("email") or "").strip().lower()
        password = str(options.get("password") or "")
        full_name = str(options.get("full_name") or "").strip()

        if not email or not password or not full_name:
            raise CommandError("--email, --password, and --full-name are required.")
        if len(password) < 8:
            raise CommandError("Password must be at least 8 characters.")

        repo = FirestoreRepository()

        existing_super = repo.find_super_admin()
        if existing_super:
            existing_id = getattr(existing_super, "id", "unknown")
            existing_email = getattr(existing_super, "email", "unknown")
            raise CommandError(
                f"Super admin already exists (id={existing_id}, email={existing_email}). "
                "Bootstrap is one-time only."
            )

        existing_email = repo.find_admin_by_email(email)
        if existing_email:
            raise CommandError("An admin with this email already exists.")

        admin = repo.create_admin(
            {
                "email": email,
                "password_hash": make_password(password),
                "full_name": full_name,
                "company_id": None,
                "role": "super_admin",
                # Bootstrap is an operator-controlled flow, so account is immediately usable.
                "is_verified": True,
                "verification_code": None,
                "verification_expires_at": None,
            }
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Super admin created successfully (id={admin.id}, email={admin.email})."
            )
        )
