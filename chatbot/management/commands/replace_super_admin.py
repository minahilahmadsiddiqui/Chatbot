from __future__ import annotations

from django.contrib.auth.hashers import make_password
from django.core.management.base import BaseCommand, CommandError

from chatbot.repositories.firestore_repository import FirestoreRepository


class Command(BaseCommand):
    help = "Replace current super admin with a target account."

    def add_arguments(self, parser):
        parser.add_argument("--email", required=True, help="Target super admin email")
        parser.add_argument(
            "--password",
            required=False,
            help="Required only if the target email does not exist yet",
        )
        parser.add_argument(
            "--full-name",
            required=False,
            help="Required only if the target email does not exist yet",
        )

    def handle(self, *args, **options):
        email = str(options.get("email") or "").strip().lower()
        password = str(options.get("password") or "")
        full_name = str(options.get("full_name") or "").strip()
        if not email:
            raise CommandError("--email is required.")

        repo = FirestoreRepository()
        current_super = repo.find_super_admin()
        target = repo.find_admin_by_email(email)

        if target:
            repo.update_admin(
                target.id,
                {
                    "role": "super_admin",
                    "is_verified": True,
                    "verification_code": None,
                    "verification_expires_at": None,
                },
            )
            target_id = target.id
        else:
            if not password or not full_name:
                raise CommandError(
                    "Target email does not exist. Provide --password and --full-name to create it."
                )
            if len(password) < 8:
                raise CommandError("Password must be at least 8 characters.")
            created = repo.create_admin(
                {
                    "email": email,
                    "password_hash": make_password(password),
                    "full_name": full_name,
                    "company_id": None,
                    "role": "super_admin",
                    "is_verified": True,
                    "verification_code": None,
                    "verification_expires_at": None,
                }
            )
            target_id = created.id

        if current_super and int(current_super.id) != int(target_id):
            repo.delete_admin(current_super.id)

        self.stdout.write(
            self.style.SUCCESS(
                f"Super admin is now {email} (id={target_id}). Previous super admin removed."
                if current_super and int(current_super.id) != int(target_id)
                else f"Super admin is now {email} (id={target_id})."
            )
        )
