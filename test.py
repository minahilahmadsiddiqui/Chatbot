import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from chatbot.services.firebase_service import get_firestore_client

db = get_firestore_client()

def delete_collection_where(collection_name, predicate=None):
    refs = db.collection(collection_name).stream()
    count = 0
    for snap in refs:
        data = snap.to_dict() or {}
        if predicate is None or predicate(data):
            snap.reference.delete()
            count += 1
    print(f"Deleted {count} from {collection_name}")

def main():
    # Extra cleanup you asked for:
    delete_collection_where("documents")
    delete_collection_where("chat_messages")

    # Existing cleanup (optional, keep super admin):
    delete_collection_where("bots")
    delete_collection_where("companies")
    delete_collection_where(
        "admins",
        lambda d: str(d.get("role", "")).strip().lower() != "super_admin"
    )

if __name__ == "__main__":
    main()