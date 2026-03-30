from django.urls import path

from chatbot.views import chat_query, chat_query_stream, delete_document, get_all_documents, get_stats, upload_document


urlpatterns = [
    path("documents/", get_all_documents),
    path("stats/", get_stats),
    # Production endpoints
    path("documents/ingest/", upload_document),
    path("chat/query/", chat_query),
    path("chat/query/stream/", chat_query_stream),
    # Backwards-compatible aliases
    path("upload/", upload_document),
    path("delete/<int:doc_id>/", delete_document),
]