from django.urls import path

from chatbot.views import (
    add_company,
    add_company_alias,
    admin_login,
    admin_signup,
    bot_detail,
    bot_detail_alias,
    bots,
    bots_alias,
    chat_query,
    chat_query_alias,
    chat_query_stream,
    chat_query_stream_alias,
    delete_document,
    documents_ingest_alias,
    generate_widget_script,
    generate_widget_script_alias,
    get_all_documents,
    get_all_documents_alias,
    get_stats,
    get_stats_alias,
    logout,
    public_chat_query,
    refresh_token,
    resend_verification_code,
    super_admin_companies,
    super_admin_company_detail,
    super_admin_dashboard_overview,
    super_admin_update_company_plan,
    super_admin_only_example,
    upload_document,
    upload_document_alias,
    verify_email,
)


urlpatterns = [
    # Auth (shared)
    path("auth/signup/", admin_signup),
    path("auth/login/", admin_login),
    path("auth/verify-email/", verify_email),
    path("auth/resend-verification-code/", resend_verification_code),
    path("auth/refresh/", refresh_token),
    path("auth/logout/", logout),

    # Super admin APIs
    path("auth/super-admin-example/", super_admin_only_example),
    path("super-admin/overview/", super_admin_dashboard_overview),
    path("super-admin/companies/", super_admin_companies),
    path("super-admin/companies/<int:company_id>/", super_admin_company_detail),
    path("super-admin/companies/<int:company_id>/plan/", super_admin_update_company_plan),

    # Company admin APIs
    path("admin/companies/", add_company),
    path("admin/bots/", bots),
    path("admin/bots/<int:bot_id>/", bot_detail),
    path("admin/bots/<int:bot_id>/widget-script/", generate_widget_script),
    path("admin/documents/", get_all_documents),
    path("admin/stats/", get_stats),
    path("admin/documents/ingest/", upload_document),
    path("admin/chat/query/", chat_query),
    path("admin/chat/query/stream/", chat_query_stream),

    # Public APIs
    path("public/chat/query/", public_chat_query),

    # Backwards-compatible aliases (legacy paths)
    path("companies/", add_company_alias),
    path("bots/", bots_alias),
    path("bots/<int:bot_id>/", bot_detail_alias),
    path("bots/<int:bot_id>/widget-script/", generate_widget_script_alias),
    path("documents/", get_all_documents_alias),
    path("stats/", get_stats_alias),
    path("documents/ingest/", documents_ingest_alias),
    path("chat/query/", chat_query_alias),
    path("chat/query/stream/", chat_query_stream_alias),
    path("upload/", upload_document_alias),
    path("delete/<int:doc_id>/", delete_document),
]