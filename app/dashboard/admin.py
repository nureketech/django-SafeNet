from django.contrib import admin

from app.dashboard.models import ApiToken, Case, Feedback


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    list_display = ("id", "risk_score", "risk_level", "primary_track", "source_type", "platform", "lang", "origin")
    list_filter = ("risk_level", "primary_track", "source_type", "platform", "lang", "origin")
    search_fields = ("id", "text", "account_id")


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("case_ref", "verdict", "username", "created_at")
    list_filter = ("verdict",)
    search_fields = ("case_ref", "comment", "username")
    readonly_fields = ("case", "case_ref", "verdict", "comment", "user", "username", "created_at")

    def has_add_permission(self, request):  # verdicts come from the dashboard only
        return False


@admin.register(ApiToken)
class ApiTokenAdmin(admin.ModelAdmin):
    """Tokens are issued with ``manage.py create_api_token``; here they can be reviewed and revoked."""

    list_display = ("name", "key_prefix", "user", "is_active", "created_at", "last_used_at")
    list_filter = ("is_active",)
    search_fields = ("name", "key_prefix")
    readonly_fields = ("key_prefix", "key_hash", "created_at", "last_used_at")
    actions = ["revoke"]

    def has_add_permission(self, request):
        return False

    @admin.action(description="Revoke selected tokens")
    def revoke(self, request, queryset):
        queryset.update(is_active=False)
