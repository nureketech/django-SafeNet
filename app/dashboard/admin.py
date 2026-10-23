from django.contrib import admin

from app.dashboard.models import Case, Feedback


@admin.register(Case)
class CaseAdmin(admin.ModelAdmin):
    list_display = ("id", "risk_score", "risk_level", "primary_track", "source_type", "platform", "lang", "origin")
    list_filter = ("risk_level", "primary_track", "source_type", "platform", "lang", "origin")
    search_fields = ("id", "text", "account_id")


@admin.register(Feedback)
class FeedbackAdmin(admin.ModelAdmin):
    list_display = ("case_ref", "verdict", "created_at")
    list_filter = ("verdict",)
    search_fields = ("case_ref", "comment")
