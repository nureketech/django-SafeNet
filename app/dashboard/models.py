"""Analysed cases shown to analysts and the analysts' verdicts."""

from __future__ import annotations

import hashlib
import secrets

from django.conf import settings
from django.db import models
from django.utils import timezone

from safenet.scoring.models import AnalysisResult


class Case(models.Model):
    """One analysed item with its full ``AnalysisResult`` (JSON)."""

    ORIGINS = [("dataset", "dataset"), ("check", "check"), ("demo", "demo"), ("api", "api")]

    id = models.CharField(primary_key=True, max_length=64)
    text = models.TextField(blank=True)
    source_type = models.CharField(max_length=32, blank=True, db_index=True)
    platform = models.CharField(max_length=32, blank=True, db_index=True)
    lang = models.CharField(max_length=16, blank=True, db_index=True)
    account_id = models.CharField(max_length=100, blank=True, db_index=True)
    image_path = models.CharField(max_length=255, blank=True)
    risk_score = models.IntegerField(db_index=True)
    risk_level = models.CharField(max_length=8, db_index=True)
    primary_track = models.CharField(max_length=16, db_index=True)
    result = models.JSONField()
    origin = models.CharField(max_length=16, choices=ORIGINS, default="dataset")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-risk_score", "id"]

    def __str__(self) -> str:
        return f"{self.id} ({self.risk_level}, {self.risk_score})"

    @property
    def analysis(self) -> AnalysisResult:
        return AnalysisResult.model_validate(self.result)

    @classmethod
    def from_result(cls, result: AnalysisResult, *, image_path: str = "", origin: str = "dataset") -> "Case":
        return cls(
            id=result.id[:64],
            text=result.text,
            source_type=result.source_type or "",
            platform=result.platform or "",
            lang=result.lang or "",
            account_id=result.account_id or "",
            image_path=image_path,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            primary_track=result.primary_track,
            result=result.model_dump(mode="json"),
            origin=origin,
        )


class Feedback(models.Model):
    """Analyst verdict on a case (also appended to ``data/feedback/feedback.csv``)."""

    VERDICTS = [
        ("confirmed", "Расталды"),
        ("false_positive", "Жалған іске қосылу"),
        ("missed", "Өткізіп жіберу"),
    ]

    case = models.ForeignKey(Case, null=True, blank=True, on_delete=models.SET_NULL, related_name="feedback")
    case_ref = models.CharField(max_length=64)
    verdict = models.CharField(max_length=16, choices=VERDICTS)
    comment = models.TextField(blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                             related_name="safenet_feedback")
    username = models.CharField(max_length=150, blank=True, help_text="snapshot kept if the account is deleted")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.case_ref}: {self.verdict}"


class ApiToken(models.Model):
    """Bearer token of an integration (collector, platform, other system); only its SHA-256 is stored."""

    PREFIX = "sn_"

    name = models.CharField(max_length=100, help_text="who uses it, e.g. «telegram-collector»")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                             related_name="safenet_api_tokens")
    key_prefix = models.CharField(max_length=12, db_index=True)
    key_hash = models.CharField(max_length=64, unique=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.key_prefix}…)"

    @staticmethod
    def hash_key(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @classmethod
    def issue(cls, name: str, user=None) -> tuple["ApiToken", str]:
        """Create a token and return it with the plain key (shown once, never stored)."""
        key = cls.PREFIX + secrets.token_urlsafe(32)
        token = cls.objects.create(name=name, user=user, key_prefix=key[:10], key_hash=cls.hash_key(key))
        return token, key

    @classmethod
    def authenticate(cls, key: str) -> "ApiToken | None":
        if not key or not key.startswith(cls.PREFIX):
            return None
        token = cls.objects.filter(key_hash=cls.hash_key(key), is_active=True).first()
        if token is not None:
            cls.objects.filter(pk=token.pk).update(last_used_at=timezone.now())
        return token
