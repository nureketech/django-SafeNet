"""Analysed cases shown to analysts and the analysts' verdicts."""

from __future__ import annotations

from django.db import models

from safenet.scoring.models import AnalysisResult


class Case(models.Model):
    """One analysed item with its full ``AnalysisResult`` (JSON)."""

    ORIGINS = [("dataset", "dataset"), ("check", "check"), ("demo", "demo")]

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
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.case_ref}: {self.verdict}"
