"""``AnalysisResult`` — the single output format of the system (API, dashboard, eval)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from safenet import DISCLAIMER_KK, DISCLAIMER_RU


class EvidenceOut(BaseModel):
    text: str
    start: int
    end: int
    source: Literal["text", "ocr", "transcript"] = "text"


class SignalOut(BaseModel):
    id: str
    track: str
    name_ru: str
    name_kk: str
    weight: float
    core: bool = False
    contribution: float
    evidence: list[EvidenceOut] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)


class ComboOut(BaseModel):
    id: str
    track: str
    name_ru: str
    name_kk: str
    contribution: float


class AdjustmentOut(BaseModel):
    id: str
    name_ru: str
    name_kk: str
    contribution: float


class TrackOut(BaseModel):
    track: Literal["lottery", "casino", "pyramid"]
    probability: float = Field(ge=0, le=1)
    score: float


class MLOut(BaseModel):
    top_class: str
    probability: float = Field(ge=0, le=1)
    contribution: float
    available: bool = True


class NetworkOut(BaseModel):
    cluster_id: str | None = None
    cluster_size: int | None = None
    cluster_risk: float | None = None
    shared_entities: list[str] = Field(default_factory=list)
    contribution: float = 0.0


class AnalysisResult(BaseModel):
    id: str
    risk_score: int = Field(ge=0, le=100)
    risk_level: Literal["low", "medium", "high"]
    primary_track: Literal["lottery", "casino", "pyramid", "none"]
    tracks: list[TrackOut]
    signals: list[SignalOut]
    negative_signals: list[SignalOut]
    combos: list[ComboOut] = Field(default_factory=list)
    adjustments: list[AdjustmentOut] = Field(default_factory=list)
    ml: MLOut
    network: NetworkOut = Field(default_factory=NetworkOut)
    explanation_ru: str
    explanation_kk: str
    disclaimer_kk: str = DISCLAIMER_KK
    disclaimer_ru: str = DISCLAIMER_RU
    text: str = ""
    lang: str | None = None
    source_type: str | None = None
    platform: str | None = None
    account_id: str | None = None
    entities: dict[str, list[str]] = Field(default_factory=dict)
    thresholds: dict[str, float] = Field(default_factory=dict)
    ocr_text: str | None = None
    ocr_confidence: float | None = None
    ocr_engine: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def contributions_total(self) -> float:
        """Sum of every component; equals ``risk_score`` up to rounding."""
        return (
            sum(s.contribution for s in self.signals)
            + sum(s.contribution for s in self.negative_signals)
            + sum(c.contribution for c in self.combos)
            + sum(a.contribution for a in self.adjustments)
            + self.ml.contribution
            + self.network.contribution
        )
