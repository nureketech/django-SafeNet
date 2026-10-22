"""Turn a :class:`Breakdown` into the public :class:`AnalysisResult`."""

from __future__ import annotations

from safenet import TRACKS
from safenet.common import mask_card, mask_phone
from safenet.preprocess import NormalizedText
from safenet.scoring.config import ScoringConfig
from safenet.scoring.explain import adjustment_names, explain
from safenet.scoring.models import (
    AdjustmentOut,
    AnalysisResult,
    ComboOut,
    EvidenceOut,
    MLOut,
    NetworkOut,
    SignalOut,
    TrackOut,
)
from safenet.scoring.score import Breakdown
from safenet.signals import SignalHit


def _r(value: float) -> float:
    return round(float(value), 1)


def _signal_out(hit: SignalHit, contribution: float) -> SignalOut:
    return SignalOut(
        id=hit.id, track=hit.track, name_ru=hit.name_ru, name_kk=hit.name_kk, weight=hit.weight, core=hit.core,
        contribution=_r(contribution),
        evidence=[EvidenceOut(text=e.text, start=e.start, end=e.end, source=e.source) for e in hit.evidence],
        details=dict(hit.details),
    )


def masked_entities(nt: NormalizedText) -> dict[str, list[str]]:
    """Entities for display: card and phone numbers are always masked."""
    ents = nt.entity_dict()
    ents["cards"] = [mask_card(c) if "*" not in c else c for c in ents["cards"]]
    ents["phones"] = [mask_phone(p) if "*" not in p else p for p in ents["phones"]]
    return {k: v for k, v in ents.items() if v}


def build_result(
    *,
    item_id: str,
    text: str,
    nt: NormalizedText,
    breakdown: Breakdown,
    cfg: ScoringConfig,
    network: NetworkOut | None = None,
    source_type: str | None = None,
    platform: str | None = None,
    account_id: str | None = None,
) -> AnalysisResult:
    b = breakdown
    network = network or NetworkOut()
    network.contribution = _r(b.network_contribution)
    tracks = sorted(
        (TrackOut(track=t, probability=round(b.track_probability.get(t, 0.0), 4), score=_r(b.track_scores[t]))
         for t in TRACKS),
        key=lambda t: t.score,
        reverse=True,
    )
    adjustments = []
    for adj in b.adjustments:
        ru, kk = adjustment_names(adj.id)
        adjustments.append(AdjustmentOut(id=adj.id, name_ru=ru, name_kk=kk, contribution=_r(adj.value)))
    return AnalysisResult(
        id=item_id,
        risk_score=b.rounded,
        risk_level=b.level,
        primary_track=b.primary_track,
        tracks=tracks,
        signals=[_signal_out(h, h.weight) for h in sorted(b.positive, key=lambda h: -h.weight)],
        negative_signals=[_signal_out(h, -h.weight) for h in b.negative],
        combos=[ComboOut(id=c["id"], track=c["track"], name_ru=c["name_ru"], name_kk=c["name_kk"],
                         contribution=_r(c["bonus"])) for c in b.combos],
        adjustments=adjustments,
        ml=MLOut(top_class=b.ml_top_class, probability=round(b.ml_probability, 4),
                 contribution=_r(b.ml_contribution), available=b.ml_available),
        network=network,
        explanation_ru=explain(b, "ru"),
        explanation_kk=explain(b, "kk"),
        text=text,
        lang=nt.lang,
        source_type=source_type,
        platform=platform,
        account_id=account_id,
        entities=masked_entities(nt),
        thresholds={"medium": cfg.medium, "high": cfg.high},
    )
