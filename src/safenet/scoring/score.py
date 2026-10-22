"""Risk score from fired signals, ML probabilities and the network bonus (TZ, stage 5).

``compute_score`` is a pure function: the same signals and probabilities always give
the same breakdown, and every point of the score is attributed to a component
(signal, combo, ML, network, negative signal or an explicit adjustment), so the
contributions always add up to the score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from safenet import TRACKS
from safenet.scoring.config import ScoringConfig, load_config
from safenet.signals import SignalHit

LEVELS = ("low", "medium", "high")


@dataclass
class Adjustment:
    id: str
    value: float


@dataclass
class Breakdown:
    """Everything the explanation and the API result need."""

    score: float
    level: str
    positive: list[SignalHit]
    negative: list[SignalHit]
    combos: list[dict]
    rule_raw: float
    ml_top_class: str
    ml_probability: float
    ml_contribution: float
    network_contribution: float
    adjustments: list[Adjustment] = field(default_factory=list)
    track_scores: dict[str, float] = field(default_factory=dict)
    track_probability: dict[str, float] = field(default_factory=dict)
    primary_track: str = "none"
    has_core: bool = False
    ml_available: bool = True

    @property
    def rounded(self) -> int:
        return int(round(self.score))


def level_for(score: float, cfg: ScoringConfig) -> str:
    score = round(score)
    if score >= cfg.high:
        return "high"
    if score >= cfg.medium:
        return "medium"
    return "low"


def matching_combos(fired: set[str], cfg: ScoringConfig) -> list[dict]:
    out = []
    for combo in cfg.combos:
        if not all(s in fired for s in combo.get("all", [])):
            continue
        if combo.get("any") and not any(s in fired for s in combo["any"]):
            continue
        out.append(combo)
    return out


def compute_score(
    hits: Sequence[SignalHit],
    ml_proba: Mapping[str, float] | None,
    network_bonus: float = 0.0,
    cfg: ScoringConfig | None = None,
) -> Breakdown:
    """Apply the TZ formula; see ``config/scoring.yaml``."""
    cfg = cfg or load_config()
    positive = [h for h in hits if not h.negative and h.evidence]
    negative = [h for h in hits if h.negative and h.evidence]
    fired = {h.id for h in positive}
    combos = matching_combos(fired, cfg)
    rule_raw = sum(h.weight for h in positive) + sum(float(c["bonus"]) for c in combos)
    adjustments: list[Adjustment] = []

    rule = rule_raw
    if rule_raw > cfg.rule_cap:
        adjustments.append(Adjustment("rule_cap", cfg.rule_cap - rule_raw))
        rule = cfg.rule_cap

    ml_available = ml_proba is not None
    proba = dict(ml_proba or {})
    track_proba = {t: float(proba.get(t, 0.0)) for t in TRACKS}
    ml_top_class = max(track_proba, key=track_proba.get) if ml_available else "none"
    ml_probability = track_proba[ml_top_class] if ml_available else 0.0
    if ml_available and float(proba.get("none", 0.0)) >= ml_probability:
        top_overall = "none"
    else:
        top_overall = ml_top_class
    ml_contribution = cfg.ml_weight * ml_probability
    network_contribution = min(max(float(network_bonus), 0.0), cfg.network_max)
    negative_total = sum(h.weight for h in negative)

    score = rule + ml_contribution - negative_total + network_contribution
    if score < 0:
        adjustments.append(Adjustment("clip_low", -score))
        score = 0.0
    if score > 100:
        adjustments.append(Adjustment("clip_high", 100.0 - score))
        score = 100.0
    has_core = any(h.core for h in positive)
    if not has_core and score > cfg.no_core_cap:
        adjustments.append(Adjustment("no_core", cfg.no_core_cap - score))
        score = cfg.no_core_cap
    if cfg.require_evidence_for_medium and not positive and round(score) >= cfg.medium:
        cap = float(int(cfg.medium) - 1) if float(cfg.medium).is_integer() else cfg.medium - 1.0
        adjustments.append(Adjustment("no_evidence", cap - score))
        score = cap

    track_scores = {}
    for track in TRACKS:
        rules_t = sum(h.weight for h in positive if h.track == track)
        rules_t += sum(float(c["bonus"]) for c in combos if c["track"] == track)
        track_scores[track] = rules_t + cfg.ml_weight * track_proba[track]
    primary = max(track_scores, key=track_scores.get)
    if track_scores[primary] <= 0 or (not positive and top_overall == "none"):
        primary = "none"

    return Breakdown(
        score=score,
        level=level_for(score, cfg),
        positive=positive,
        negative=negative,
        combos=combos,
        rule_raw=rule_raw,
        ml_top_class=ml_top_class,
        ml_probability=ml_probability,
        ml_contribution=ml_contribution,
        network_contribution=network_contribution,
        adjustments=adjustments,
        track_scores=track_scores,
        track_probability=track_proba,
        primary_track=primary,
        has_core=has_core,
        ml_available=ml_available,
    )
