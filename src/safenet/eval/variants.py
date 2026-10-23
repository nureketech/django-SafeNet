"""Ablation variants and their decision thresholds (chosen on val with the stage-5 criteria).

* ``rules``         — signals, combos, negative signals, limiter; no ML;
* ``ml``            — only the classifier: score = 100 * P(most likely track != none);
* ``hybrid``        — the production scoring (rules + ML), stage-5 thresholds;
* ``hybrid_ocr``    — hybrid where screenshots are read from the image by OCR;
* ``hybrid_graph``  — hybrid + network bonus from the account graph.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from safenet import TRACKS
from safenet.eval.collect import Record
from safenet.scoring.config import ScoringConfig
from safenet.scoring.score import compute_score

VARIANTS = ("rules", "ml", "hybrid", "hybrid_ocr", "hybrid_graph")
VARIANT_NAMES = {
    "rules": "Тек ережелер",
    "ml": "Тек ML",
    "hybrid": "Гибрид (ережелер + ML)",
    "hybrid_ocr": "Гибрид + OCR",
    "hybrid_graph": "Гибрид + граф",
}


@dataclass
class Scored:
    score: np.ndarray        # raw score (before the evidence cap)
    has_evidence: np.ndarray
    track: np.ndarray        # predicted track when flagged
    level_score: np.ndarray  # score used for low/medium/high


def _raw_cfg(cfg: ScoringConfig) -> ScoringConfig:
    return dataclasses.replace(cfg, require_evidence_for_medium=False)


def score_records(records: list[Record], variant: str, cfg: ScoringConfig, graph=None) -> Scored | None:
    """Scores of every record under ``variant``; ``None`` when the variant cannot run (e.g. no OCR)."""
    raw = _raw_cfg(cfg)
    if variant == "hybrid_ocr" and not any(r.ocr_hits is not None for r in records if r.source_type == "screenshot"):
        has_screens = any(r.source_type == "screenshot" for r in records)
        if has_screens:
            return None
    scores, evidence, tracks = [], [], []
    for r in records:
        if variant == "ml":
            proba = {t: r.proba.get(t, 0.0) for t in TRACKS}
            top = max(proba, key=proba.get)
            scores.append(100.0 * proba[top])
            evidence.append(True)
            tracks.append(top)
            continue
        hits, proba = r.hits, r.proba
        if variant == "hybrid_ocr" and r.ocr_hits is not None:
            hits, proba = r.ocr_hits, r.ocr_proba
        if variant == "rules":
            proba = None
        bonus = 0.0
        if variant == "hybrid_graph" and graph is not None and r.cluster is not None:
            pre = compute_score(hits, proba, 0.0, raw)
            bonus = graph.bonus(r.cluster, pre.score, cfg.medium, cfg.network_max)
        b = compute_score(hits, proba, bonus, raw)
        scores.append(b.score)
        evidence.append(bool(b.positive))
        tracks.append(b.primary_track)
    s = np.array(scores, dtype=float)
    return Scored(s, np.array(evidence), np.array(tracks, dtype=object), s)


def flagged(scored: Scored, threshold: float) -> np.ndarray:
    return (np.round(scored.score) >= threshold) & scored.has_evidence


def choose_threshold(scored: Scored, labels: np.ndarray, hard_neg: np.ndarray, tuning: dict,
                     candidates=range(5, 96)) -> dict:
    """Max recall s.t. precision >= min_precision and hard-negative FPR <= max (else best F1)."""
    positive = labels != "none"
    rows = []
    for t in candidates:
        f = flagged(scored, t)
        tp, fp = int((f & positive).sum()), int((f & ~positive).sum())
        fn = int((~f & positive).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        hn_fpr = float((f & hard_neg).sum() / hard_neg.sum()) if hard_neg.any() else 0.0
        rows.append((t, precision, recall, f1, hn_fpr))
    feasible = [r for r in rows if r[1] >= tuning.get("min_precision", 0.85)
                and r[4] <= tuning.get("max_hard_negative_fpr", 0.10)]
    if feasible:
        best = max(feasible, key=lambda r: (r[2], r[1], -r[0]))
        return {"threshold": best[0], "constraints_met": True}
    best = max(rows, key=lambda r: (r[3], -r[0]))
    return {"threshold": best[0], "constraints_met": False}
