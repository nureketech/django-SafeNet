"""Metric helpers (binary "signs present / absent", per track, groups, explanations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from safenet import LABELS, TRACKS
from safenet.eval.collect import Record


def binary(labels: np.ndarray, hard_neg: np.ndarray, flags: np.ndarray) -> dict:
    positive = labels != "none"
    negative = ~positive
    tp = int((flags & positive).sum())
    fp = int((flags & negative).sum())
    fn = int((~flags & positive).sum())
    tn = int((~flags & negative).sum())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "hard_negative_fpr": float((flags & hard_neg).sum() / hard_neg.sum()) if hard_neg.any() else 0.0,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def predicted_class(flags: np.ndarray, tracks: np.ndarray) -> np.ndarray:
    return np.where(flags, tracks, "none").astype(object)


def per_track(labels: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for track in TRACKS:
        y, p = labels == track, pred == track
        tp = int((y & p).sum())
        precision = tp / p.sum() if p.sum() else 0.0
        recall = tp / y.sum() if y.sum() else 0.0
        out[track] = {
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
            "support": int(y.sum()),
        }
    out["macro_f1"] = float(np.mean([out[t]["f1"] for t in TRACKS]))
    return out


def confusion(labels: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    return pd.crosstab(pd.Series(labels, name="эталон"), pd.Series(pred, name="болжам")).reindex(
        index=list(LABELS), columns=list(LABELS), fill_value=0)


def explanation_quality(records: list[Record], flags: np.ndarray) -> dict:
    """Among correctly flagged schemes: share whose fired signals overlap ``gold_signals``."""
    overlaps, jaccards = [], []
    for r, f in zip(records, flags):
        if not f or r.label == "none" or not r.gold_signals:
            continue
        fired = {h.id for h in r.hits if not h.negative}
        if r.ocr_hits:
            fired |= {h.id for h in r.ocr_hits if not h.negative}
        gold = {s for s in r.gold_signals if not s.startswith("N_")}
        overlaps.append(bool(fired & gold))
        union = fired | gold
        jaccards.append(len(fired & gold) / len(union) if union else 0.0)
    return {
        "n": len(overlaps),
        "share_with_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
    }


def by_group(records: list[Record], flags: np.ndarray, key) -> pd.DataFrame:
    rows = []
    groups = sorted({key(r) for r in records})
    for g in groups:
        idx = np.array([key(r) == g for r in records])
        labels = np.array([r.label for r in records])[idx]
        hard = np.array([r.is_hard_negative for r in records])[idx]
        m = binary(labels, hard, flags[idx])
        rows.append({"group": g, **{k: m[k] for k in ("n", "precision", "recall", "f1", "fpr", "hard_negative_fpr")}})
    return pd.DataFrame(rows)


def level_agreement(records: list[Record], scores: np.ndarray, has_evidence: np.ndarray,
                    medium: float, high: float) -> dict:
    s = np.round(scores)
    s = np.where(has_evidence | (s < medium), s, medium - 1)
    pred = np.where(s >= high, "high", np.where(s >= medium, "medium", "low"))
    gold = np.array([r.risk_gold for r in records])
    return {
        "accuracy": float((pred == gold).mean()),
        "macro_f1": float(f1_score(gold, pred, labels=["low", "medium", "high"], average="macro")),
        "confusion": pd.crosstab(pd.Series(gold, name="эталон"), pd.Series(pred, name="болжам")).reindex(
            index=["low", "medium", "high"], columns=["low", "medium", "high"], fill_value=0),
    }
