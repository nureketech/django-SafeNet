"""``python -m safenet.scoring.tune`` — choose risk-level thresholds on the **val** split.

* ``medium`` (= "signs found, needs review"): maximise recall subject to
  precision >= 0.85 and FPR on hard negatives <= 10% (TZ, stage 5);
* ``high``: above ``no_core_cap`` (no core signal -> never high) and with
  precision of "high" >= 0.95; among those, best agreement with ``risk_gold``.

Writes ``models/thresholds.json`` and ``reports/thresholds.md``. Test and gold are
never used here.
"""

from __future__ import annotations

import dataclasses
import json
from statistics import median

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from safenet import paths
from safenet.ml.model import load_model, predict_proba
from safenet.scoring.config import ScoringConfig, load_config, reload_config
from safenet.scoring.score import compute_score
from safenet.signals import get_detector


def score_split(df: pd.DataFrame, cfg: ScoringConfig) -> pd.DataFrame:
    """Raw scores (before the evidence cap, which depends on the threshold) for every record."""
    raw_cfg = dataclasses.replace(cfg, require_evidence_for_medium=False)
    probas = predict_proba(df["text"].tolist(), load_model()).to_dict(orient="records")
    detector = get_detector()
    rows = []
    for text, proba in zip(df["text"], probas):
        b = compute_score(detector.detect(text), proba, 0.0, raw_cfg)
        rows.append({"score": round(b.score), "has_evidence": bool(b.positive), "primary_track": b.primary_track})
    out = pd.DataFrame(rows, index=df.index)
    return pd.concat([df[["id", "label", "is_hard_negative", "risk_gold"]], out], axis=1)


def binary_metrics(scored: pd.DataFrame, t: float) -> dict:
    flagged = (scored["score"] >= t) & scored["has_evidence"]
    positive = scored["label"] != "none"
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    neg = ~positive
    hn = scored["is_hard_negative"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": t, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": float((flagged & neg).sum() / neg.sum()) if neg.any() else 0.0,
        "hard_negative_fpr": float((flagged & hn).sum() / hn.sum()) if hn.any() else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
    }


def levels(scored: pd.DataFrame, medium: float, high: float) -> pd.Series:
    s = scored["score"].where(scored["has_evidence"] | (scored["score"] < medium), medium - 1)
    return pd.Series(np.where(s >= high, "high", np.where(s >= medium, "medium", "low")), index=scored.index)


def choose_thresholds(scored: pd.DataFrame, cfg: ScoringConfig) -> tuple[dict, list[dict], list[dict]]:
    tun = cfg.tuning
    lo, hi = tun.get("medium_range", [10, 64])
    medium_rows = [binary_metrics(scored, t) for t in range(int(lo), int(hi) + 1)]
    feasible = [r for r in medium_rows if r["precision"] >= tun.get("min_precision", 0.85)
                and r["hard_negative_fpr"] <= tun.get("max_hard_negative_fpr", 0.10)]
    if feasible:
        best_recall = max(r["recall"] for r in feasible)
        tied = [r for r in feasible if r["recall"] == best_recall]
        best_precision = max(r["precision"] for r in tied)
        tied = [r["threshold"] for r in tied if r["precision"] == best_precision]
        medium = float(int(median(tied)))
        medium_feasible = True
    else:
        medium, medium_feasible = cfg.medium, False

    h_lo, h_hi = tun.get("high_range", [46, 95])
    high_rows = []
    positive = scored["label"] != "none"
    for t in range(max(int(h_lo), int(medium) + 1), int(h_hi) + 1):
        lv = levels(scored, medium, t)
        is_high = lv == "high"
        high_precision = float((is_high & positive).sum() / is_high.sum()) if is_high.any() else 1.0
        macro = f1_score(scored["risk_gold"], lv, labels=["low", "medium", "high"], average="macro")
        high_rows.append({"threshold": t, "high_precision": high_precision, "level_macro_f1": float(macro),
                          "n_high": int(is_high.sum())})
    feasible_h = [r for r in high_rows if r["high_precision"] >= tun.get("min_high_precision", 0.95)]
    if feasible_h:
        best = max(r["level_macro_f1"] for r in feasible_h)
        high = float(int(median([r["threshold"] for r in feasible_h if r["level_macro_f1"] == best])))
    else:
        high = max(cfg.high, medium + 1)
    chosen = {"medium": medium, "high": high, "medium_constraints_met": medium_feasible,
              "high_constraints_met": bool(feasible_h)}
    return chosen, medium_rows, high_rows


def write_outputs(scored: pd.DataFrame, chosen: dict, medium_rows: list[dict], high_rows: list[dict],
                  cfg: ScoringConfig) -> dict:
    at = binary_metrics(scored, chosen["medium"])
    default = binary_metrics(scored, cfg.medium if cfg.thresholds_source == "default" else 35)
    lv = levels(scored, chosen["medium"], chosen["high"])
    confusion = pd.crosstab(scored["risk_gold"], lv).reindex(index=["low", "medium", "high"],
                                                             columns=["low", "medium", "high"], fill_value=0)
    flagged_pos = (lv != "low") & (scored["label"] != "none")
    track_acc = float((scored.loc[flagged_pos, "primary_track"] == scored.loc[flagged_pos, "label"]).mean())
    data = {**chosen, "source": "val", "val": at, "track_accuracy_on_flagged": track_acc}
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths.THRESHOLDS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    lines = [
        "# Тәуекел деңгейлерінің шектері",
        "",
        "`python -m safenet.scoring.tune` (`make train` ішінде) арқылы **val** жинағында таңдалған. "
        "test пен gold қолданылмайды.",
        "",
        f"- **medium** шегі = **{int(chosen['medium'])}** (талап: precision ≥ "
        f"{cfg.tuning.get('min_precision')}, қиын негативтердегі FPR ≤ {cfg.tuning.get('max_hard_negative_fpr')}; "
        f"осы шартта recall барынша жоғары). Шарт орындалды: {chosen['medium_constraints_met']}.",
        f"- **high** шегі = **{int(chosen['high'])}** (core-белгісіз шек {cfg.no_core_cap:.0f}-тен жоғары, "
        f"«high» precision ≥ {cfg.tuning.get('min_high_precision')}, risk_gold-пен келісім барынша жоғары). "
        f"Шарт орындалды: {chosen['high_constraints_met']}.",
        "",
        "## val метрикалары (таңдалған шектер)",
        "",
        "| көрсеткіш | мәні |",
        "|---|---:|",
        f"| precision (белгілер бар/жоқ) | {at['precision']:.3f} |",
        f"| recall | {at['recall']:.3f} |",
        f"| F1 | {at['f1']:.3f} |",
        f"| FPR (барлық `none`) | {pct(at['fpr'])} |",
        f"| FPR (қиын негативтер) | {pct(at['hard_negative_fpr'])} |",
        f"| медиум/жоғары деңгейлі схемалардағы трек дәлдігі | {pct(track_acc)} |",
        "",
        f"Әдепкі шекпен (35) салыстыру: precision {default['precision']:.3f}, recall {default['recall']:.3f}, "
        f"қиын негативтердегі FPR {pct(default['hard_negative_fpr'])}.",
        "",
        "## Деңгейлер: эталон (`risk_gold`, жол) × болжам (баған)",
        "",
        "| | low | medium | high |",
        "|---|---:|---:|---:|",
        *[f"| {idx} | " + " | ".join(str(v) for v in row) + " |" for idx, row in confusion.iterrows()],
        "",
        "## medium шегінің кандидаттары (әр 5 ұпай)",
        "",
        "| шек | precision | recall | F1 | FPR (қиын негатив) |",
        "|---:|---:|---:|---:|---:|",
        *[f"| {r['threshold']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {pct(r['hard_negative_fpr'])} |"
          for r in medium_rows if r["threshold"] % 5 == 0 or r["threshold"] == chosen["medium"]],
        "",
        "## high шегінің кандидаттары (әр 5 ұпай)",
        "",
        "| шек | high precision | деңгейлер macro-F1 | high саны |",
        "|---:|---:|---:|---:|",
        *[f"| {r['threshold']} | {r['high_precision']:.3f} | {r['level_macro_f1']:.3f} | {r['n_high']} |"
          for r in high_rows if r["threshold"] % 5 == 0 or r["threshold"] == chosen["high"]],
        "",
    ]
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "thresholds.md").write_text("\n".join(lines), encoding="utf-8")
    return data


def main() -> None:
    cfg = load_config(use_tuned=False)
    val = pd.read_parquet(paths.split_path("val"))
    scored = score_split(val, cfg)
    chosen, medium_rows, high_rows = choose_thresholds(scored, cfg)
    data = write_outputs(scored, chosen, medium_rows, high_rows, cfg)
    reload_config()
    v = data["val"]
    print(f"thresholds: medium={int(data['medium'])}, high={int(data['high'])}; val precision={v['precision']:.3f}, "
          f"recall={v['recall']:.3f}, hard-negative FPR={v['hard_negative_fpr']:.3f}")


if __name__ == "__main__":
    main()
