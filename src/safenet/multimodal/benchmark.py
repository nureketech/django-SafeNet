"""OCR quality on the synthetic screenshots (TZ, stage 6).

* key-token recovery: amounts and the words «каспи», «промокод», «төлем»/«выплата»
  found in the image text, and how many of them OCR recovers;
* risk levels of screenshots analysed from the image only: schemes should get at
  least ``medium``, hard negatives (receipts, free contests) should stay ``low``.
"""

from __future__ import annotations

import time

import pandas as pd

from safenet import paths
from safenet.multimodal import ocr as ocr_module
from safenet.preprocess import normalize

KEY_WORDS = ("каспи", "промокод", "толем", "выплат")


def key_tokens(text: str) -> set[str]:
    nt = normalize(text)
    tokens = {f"amount:{int(round(v))}" for v in nt.amounts() if v >= 100}
    tokens |= {f"word:{w}" for w in KEY_WORDS if w in nt.folded}
    return tokens


def token_recovery(expected_text: str, ocr_text: str) -> float | None:
    expected = key_tokens(expected_text)
    if not expected:
        return None
    return len(expected & key_tokens(ocr_text)) / len(expected)


def screenshots(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else pd.read_parquet(paths.DATASET_PATH)
    return df[df["source_type"] == "screenshot"].reset_index(drop=True)


def level_shares(levels: pd.Series, df: pd.DataFrame) -> dict:
    scheme = df["label"] != "none"
    hard_neg = df["is_hard_negative"]
    return {
        "scheme_at_least_medium": float((levels[scheme] != "low").mean()),
        "hard_negative_low": float((levels[hard_neg] == "low").mean()),
        "n_scheme": int(scheme.sum()),
        "n_hard_negative": int(hard_neg.sum()),
    }


def run_benchmark(df: pd.DataFrame | None = None, use_ml: bool = True) -> dict:
    """Real OCR of every screenshot; raises :class:`OCRUnavailable` when there is no engine."""
    from safenet.pipeline import analyze

    shots = screenshots(df)
    engine, reason = ocr_module.available_engine()
    if engine is None:
        raise ocr_module.OCRUnavailable(reason)
    rows = []
    start = time.perf_counter()
    for row in shots.itertuples():
        result = analyze({"id": row.id, "text": "", "image_path": row.image_path, "source_type": "screenshot"},
                         use_ml=use_ml)
        rows.append({
            "id": row.id,
            "recovery": token_recovery(row.text, result.ocr_text or ""),
            "confidence": result.ocr_confidence,
            "level": result.risk_level,
            "ocr_text": result.ocr_text or "",
        })
    elapsed = time.perf_counter() - start
    out = pd.DataFrame(rows)
    recovery = out["recovery"].dropna()
    return {
        "engine": engine,
        "n": len(shots),
        "mean_token_recovery": float(recovery.mean()) if len(recovery) else 0.0,
        "mean_confidence": float(out["confidence"].mean()),
        "seconds_per_image": elapsed / max(1, len(shots)),
        **level_shares(out["level"], shots),
        "details": out,
    }


def oracle_levels(df: pd.DataFrame | None = None, use_ml: bool = True) -> dict:
    """Levels when OCR is perfect (the image text itself): checks the scoring of screenshot texts."""
    from safenet.pipeline import analyze_many

    shots = screenshots(df)
    results = analyze_many({"id": r.id, "text": r.text, "source_type": "screenshot"} for r in shots.itertuples())
    return level_shares(pd.Series([r.risk_level for r in results]), shots)


def write_report() -> dict:
    lines = [
        "# OCR: скриншоттар бойынша есеп",
        "",
        "`python -m safenet.multimodal` арқылы синтетикалық скриншоттарда есептелген.",
        "",
    ]
    oracle = oracle_levels()
    try:
        bench = run_benchmark()
    except ocr_module.OCRUnavailable as exc:
        bench = None
        lines += [
            "## Нақты OCR",
            "",
            f"**Іске қосылмады — OCR қозғалтқышы жоқ.** {exc}",
            "",
        ]
    if bench:
        lines += [
            "## Нақты OCR",
            "",
            f"- Қозғалтқыш: **{bench['engine']}**; скриншоттар: {bench['n']}",
            f"- Негізгі токендердің қалпына келу үлесі (сомалар, «каспи», «промокод», «төлем/выплата»): "
            f"**{bench['mean_token_recovery']:.3f}** (талап ≥ 0.8)",
            f"- Орташа сенімділік: {bench['mean_confidence']:.3f}; бір суретке {bench['seconds_per_image']:.2f} с",
            f"- Схема скриншоттары: `medium`/`high` — **{bench['scheme_at_least_medium'] * 100:.1f}%** "
            f"(талап ≥ 75%, n = {bench['n_scheme']})",
            f"- Қиын негатив скриншоттары: `low` — **{bench['hard_negative_low'] * 100:.1f}%** "
            f"(талап ≥ 85%, n = {bench['n_hard_negative']})",
            "",
        ]
    lines += [
        "## Мінсіз OCR (суреттегі мәтіннің өзі)",
        "",
        "Скриншот мәтіндерінің скорингін OCR қателерінсіз тексеру үшін:",
        "",
        f"- Схема скриншоттары `medium`/`high`: **{oracle['scheme_at_least_medium'] * 100:.1f}%** "
        f"(n = {oracle['n_scheme']})",
        f"- Қиын негатив скриншоттары `low`: **{oracle['hard_negative_low'] * 100:.1f}%** "
        f"(n = {oracle['n_hard_negative']})",
        "",
        "## Алдын ала өңдеу",
        "",
        "Сұр түс → кіші суретті 2 есе үлкейту → қараңғы тақырыпты инверсиялау → медианалық сүзгі → "
        "Otsu бинаризациясы → тұтас түсті жолақтарды (тақырып) алып тастау → қисаюды түзету (±5°, жол "
        "профилінің өткірлігі бойынша) → ақ фонда қара мәтін.",
        "",
    ]
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "ocr_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"oracle": oracle, "real": {k: v for k, v in (bench or {}).items() if k != "details"}}
