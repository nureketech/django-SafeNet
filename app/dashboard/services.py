"""Loading cases into the queue and recording analyst feedback."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.db import transaction

from app.dashboard.models import Case, Feedback
from safenet import paths
from safenet.multimodal import ocr as ocr_module
from safenet.pipeline import analyze, analyze_many

FEEDBACK_FIELDS = ["timestamp", "case_id", "verdict", "risk_score", "risk_level", "primary_track", "source_type",
                   "lang", "comment"]


def feedback_path() -> Path:
    return Path(getattr(settings, "SAFENET_FEEDBACK_PATH", paths.FEEDBACK_PATH))


def _dataset(split: str) -> pd.DataFrame:
    if split == "all":
        return pd.read_parquet(paths.DATASET_PATH)
    if split == "gold":
        return pd.read_parquet(paths.GOLD_PATH)
    return pd.read_parquet(paths.split_path(split))


def load_dataset_cases(split: str = "test", limit: int | None = None, replace: bool = False) -> int:
    """Analyse records of a split and store them as cases (screenshots are read by OCR when possible)."""
    df = _dataset(split)
    if limit:
        df = df.head(limit)
    ocr_ok = ocr_module.available_engine()[0] is not None
    texts = df[df["source_type"] != "screenshot"]
    shots = df[df["source_type"] == "screenshot"]
    results = analyze_many(
        {"id": r.id, "text": r.text, "source_type": r.source_type, "platform": r.platform,
         "account_id": r.account_id} for r in texts.itertuples()
    )
    cases = [Case.from_result(r) for r in results]
    for r in shots.itertuples():
        image_path = r.image_path if isinstance(r.image_path, str) else ""
        item = {"id": r.id, "text": "", "source_type": "screenshot", "platform": r.platform,
                "account_id": r.account_id}
        if ocr_ok and image_path:
            result = analyze(item, image=paths.ROOT / image_path)
        else:  # no OCR engine: use the image text itself as perfect OCR
            result = analyze({**item, "text": r.text})
        cases.append(Case.from_result(result, image_path=image_path))
    with transaction.atomic():
        if replace:
            Case.objects.filter(origin="dataset").delete()
        Case.objects.bulk_create(cases, ignore_conflicts=True)
    return len(cases)


def record_feedback(case: Case, verdict: str, comment: str = "") -> Feedback:
    """Store the verdict in the DB and append it to ``feedback.csv``."""
    fb = Feedback.objects.create(case=case, case_ref=case.id, verdict=verdict, comment=comment[:2000])
    path = feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "case_id": case.id,
            "verdict": verdict,
            "risk_score": case.risk_score,
            "risk_level": case.risk_level,
            "primary_track": case.primary_track,
            "source_type": case.source_type,
            "lang": case.lang,
            "comment": comment.replace("\n", " ")[:2000],
        })
    return fb
