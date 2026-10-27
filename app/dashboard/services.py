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
from safenet.common import dedup_key
from safenet.multimodal import ocr as ocr_module
from safenet.pipeline import analyze, analyze_many
from safenet.scoring.models import AnalysisResult

FEEDBACK_FIELDS = ["timestamp", "case_id", "verdict", "risk_score", "risk_level", "primary_track", "source_type",
                   "lang", "comment", "user"]


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


def save_case(result: AnalysisResult, *, origin: str, image_path: str = "", group_id: str = "",
              source_url: str = "", published_at=None) -> Case:
    """Insert or refresh a case; re-analysing an id keeps its verdict history linked and moves it up
    in the «newest first» order (``analysed_at`` is refreshed on every save)."""
    fresh = Case.from_result(result, image_path=image_path, origin=origin, group_id=group_id, source_url=source_url,
                             published_at=published_at)
    fields = {f.attname: getattr(fresh, f.attname) for f in Case._meta.concrete_fields
              if f.name not in ("id", "created_at", "analysed_at")}
    case, _ = Case.objects.update_or_create(pk=fresh.pk, defaults=fields)
    return case


def absorb_manual_copies(case: Case, visible_text: str) -> int:
    """Fold earlier hand-pasted checks of the same post (no account, no link) into ``case``.

    Without this, a post checked once by pasting its text and once by its link would look like two
    accounts sharing a promo code, i.e. a network of one post with itself. Verdicts move to ``case``.
    """
    key = dedup_key(visible_text)
    if not key:
        return 0
    copies = [c.pk for c in Case.objects.filter(origin="check", account_id="", source_url="").exclude(pk=case.pk)
              if dedup_key(c.text) == key]
    if copies:
        with transaction.atomic():
            Feedback.objects.filter(case_id__in=copies).update(case=case, case_ref=case.pk)
            Case.objects.filter(pk__in=copies).delete()
    return len(copies)


def _open_feedback_csv(path: Path):
    """Append handle; a file with an older header is moved aside so the columns always match."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != FEEDBACK_FIELDS:
            path.rename(path.with_name(f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}{path.suffix}"))
    new_file = not path.exists() or path.stat().st_size == 0
    return path.open("a", newline="", encoding="utf-8"), new_file


def record_feedback(case: Case, verdict: str, comment: str = "", user=None) -> Feedback:
    """Store the verdict (with the analyst) in the DB and append it to ``feedback.csv``."""
    username = user.get_username() if user is not None and user.is_authenticated else ""
    fb = Feedback.objects.create(case=case, case_ref=case.id, verdict=verdict, comment=comment[:2000],
                                 user=user if username else None, username=username)
    handle, new_file = _open_feedback_csv(feedback_path())
    with handle as f:
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
            "user": username,
        })
    return fb
