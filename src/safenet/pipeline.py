"""Single entry point: ``analyze(item) -> AnalysisResult``.

    from safenet.pipeline import analyze
    result = analyze({"text": "Вложи 5000 — получи 10000 за 3 дня", "source_type": "post"})
    result = analyze({"text": "смотрите отзыв", "image_path": "data/images/syn-00001.png"})

Steps: normalise (stage 2) -> detect signals with evidence (stage 3) -> OCR of an
attached screenshot, analysed by the same detector with ``source: "ocr"``
(stage 6) -> ML track probabilities on the combined text (stage 4) -> optional
network bonus (stage 7) -> score and explain (stage 5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Iterable

from pydantic import BaseModel, Field

from safenet.ml.model import ModelNotTrained, load_model, predict_proba
from safenet.multimodal import ocr as ocr_module
from safenet.preprocess import NormalizedText, normalize
from safenet.scoring.config import ScoringConfig, load_config
from safenet.scoring.models import AnalysisResult, NetworkOut
from safenet.scoring.result import build_result
from safenet.scoring.score import compute_score
from safenet.signals import SignalHit, get_detector, merge_hits

SOURCE_TYPES = ("post", "chat_message", "stream_transcript", "screenshot", "account_bio")


class Item(BaseModel):
    """Input of the analysis (a post, message, transcript, bio and/or a screenshot)."""

    text: str = Field(default="", max_length=20000)
    id: str | None = None
    source_type: str | None = None
    platform: str | None = None
    account_id: str | None = None
    image_path: str | None = None


# (item, normalised text, signals, score before network) -> NetworkOut with ``contribution`` = bonus
NetworkLookup = Callable[[Item, NormalizedText, list[SignalHit], float], NetworkOut | None]


@dataclass
class _Prepared:
    item: Item
    image: object | None = None
    ocr: ocr_module.OCRResult | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def combined_text(self) -> str:
        parts = [self.item.text.strip()]
        if self.ocr and self.ocr.text.strip():
            parts.append(self.ocr.text.strip())
        return "\n".join(p for p in parts if p)


def _item_id(item: Item, ocr_text: str = "") -> str:
    digest = hashlib.sha1((item.text + "\n" + ocr_text).encode("utf-8")).hexdigest()[:12]
    return item.id or f"item-{digest}"


def _as_item(item: Item | dict | str) -> Item:
    if isinstance(item, Item):
        return item
    if isinstance(item, str):
        return Item(text=item)
    return Item.model_validate(item)


def _prepare(item: Item, image=None, use_ocr: bool = True) -> _Prepared:
    prepared = _Prepared(item=item, image=image if image is not None else item.image_path)
    if prepared.image is None or not use_ocr:
        return prepared
    try:
        prepared.ocr = ocr_module.ocr_image(prepared.image)
    except ocr_module.OCRUnavailable as exc:
        prepared.warnings.append(f"OCR unavailable: {exc}")
    except (OSError, ValueError) as exc:
        prepared.warnings.append(f"Image could not be read: {exc}")
    return prepared


def _ml_probabilities(texts: list[str], use_ml: bool) -> list[dict[str, float] | None]:
    if not use_ml or not texts:
        return [None] * len(texts)
    try:
        model = load_model()
    except ModelNotTrained:
        return [None] * len(texts)
    return predict_proba(texts, model).to_dict(orient="records")


def _analyze_one(p: _Prepared, proba: dict[str, float] | None, cfg: ScoringConfig,
                 network_lookup: NetworkLookup | None) -> AnalysisResult:
    item = p.item
    detector = get_detector()
    source = "transcript" if item.source_type == "stream_transcript" else "text"
    nt = normalize(item.text)
    hits = detector.detect(nt, source=source)
    nt_ocr = None
    if p.ocr and p.ocr.text.strip():
        nt_ocr = normalize(p.ocr.text)
        hits = merge_hits([hits, detector.detect(nt_ocr, source="ocr")])
    network = None
    bonus = 0.0
    if network_lookup is not None:
        pre = compute_score(hits, proba, 0.0, cfg)
        network = network_lookup(item, nt, hits, pre.score)
        bonus = network.contribution if network else 0.0
    breakdown = compute_score(hits, proba, bonus, cfg)
    result = build_result(
        item_id=_item_id(item, p.ocr.text if p.ocr else ""), text=item.text, nt=nt, breakdown=breakdown, cfg=cfg,
        network=network, source_type=item.source_type, platform=item.platform, account_id=item.account_id,
        extra_nt=nt_ocr,
    )
    if p.ocr is not None:
        result.ocr_text = p.ocr.text
        result.ocr_confidence = p.ocr.confidence
        result.ocr_engine = p.ocr.engine
    result.warnings = p.warnings
    return result


def analyze(item: Item | dict | str, *, image=None, use_ml: bool = True, use_ocr: bool = True,
            network_lookup: NetworkLookup | None = None, cfg: ScoringConfig | None = None) -> AnalysisResult:
    """Analyse one item (``image``: path, bytes or PIL image of an attached screenshot)."""
    prepared = _prepare(_as_item(item), image, use_ocr)
    cfg = cfg or load_config()
    proba = _ml_probabilities([prepared.combined_text], use_ml)[0]
    return _analyze_one(prepared, proba, cfg, network_lookup)


def analyze_many(items: Iterable[Item | dict | str], *, use_ml: bool = True, use_ocr: bool = True,
                 network_lookup: NetworkLookup | None = None, cfg: ScoringConfig | None = None
                 ) -> list[AnalysisResult]:
    """Analyse several items (one ML call for the whole batch)."""
    prepared = [_prepare(_as_item(i), None, use_ocr) for i in items]
    cfg = cfg or load_config()
    probas = _ml_probabilities([p.combined_text for p in prepared], use_ml)
    return [_analyze_one(p, proba, cfg, network_lookup) for p, proba in zip(prepared, probas)]
