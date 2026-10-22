"""Single entry point: ``analyze(item) -> AnalysisResult``.

    from safenet.pipeline import analyze
    result = analyze({"text": "Вложи 5000 — получи 10000 за 3 дня", "source_type": "post"})

Steps: normalise (stage 2) -> detect signals with evidence (stage 3) -> ML track
probabilities (stage 4) -> optional network bonus (stage 7) -> score and explain
(stage 5).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Iterable

from pydantic import BaseModel, Field

from safenet.ml.model import ModelNotTrained, load_model, predict_proba
from safenet.preprocess import NormalizedText, normalize
from safenet.scoring.config import ScoringConfig, load_config
from safenet.scoring.models import AnalysisResult, NetworkOut
from safenet.scoring.result import build_result
from safenet.scoring.score import compute_score
from safenet.signals import SignalHit, get_detector

SOURCE_TYPES = ("post", "chat_message", "stream_transcript", "screenshot", "account_bio")


class Item(BaseModel):
    """Input of the analysis (a post, message, transcript, bio or OCR text)."""

    text: str = Field(default="", max_length=20000)
    id: str | None = None
    source_type: str | None = None
    platform: str | None = None
    account_id: str | None = None


# (item, normalised text, signals, score before network) -> NetworkOut with ``contribution`` = bonus
NetworkLookup = Callable[[Item, NormalizedText, list[SignalHit], float], NetworkOut | None]


def _item_id(item: Item) -> str:
    return item.id or "item-" + hashlib.sha1(item.text.encode("utf-8")).hexdigest()[:12]


def _as_item(item: Item | dict | str) -> Item:
    if isinstance(item, Item):
        return item
    if isinstance(item, str):
        return Item(text=item)
    return Item.model_validate(item)


def _ml_probabilities(texts: list[str], use_ml: bool) -> list[dict[str, float] | None]:
    if not use_ml or not texts:
        return [None] * len(texts)
    try:
        model = load_model()
    except ModelNotTrained:
        return [None] * len(texts)
    return predict_proba(texts, model).to_dict(orient="records")


def _analyze_one(item: Item, proba: dict[str, float] | None, cfg: ScoringConfig,
                 network_lookup: NetworkLookup | None) -> AnalysisResult:
    source = "transcript" if item.source_type == "stream_transcript" else "text"
    nt = normalize(item.text)
    hits = get_detector().detect(nt, source=source)
    network = None
    bonus = 0.0
    if network_lookup is not None:
        pre = compute_score(hits, proba, 0.0, cfg)
        network = network_lookup(item, nt, hits, pre.score)
        bonus = network.contribution if network else 0.0
    breakdown = compute_score(hits, proba, bonus, cfg)
    return build_result(
        item_id=_item_id(item), text=item.text, nt=nt, breakdown=breakdown, cfg=cfg, network=network,
        source_type=item.source_type, platform=item.platform, account_id=item.account_id,
    )


def analyze(item: Item | dict | str, *, use_ml: bool = True, network_lookup: NetworkLookup | None = None,
            cfg: ScoringConfig | None = None) -> AnalysisResult:
    """Analyse one item; see the module docstring."""
    item = _as_item(item)
    cfg = cfg or load_config()
    proba = _ml_probabilities([item.text], use_ml)[0]
    return _analyze_one(item, proba, cfg, network_lookup)


def analyze_many(items: Iterable[Item | dict | str], *, use_ml: bool = True,
                 network_lookup: NetworkLookup | None = None, cfg: ScoringConfig | None = None
                 ) -> list[AnalysisResult]:
    """Analyse several items (one ML call for the whole batch)."""
    items = [_as_item(i) for i in items]
    cfg = cfg or load_config()
    probas = _ml_probabilities([i.text for i in items], use_ml)
    return [_analyze_one(i, p, cfg, network_lookup) for i, p in zip(items, probas)]
