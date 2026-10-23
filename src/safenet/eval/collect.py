"""Collect everything the variants need for one split: signals, ML probabilities, OCR, graph cluster."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from safenet.common import to_list
from safenet.graph.build import Cluster, GraphModel
from safenet.graph.entities import graph_entities
from safenet.ml.model import load_model, predict_proba
from safenet.multimodal import ocr as ocr_module
from safenet.preprocess import normalize
from safenet.signals import SignalHit, get_detector


@dataclass
class Record:
    id: str
    text: str
    label: str
    lang: str
    source_type: str
    is_hard_negative: bool
    risk_gold: str
    gold_signals: list[str]
    obfuscated: bool
    hits: list[SignalHit]
    proba: dict[str, float]
    cluster: Cluster | None = None
    ocr_hits: list[SignalHit] | None = None
    ocr_proba: dict[str, float] | None = None
    ocr_text: str | None = None
    extra: dict = field(default_factory=dict)


def collect(df: pd.DataFrame, graph: GraphModel | None = None, use_ocr: bool = False) -> list[Record]:
    """Detect signals and predict probabilities once per record (and per OCR text)."""
    detector = get_detector()
    model = load_model()
    probas = predict_proba(df["text"].tolist(), model).to_dict(orient="records")
    records: list[Record] = []
    for row, proba in zip(df.itertuples(index=False), probas):
        source = "transcript" if row.source_type == "stream_transcript" else "text"
        nt = normalize(row.text)
        cluster = None
        if graph is not None:
            cluster = graph.cluster_of(getattr(row, "account_id", None), [e.node_id for e in graph_entities(nt)])
        records.append(Record(
            id=row.id, text=row.text, label=row.label, lang=row.lang, source_type=row.source_type,
            is_hard_negative=bool(row.is_hard_negative), risk_gold=row.risk_gold,
            gold_signals=[str(s) for s in to_list(row.gold_signals)],
            obfuscated=bool(len(to_list(getattr(row, "obfuscations", [])))),
            hits=detector.detect(nt, source=source), proba=proba, cluster=cluster,
        ))

    if use_ocr and ocr_module.available_engine()[0] is not None:
        shots = [(i, r) for i, r in enumerate(df.itertuples(index=False))
                 if r.source_type == "screenshot" and isinstance(getattr(r, "image_path", None), str)]
        texts = []
        for i, row in shots:
            text = ocr_module.ocr_image(row.image_path).text
            records[i].ocr_text = text
            records[i].ocr_hits = detector.detect(normalize(text), source="ocr")
            texts.append(text)
        if texts:
            for (i, _), proba in zip(shots, predict_proba(texts, model).to_dict(orient="records")):
                records[i].ocr_proba = proba
    return records
