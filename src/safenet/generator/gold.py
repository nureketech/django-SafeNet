"""Build ``data/gold/gold.parquet`` from the hand-written ``gold_source.yaml``.

The gold set is used ONLY for the final evaluation (stage 8): never for training,
never for choosing thresholds or weights.
"""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import yaml

from safenet import paths
from safenet.common import luhn_valid, unique

GOLD_SOURCE = paths.GOLD_DIR / "gold_source.yaml"

_CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)|\d{4}[ -]?(?:\d{2}\*\*|\*{4})[ -]?\*{4}[ -]?\d{4}")
_PHONE_RE = re.compile(r"(?:\+7|(?<!\d)8)[\s(]*7\d{2}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)")
_DOMAIN_RE = re.compile(r"\b[\w-]+\.(?:test|example)\b")
_LINK_RE = re.compile(r"(?:https?://)?(?:bit\.ly|clck\.ru|cutt\.ly|[\w-]+\.(?:test|example))/\S*")
_PROMO_RE = re.compile(r"(?:промокод|промик|промо|код|promokod)\s*:?\s*([A-Z][A-Z0-9-]{3,}|[A-Z0-9]*\d[A-Z0-9]*[A-Z][A-Z0-9]*)")

SCHEMA_FIELDS = (
    "text", "label", "lang", "source_type", "platform", "risk", "signals",
)


def simple_entities(text: str) -> dict[str, list[str]]:
    """Lightweight regex entity extraction (the full extractor lives in stage 2)."""
    return {
        "cards": unique(m.group(0) for m in _CARD_RE.finditer(text)),
        "phones": unique(m.group(0) for m in _PHONE_RE.finditer(text)),
        "domains": unique(m.group(0) for m in _DOMAIN_RE.finditer(text)),
        "promo_codes": unique(m.group(1) for m in _PROMO_RE.finditer(text)),
        "links": unique(m.group(0) for m in _LINK_RE.finditer(text)),
    }


def load_gold_source() -> list[dict]:
    items = yaml.safe_load(GOLD_SOURCE.read_text(encoding="utf-8"))
    for i, item in enumerate(items):
        missing = [f for f in SCHEMA_FIELDS if f not in item]
        if missing:
            raise ValueError(f"gold item {i}: missing {missing}")
    return items


def build_gold() -> pd.DataFrame:
    rows = []
    for i, item in enumerate(load_gold_source(), 1):
        entities = simple_entities(item["text"])
        for card in entities["cards"]:
            if "*" not in card and luhn_valid(card):
                raise ValueError(f"gold item {i}: card {card} passes Luhn — use a fictional number")
        rows.append({
            "id": f"gold-{i:03d}",
            "text": item["text"],
            "source_type": item["source_type"],
            "platform": item["platform"],
            "lang": item["lang"],
            "label": item["label"],
            "is_hard_negative": bool(item.get("hn", False)),
            "risk_gold": item["risk"],
            "gold_signals": list(item["signals"]),
            "obfuscations": [],
            "template_id": f"gold-{i:03d}",
            "template_fold": -1,
            "split": "gold",
            "subtype": "gold",
            "account_id": f"gold_acc_{i:03d}",
            "group_id": f"gold_grp_{i:03d}",
            "entities": entities,
            "image_path": None,
            "created_at": datetime(2026, 9, 1),
            "synthetic": True,
            "network_id": None,
            "handwritten": True,
        })
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def save_gold(df: pd.DataFrame) -> None:
    paths.GOLD_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.GOLD_PATH, index=False)
