"""Small helpers shared by several stages."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Any, Iterable

import numpy as np

# Words that must never appear in system output (TZ, section 0).
FORBIDDEN_WORDS: tuple[str, ...] = (
    "незаконно",
    "мошенник",
    "преступник",
    "виновен",
    "заңсыз",
    "алаяқ",
    "қылмыскер",
    "кінәлі",
)
_FORBIDDEN_RE = re.compile("|".join(re.escape(w) for w in FORBIDDEN_WORDS), re.IGNORECASE)


def find_forbidden(text: str) -> list[str]:
    """Return the forbidden words (case-insensitive substrings) found in ``text``."""
    return [m.group(0) for m in _FORBIDDEN_RE.finditer(text or "")]


def luhn_valid(number: str) -> bool:
    """Return True when the digit string passes the Luhn checksum."""
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 12:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def digits_only(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit() or ch == "*")


def mask_card(value: str) -> str:
    """Mask a card number as ``4400 43** **** 1234`` (keeps BIN prefix and last four)."""
    raw = digits_only(value)
    if len(raw) < 12:
        return "**** " + raw[-4:]
    return f"{raw[:4]} {raw[4:6]}** **** {raw[-4:]}"


def mask_phone(value: str) -> str:
    """Mask a phone number as ``+7 701 *** ** 78``."""
    raw = digits_only(value)
    if len(raw) < 10:
        return "***"
    return f"+7 {raw[-10:-7]} *** ** {raw[-2:]}"


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_DUP_RE = re.compile(r"[^\w]+", re.UNICODE)


def dedup_key(text: str) -> str:
    """Aggressive normalisation used to detect near-duplicate texts."""
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    return _DUP_RE.sub(" ", text).strip()


def to_list(value: Any) -> list:
    """Convert parquet list/array values (numpy arrays, None) into plain lists."""
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return [value]


def to_entities(value: Any) -> dict[str, list[str]]:
    """Convert a parquet ``entities`` struct back into ``dict[str, list[str]]``."""
    if not value:
        return {}
    return {k: [str(x) for x in to_list(v)] for k, v in dict(value).items()}


def unique(items: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen)
