"""Load and compile ``config/signals.yaml``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from safenet import paths
from safenet.preprocess import fold
from safenet.preprocess.tables import FOLD

TRACKS = {"common", "lottery", "casino", "pyramid", "negative"}
REQUIRED = ("id", "track", "name_ru", "name_kk", "weight")
_LETTER = r"[а-яa-z0-9]"


def keyword_regex(keyword: str) -> str:
    """Regex for one keyword: every word is a stem (prefix), short words match exactly."""
    words = fold(keyword).split()
    parts = []
    for word in words:
        esc = re.escape(word)
        parts.append(esc if len(word) <= 2 or not word[-1].isalnum() else esc + _LETTER + "*")
    body = r"[\s\-]+".join(parts)
    lead = rf"(?<!{_LETTER})" if words and words[0][0].isalnum() else ""
    return lead + body


def fold_pattern(pattern: str) -> str:
    """Fold Cyrillic letters inside a regex without touching its syntax."""
    return pattern.translate(FOLD)


@dataclass
class Window:
    window: int
    regex: re.Pattern | None


@dataclass
class Signal:
    """One compiled signal definition."""

    id: str
    track: str
    name_ru: str
    name_kk: str
    weight: float
    core: bool
    description: str
    negative: bool
    keyword_regex: re.Pattern | None
    pattern_regex: re.Pattern | None
    entities: tuple[str, ...]
    computed: str | None
    near_amount: dict | None
    context: Window | None
    exclude: Window | None
    emoji_ok: bool
    raw: dict = field(repr=False, default_factory=dict)


def _window(spec: dict | None) -> Window | None:
    if not spec:
        return None
    pats = [fold_pattern(p) for p in spec.get("patterns", [])]
    return Window(int(spec.get("window", 30)), re.compile("|".join(f"(?:{p})" for p in pats), re.M) if pats else None)


def _alternation(parts: list[str]) -> re.Pattern | None:
    return re.compile("|".join(f"(?:{p})" for p in parts), re.M) if parts else None


def compile_signal(raw: dict) -> Signal:
    keywords = sorted((keyword_regex(k) for k in raw.get("keywords_ru", []) + raw.get("keywords_kk", [])),
                      key=len, reverse=True)
    patterns = [fold_pattern(p) for p in raw.get("patterns", [])]
    return Signal(
        id=raw["id"],
        track=raw["track"],
        name_ru=raw["name_ru"],
        name_kk=raw["name_kk"],
        weight=float(raw["weight"]),
        core=bool(raw.get("core", False)),
        description=raw.get("description", ""),
        negative=raw["track"] == "negative",
        keyword_regex=_alternation(keywords),
        pattern_regex=_alternation(patterns),
        entities=tuple(raw.get("entities", [])),
        computed=raw.get("computed"),
        near_amount=raw.get("near_amount"),
        context=_window(raw.get("context")),
        exclude=_window(raw.get("exclude")),
        emoji_ok=bool(raw.get("emoji_ok", False)),
        raw=raw,
    )


def load_raw(path: Path = paths.SIGNALS_CONFIG) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data["signals"]


def validate(raw_signals: list[dict]) -> None:
    """Raise ``ValueError`` when the dictionary is malformed."""
    ids = [s.get("id") for s in raw_signals]
    if len(ids) != len(set(ids)):
        raise ValueError("signal ids must be unique")
    for s in raw_signals:
        missing = [f for f in REQUIRED if f not in s]
        if missing:
            raise ValueError(f"{s.get('id')}: missing {missing}")
        if s["track"] not in TRACKS:
            raise ValueError(f"{s['id']}: unknown track {s['track']}")
        if not (s.get("keywords_ru") or s.get("keywords_kk") or s.get("patterns") or s.get("entities")
                or s.get("computed")):
            raise ValueError(f"{s['id']}: no way to fire")


@lru_cache(maxsize=4)
def load_signals(path: str = str(paths.SIGNALS_CONFIG)) -> tuple[Signal, ...]:
    raw = load_raw(Path(path))
    validate(raw)
    return tuple(compile_signal(s) for s in raw)
