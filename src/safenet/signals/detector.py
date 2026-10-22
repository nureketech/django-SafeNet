"""Signal detector: which signals fired and where (evidence spans in the original text)."""

from __future__ import annotations

from dataclasses import dataclass, field

from safenet import paths
from safenet.preprocess import NormalizedText, normalize
from safenet.signals.computed import COMPUTED
from safenet.signals.config import Signal, load_signals

MAX_EVIDENCE = 3


@dataclass
class Evidence:
    """A fragment of the original text (or OCR text / transcript) supporting a signal."""

    text: str
    start: int
    end: int
    source: str = "text"


@dataclass
class SignalHit:
    id: str
    track: str
    name_ru: str
    name_kk: str
    weight: float
    core: bool
    negative: bool
    evidence: list[Evidence] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def _has_alnum(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _is_decoration(fragment: str) -> bool:
    """A run of the same emoji («🎰🎰») is decoration, not a word replaced by an emoji."""
    chars = [c for c in fragment if not c.isspace()]
    return len(chars) >= 2 and len(set(chars)) == 1


def _word_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Extend a span so that it does not cut a word in half (stems match word prefixes)."""
    while end < len(text) and text[end].isalpha() and end > start and text[end - 1].isalpha():
        end += 1
    while start > 0 and text[start - 1].isalpha() and start < end and text[start].isalpha():
        start -= 1
    return start, end


class SignalDetector:
    """Runs every signal of ``config/signals.yaml`` over a normalised text."""

    def __init__(self, config_path: str | None = None) -> None:
        self.signals: tuple[Signal, ...] = load_signals(str(config_path or paths.SIGNALS_CONFIG))
        self.by_id = {s.id: s for s in self.signals}

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _window_match(window, folded: str, start: int, end: int) -> bool:
        if window is None or window.regex is None:
            return False
        lo, hi = max(0, start - window.window), min(len(folded), end + window.window)
        return bool(window.regex.search(folded, lo, hi))

    @staticmethod
    def _near_amount(nt: NormalizedText, start: int, end: int, spec: dict) -> bool:
        window = int(spec.get("window", 30))
        min_value = float(spec.get("min_value", 0))
        for ent in nt.by_kind("amount"):
            if ent.value is None or ent.value < min_value or "nstart" not in ent.meta:
                continue
            gap = max(0, ent.meta["nstart"] - end, start - ent.meta["nend"])
            if gap <= window:
                return True
        return False

    def _spans_for(self, sig: Signal, nt: NormalizedText) -> tuple[list[tuple[int, int]], dict]:
        """Original-text spans supporting ``sig`` plus computed details."""
        folded = nt.folded
        spans: list[tuple[int, int]] = []
        details: dict = {}
        for regex, is_keyword in ((sig.keyword_regex, True), (sig.pattern_regex, False)):
            if regex is None:
                continue
            for m in regex.finditer(folded):
                s, e = m.start(), m.end()
                if e <= s:
                    continue
                if sig.exclude and self._window_match(sig.exclude, folded, s, e):
                    continue
                if sig.context and not self._window_match(sig.context, folded, s, e):
                    continue
                if is_keyword and sig.near_amount and not self._near_amount(nt, s, e, sig.near_amount):
                    continue
                os_, oe = _word_bounds(nt.original, *nt.to_original(s, e))
                fragment = nt.original[os_:oe]
                if not _has_alnum(fragment) and (not sig.emoji_ok or _is_decoration(fragment)):
                    continue
                spans.append((os_, oe))
        for kind in sig.entities:
            for ent in nt.by_kind(kind):
                spans.append((ent.start, ent.end))
        if sig.computed:
            computed_hits = COMPUTED[sig.computed](nt)
            for hit in computed_hits:
                spans.append(nt.to_original(hit.start, hit.end))
            if computed_hits:
                best = max(computed_hits, key=lambda h: h.details.get("monthly_pct", h.details.get("annual_pct", 0)))
                details = dict(best.details)
        return spans, details

    # ---------------------------------------------------------------- public
    def detect(self, text: str | NormalizedText, source: str = "text") -> list[SignalHit]:
        """Return fired signals (positive and negative) with evidence."""
        nt = text if isinstance(text, NormalizedText) else normalize(text)
        hits: list[SignalHit] = []
        for sig in self.signals:
            spans, details = self._spans_for(sig, nt)
            if not spans:
                continue
            spans.sort()
            merged: list[tuple[int, int]] = []
            for s, e in spans:
                if merged and s < merged[-1][1]:
                    if e > merged[-1][1]:
                        merged[-1] = (merged[-1][0], e)
                    continue
                merged.append((s, e))
            evidence = [Evidence(nt.original[s:e], s, e, source) for s, e in merged[:MAX_EVIDENCE]]
            hits.append(SignalHit(
                id=sig.id, track=sig.track, name_ru=sig.name_ru, name_kk=sig.name_kk, weight=sig.weight,
                core=sig.core, negative=sig.negative, evidence=evidence, details=details,
            ))
        return hits


_DEFAULT: SignalDetector | None = None


def get_detector() -> SignalDetector:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = SignalDetector()
    return _DEFAULT


def detect(text: str | NormalizedText, source: str = "text") -> list[SignalHit]:
    return get_detector().detect(text, source)


def fired_ids(text: str | NormalizedText) -> set[str]:
    return {h.id for h in detect(text)}


def merge_hits(groups: list[list[SignalHit]]) -> list[SignalHit]:
    """Merge hits found in several sources (post text, OCR, transcript).

    A signal counts once; its evidence keeps every source, so the analyst sees
    where each fragment came from.
    """
    merged: dict[str, SignalHit] = {}
    for hits in groups:
        for hit in hits:
            if hit.id not in merged:
                merged[hit.id] = SignalHit(
                    id=hit.id, track=hit.track, name_ru=hit.name_ru, name_kk=hit.name_kk, weight=hit.weight,
                    core=hit.core, negative=hit.negative, evidence=list(hit.evidence), details=dict(hit.details),
                )
                continue
            target = merged[hit.id]
            target.evidence.extend(hit.evidence)
            if hit.details.get("monthly_pct", 0) > target.details.get("monthly_pct", 0):
                target.details = dict(hit.details)
    return list(merged.values())
