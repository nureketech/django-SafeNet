"""Turn an ``AnalysisResult`` into what the templates show: highlighted text, bars, labels."""

from __future__ import annotations

from dataclasses import dataclass

from django.utils.html import escape
from django.utils.safestring import SafeString, mark_safe

from app.dashboard.i18n import ENTITY_KINDS
from safenet.scoring.models import AnalysisResult


@dataclass
class Span:
    start: int
    end: int
    css: str
    title: str


def highlight(text: str, spans: list[Span]) -> SafeString:
    """HTML of ``text`` with non-overlapping evidence spans wrapped in ``<mark>``."""
    out: list[str] = []
    pos = 0
    for span in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if span.start < pos or span.end <= span.start or span.end > len(text):
            continue
        out.append(escape(text[pos:span.start]))
        out.append(f'<mark class="ev {span.css}" title="{escape(span.title)}">{escape(text[span.start:span.end])}</mark>')
        pos = span.end
    out.append(escape(text[pos:]))
    return mark_safe("".join(out).replace("\n", "<br>"))


def evidence_spans(result: AnalysisResult, source: str, lang: str) -> list[Span]:
    spans = []
    for sig in result.signals:
        name = sig.name_kk if lang == "kk" else sig.name_ru
        for ev in sig.evidence:
            if ev.source == source or (source == "text" and ev.source == "transcript"):
                spans.append(Span(ev.start, ev.end, f"ev-{sig.track}", name))
    for sig in result.negative_signals:
        name = sig.name_kk if lang == "kk" else sig.name_ru
        for ev in sig.evidence:
            if ev.source == source or (source == "text" and ev.source == "transcript"):
                spans.append(Span(ev.start, ev.end, "ev-negative", name))
    return spans


@dataclass
class Bar:
    label: str
    value: float
    css: str
    width: float = 0.0


def contribution_bars(result: AnalysisResult, lang: str, t: dict) -> list[Bar]:
    """Every component of the score (they add up to ``risk_score``)."""
    kk = lang == "kk"
    bars = [Bar(s.name_kk if kk else s.name_ru, s.contribution, f"bar-{s.track}") for s in result.signals]
    bars += [Bar(f"{t['combo_part']}: {c.name_kk if kk else c.name_ru}", c.contribution, f"bar-{c.track}")
             for c in result.combos]
    if result.ml.contribution:
        bars.append(Bar(f"{t['ml_part']} ({t['probability']} {result.ml.probability * 100:.0f}%)",
                        result.ml.contribution, "bar-ml"))
    if result.network.contribution:
        bars.append(Bar(f"{t['network_part']} ({result.network.cluster_id})", result.network.contribution,
                        "bar-network"))
    bars += [Bar(s.name_kk if kk else s.name_ru, s.contribution, "bar-negative") for s in result.negative_signals]
    bars += [Bar(a.name_kk if kk else a.name_ru, a.contribution, "bar-adjustment") for a in result.adjustments]
    peak = max((abs(b.value) for b in bars), default=1.0) or 1.0
    for b in bars:
        b.width = round(abs(b.value) / peak * 100, 1)
    return bars


def card_context(result: AnalysisResult, lang: str, t: dict) -> dict:
    """Everything the shared result partial needs."""
    return {
        "r": result,
        "explanation": result.explanation_kk if lang == "kk" else result.explanation_ru,
        "highlighted_text": highlight(result.text, evidence_spans(result, "text", lang)),
        "highlighted_ocr": highlight(result.ocr_text, evidence_spans(result, "ocr", lang)) if result.ocr_text else "",
        "bars": contribution_bars(result, lang, t),
        "level_label": t[f"level_{result.risk_level}"],
        "level_hint": t[f"hint_{result.risk_level}"],
        "tracks_present": sorted({s.track for s in result.signals}),
        "network_entities": [_split_entity(e, lang) for e in result.network.shared_entities],
    }


def _split_entity(entry: str, lang: str) -> tuple[str, str]:
    """``"card: 4400 43** **** 1234"`` -> ``("карта", "4400 43** **** 1234")``."""
    kind, _, label = entry.partition(": ")
    return (ENTITY_KINDS[lang].get(kind, kind), label) if label else ("", entry)
