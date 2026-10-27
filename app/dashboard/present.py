"""Turn an ``AnalysisResult`` into what the templates show: highlighted text, bars, labels."""

from __future__ import annotations

from dataclasses import dataclass

from django.utils.html import escape
from django.utils.safestring import SafeString, mark_safe

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
    }


# ------------------------------------------------------------------ links in plain words
def _short(label: str, limit: int = 60) -> str:
    return label if len(label) <= limit else label[:limit - 1] + "…"


def _ru_with(n: int) -> str:
    return "аккаунтом" if n % 10 == 1 and n % 100 != 11 else "аккаунтами"


def network_summary(cluster, lang: str, t: dict) -> dict:
    """Name, counts and the shared things of a network, written for a reader without graph terms."""
    from app.dashboard.i18n import SHARED_KINDS
    from app.dashboard.network import network_name

    high = round(cluster.risk_share * len(cluster.record_ids))
    return {
        "id": cluster.cluster_id,
        "name": network_name(cluster),
        "track": cluster.top_track,
        "stats": t["story_stats"].format(accounts=len(cluster.accounts), posts=len(cluster.record_ids), high=high),
        "reasons": [
            (SHARED_KINDS[lang].get(e.kind, e.kind), _short(e.label), t["story_in_accounts"].format(n=e.n_accounts))
            for e in cluster.shared[:4]
        ],
    }


def link_story(links, lang: str, t: dict) -> dict:
    """«Байланыстар» of a case card as sentences: who it is linked with and why, or why nothing was found."""
    from app.dashboard.i18n import FOUND_KINDS
    from app.dashboard.network import account_label

    if links.cluster is not None:
        others = [account_label(a) for a in links.other_accounts]
        names = ", ".join(others[:5]) + ("…" if len(others) > 5 else "")
        return {
            "linked": True,
            "headline": t["story_linked"].format(n=len(others), names=names, acc=_ru_with(len(others))),
            "network": network_summary(links.cluster, lang, t),
        }
    found = [e for e in links.own if e.kind != "group"]
    if found:
        items = ", ".join(f"{FOUND_KINDS[lang].get(e.kind, e.kind)} {_short(e.label, 40)}" for e in found[:4])
        reason = t["story_not_shared"].format(items=items)
    elif links.account_id:
        reason = t["story_no_entities"]
    else:
        reason = t["story_nothing"]
    return {"linked": False, "reason": reason, "hint": not links.account_id}
