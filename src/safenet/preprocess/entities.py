"""Entity extraction.

*Raw* entities (links, domains, shorteners, referral parameters, cards, phones,
promo codes) are found in the original text, because case and punctuation matter
for them. *Normalised* entities (amounts, percents with periods, multipliers,
Kaspi mentions) are found in the folded normalised text, so «5к», «5 мың теңге»
and «vlozhi 5k» are all understood. All spans are in original-text coordinates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from safenet.preprocess import tables as T

if TYPE_CHECKING:
    from safenet.preprocess.normalize import NormalizedText


@dataclass
class Entity:
    """One extracted entity; ``start``/``end`` refer to the original text."""

    kind: str
    text: str
    start: int
    end: int
    value: float | None = None
    meta: dict = field(default_factory=dict)


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < e and s < end for s, e in spans)


# ------------------------------------------------------------- raw entities
_TLD = "|".join(sorted(T.TLDS, key=len, reverse=True))
URL_RE = re.compile(
    rf"(?i)(?<![\w@.])(?P<scheme>https?://)?(?:www\.)?"
    rf"(?P<host>(?:[a-z0-9](?:[a-z0-9-]{{0,61}}[a-z0-9])?\.)+(?:{_TLD}))"
    rf"(?::\d{{2,5}})?(?P<path>/[^\s<>\"«»()\[\]{{}}]*)?(?![\w-])"
)
REF_RE = re.compile(r"(?i)[?&](?P<name>ref|aff|affid|partner|pid|clickid|sub\d*|utm_[a-z]+)=(?P<val>[^&#\s]+)")
REF_PATH_RE = re.compile(r"(?i)/(?:r|ref|invite)/[\w-]+")
CARD_RE = re.compile(r"(?<![\d*])([\d*]{4})([ -]?)([\d*]{4})\2([\d*]{4})\2(\d{4})(?![\d*])")
PHONE_RE = re.compile(r"(?<![\d*+])(?:\+7|8)[ \t(-]*7\d{2}[ \t)-]*[\d*]{3}[ \t-]*[\d*]{2}[ \t-]*\d{2}(?![\d*])")
_CODE = r"[A-Za-z0-9][A-Za-z0-9_-]{2,24}"
PROMO_BEFORE_RE = re.compile(
    rf"(?i:промо[\s-]?код\w*|промик\w*|промо|бонус[\s-]?код\w*|код\w{{0,4}}|promo[\s-]?code|"
    rf"promokod\w*|promik\w*|promo|kod\w{{0,4}}|code|используй\w*|вводи\w*|введи\w*|enter|"
    rf"енгіз\w*|енгиз\w*|қолдан\w*)\s*[:\-=]?\s*(?P<code>{_CODE})"
)
PROMO_AFTER_RE = re.compile(
    rf"(?P<code>{_CODE})\s+(?i:промокод\w*|коды\w*|кодын\w*|кодымен|promokod\w*|kody\w*|kodyn\w*)"
)
PROMO_STANDALONE_RE = re.compile(r"(?<![\w@./-])([A-Z][A-Z0-9-]{4,24})(?![\w-])")


def _valid_code(code: str, standalone: bool = False) -> bool:
    if code.upper() != code:
        return False
    letters = sum(c.isalpha() for c in code)
    digits = sum(c.isdigit() for c in code)
    if standalone:
        return letters >= 3 and digits >= 1
    return letters >= 1 and (digits >= 1 or letters >= 4)


def extract_raw_entities(text: str) -> list[Entity]:
    """Links, domains, shorteners, referral params, cards, phones and promo codes."""
    out: list[Entity] = []
    url_spans: list[tuple[int, int]] = []
    for m in URL_RE.finditer(text):
        start, end = m.start(), m.end()
        full = m.group(0)
        while full and full[-1] in ".,!?;:'\"":
            full = full[:-1]
            end -= 1
        host = m.group("host").lower()
        url_spans.append((start, end))
        hs = m.start("host")
        out.append(Entity("domain", text[hs:m.end("host")], hs, m.end("host"), meta={"host": host}))
        if m.group("scheme") or m.group("path"):
            out.append(Entity("url", full, start, end, meta={"host": host}))
        if host in T.SHORTENERS:
            out.append(Entity("shortener", full, start, end, meta={"host": host}))
        for r in REF_PATH_RE.finditer(full):
            out.append(Entity("ref_param", r.group(0), start + r.start(), start + r.end(), meta={"name": "path"}))
    for m in REF_RE.finditer(text):
        out.append(Entity("ref_param", m.group(0), m.start(), m.end(), meta={"name": m.group("name").lower()}))

    taken: list[tuple[int, int]] = list(url_spans)
    for m in CARD_RE.finditer(text):
        raw = m.group(0)
        digits = sum(c.isdigit() for c in raw)
        if digits < 4 or (digits < 16 and "*" not in raw):
            continue
        out.append(Entity("card", raw, m.start(), m.end(), meta={"masked": "*" in raw}))
        taken.append((m.start(), m.end()))
    for m in PHONE_RE.finditer(text):
        if _overlaps(m.start(), m.end(), taken):
            continue
        out.append(Entity("phone", m.group(0), m.start(), m.end(), meta={"masked": "*" in m.group(0)}))
        taken.append((m.start(), m.end()))

    promo_spans: list[tuple[int, int]] = []

    def add_promo(code: str, start: int, end: int, standalone: bool = False) -> None:
        if not _valid_code(code, standalone) or _overlaps(start, end, taken + promo_spans):
            return
        out.append(Entity("promo_code", code, start, end))
        promo_spans.append((start, end))

    for m in PROMO_BEFORE_RE.finditer(text):
        add_promo(m.group("code"), m.start("code"), m.end("code"))
    for m in PROMO_AFTER_RE.finditer(text):
        add_promo(m.group("code"), m.start("code"), m.end("code"))
    for m in PROMO_STANDALONE_RE.finditer(text):
        add_promo(m.group(1), m.start(1), m.end(1), standalone=True)
    return out


# ------------------------------------------------------ normalised entities
_NUM = r"(?<![\w.,:/])(\d{1,3}(?:[  ]\d{3})+|\d+(?:[.,]\d+)?)(?![.,]?\d)"
_MULT = r"(?:\s?(кк|к|k|тыс(?:\.|яч[аиу]?)?|тыщ[иау]?|мын|млн|миллион(?:ов|а)?|лям(?:ов|а)?)(?![а-яa-z]))?"
_CUR = r"(?:\s?(тг|тнг|тенге|₸|kzt|руб(?:\.|лей|ля)?|\$|usd|долл\w*)(?![а-яa-z]))?"
AMOUNT_RE = re.compile(_NUM + _MULT + _CUR + r"(?!\s?(?:%|процент|пайыз|:\d))")
_COUNT_WORDS = "|".join(sorted((w for w in T.NUMBER_WORDS if w not in ("мын", "тысяча", "тысячу")), key=len, reverse=True))
WORD_AMOUNT_RE = re.compile(
    r"(?<![а-я])(" + _COUNT_WORDS + r")\s+"
    r"(мын|тысяч\w*|тг|тенге)(?![а-я])"
)
PERCENT_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)?)\s?(%|процент\w*|пайыз\w*)")
WORD_PERCENT_RE = re.compile(
    r"(?<![а-я])(" + "|".join(sorted(T.NUMBER_WORDS, key=len, reverse=True)) + r")\s+(процент\w*|пайыз\w*)"
)
MULT_RE = re.compile(
    r"(?<![\w\d])[хx×]\s?(\d{1,3}(?:[.,]\d)?)(?![\d\w])"
    r"|(?<![\w\d.,])(\d{1,3})\s?[хx×](?![\w\d])"
    r"|(?<![а-я])в\s(\d{1,2})\s?раз[а]?(?![а-я])"
    r"|(?<![а-я])(\d{1,2}|еки|уш)\s(?:есе|еселе\w*)"
)
KASPI_RE = re.compile(r"(?<![а-яa-z])(?:каспи(?!й)\w*|kaspi\w*)")

PERIODS = [
    (r"в\s?день|в\s?сутки|ежедневно|за\s?день|в\s?час|кунине|кун\s?сайын|таулигине|кундик|ар\s?кун", 1),
    (r"в\s?недел\w|еженедельно|за\s?недел\w|аптасына|апта\s?сайын|аптада", 7),
    (r"в\s?месяц|ежемесячно|за\s?месяц|айына|ай\s?сайын|айда", 30),
    (r"годовых|годовой|в\s?год|за\s?год|ежегодно|жылдык|жылына|жыл\s?сайын", 365),
]
_PERIOD_RES = [(re.compile(rf"(?<![а-яa-z])(?:{p})"), days) for p, days in PERIODS]
MULTIPLIERS = {
    "к": 1e3, "k": 1e3, "кк": 1e6, "мын": 1e3, "млн": 1e6, "миллион": 1e6, "миллиона": 1e6,
    "миллионов": 1e6, "лям": 1e6, "ляма": 1e6, "лямов": 1e6,
}


def _num(raw: str) -> float:
    raw = raw.replace(" ", "").replace(" ", "")
    if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
        raw = raw.replace(",", "")
    return float(raw.replace(",", "."))


def _mult(word: str | None) -> float:
    if not word:
        return 1.0
    word = word.rstrip(".")
    if word.startswith(("тыс", "тыщ")):
        return 1e3
    return MULTIPLIERS.get(word, 1.0)


def find_period(folded: str, start: int, end: int, window: int = 32) -> int | None:
    """Days of the period phrase nearest after (preferred) or before ``[start, end)``."""
    after = folded[end:end + window]
    best: tuple[int, int] | None = None
    for rx, days in _PERIOD_RES:
        m = rx.search(after)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), days)
    if best:
        return best[1]
    before = folded[max(0, start - window):start]
    for rx, days in _PERIOD_RES:
        matches = list(rx.finditer(before))
        if matches and (best is None or matches[-1].end() > best[0]):
            best = (matches[-1].end(), days)
    return best[1] if best else None


def extract_normalized_entities(nt: "NormalizedText", raw: list[Entity]) -> list[Entity]:
    """Amounts, percents (with period), multipliers and Kaspi mentions."""
    folded = nt.folded
    blocked = [(e.start, e.end) for e in raw if e.kind in ("card", "phone", "url", "domain", "promo_code")]
    out: list[Entity] = []

    def add(kind: str, s: int, e: int, value: float | None, **meta) -> None:
        os_, oe = nt.to_original(s, e)
        if kind in ("amount", "percent", "multiplier") and _overlaps(os_, oe, blocked):
            return
        out.append(Entity(kind, nt.original[os_:oe], os_, oe, value, {"norm": nt.text[s:e], **meta}))

    for m in AMOUNT_RE.finditer(folded):
        value = _num(m.group(1)) * _mult(m.group(2))
        add("amount", m.start(), m.end(), value, currency=bool(m.group(3)), multiplier=bool(m.group(2)))
    numeric = [(e.start, e.end) for e in out]
    for m in WORD_AMOUNT_RE.finditer(folded):
        if _overlaps(*nt.to_original(m.start(), m.end()), numeric):
            continue
        unit = m.group(2)
        value = T.NUMBER_WORDS[m.group(1)] * (1000 if unit.startswith(("мын", "тысяч")) else 1)
        add("amount", m.start(), m.end(), float(value), currency=not unit.startswith(("мын", "тысяч")), words=True)
    for rx, is_word in ((PERCENT_RE, False), (WORD_PERCENT_RE, True)):
        for m in rx.finditer(folded):
            value = float(T.NUMBER_WORDS[m.group(1)]) if is_word else _num(m.group(1))
            add("percent", m.start(), m.end(), value, period_days=find_period(folded, m.start(), m.end()))
    for m in MULT_RE.finditer(folded):
        raw_value = next(g for g in m.groups() if g)
        value = {"еки": 2.0, "уш": 3.0}.get(raw_value) or _num(raw_value)
        if value < 1.2:
            continue
        add("multiplier", m.start(), m.end(), value, period_days=find_period(folded, m.start(), m.end()))
    for m in KASPI_RE.finditer(folded):
        add("kaspi", m.start(), m.end(), None)
    return out
