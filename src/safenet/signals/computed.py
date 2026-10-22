"""Computed signals: return is calculated from amounts and periods, not just keywords.

``unreal_return`` fires when the promised return, converted to a monthly rate, is
at least 30% (TZ, stage 3): «вложи 5000 получи 10000 за 3 дня» -> +100% in 3 days.
``bank_rate`` fires for annual rates <= 20% («депозит 15% годовых»).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from safenet.preprocess import NormalizedText
from safenet.preprocess.entities import Entity

MONTHLY_THRESHOLD = 30.0  # % per month
BANK_RATE_MAX = 20.0      # % per year

_L = r"[а-яa-z]"
INVEST_RE = re.compile(
    rf"(?<!{_L})(?:влож\w*|вклад\w*|вкладыва\w*|инвест\w*|закин\w*|положи\w*|пополни\w*|вход\w*|старт\w*|тариф\w*"
    rf"|сал(?:сан|дым|ып|ыныз|ындар|а|ган|гандар)?|салым\w*|салу|кируге|кіру|кириниз)(?!{_L})"
)
RETURN_RE = re.compile(
    rf"(?:получ\w*|забира\w*|забер\w*|забрать|вывед\w*|вывел\w*|вывод\w*|будет|станет|превратят\w*|делаем|сделаем"
    rf"|возвраща\w*|вернем|начисл\w*|(?<!{_L})ал(?:асын|асыз|дым|ыныз)?(?!{_L})|шыгардым|шыкты|шыгады|болады"
    rf"|айналады|тусед\w*|табыс\w*|пайда\w*|доход\w*|прибыл\w*|→|->|=>|=)"
)
ARROW_RE = re.compile(r"→|->|=>|—>|>>")
# Verbs that turn an amount into a bigger one; enough without an explicit «вложи».
STRONG_RETURN_RE = re.compile(r"делаем|сделаем|превратят\w*|станет|станут|айналады|жасаймыз|превращ\w*")
# A payout announced before the amounts («Выплата получена! +20 000 ... Вложено: 13 000»).
PAYOUT_BEFORE_RE = re.compile(r"выплат\w*|получен\w*|алынды|вывод\w*|\+\s*$")
INVESTED_LABEL_RE = re.compile(r"(?:вложено|салынды|вклад|инвестиция|сумма вклада)\s*:?\s*$")
# Words right after the returned amount that belong to the evidence («... 10000 за 3 дня», «... 60 мың шықты»).
TAIL_RE = re.compile(r"\s*(?:(?:за|через)\s+\d+\s*[а-я]+|ал(?:асын|асыз)?|шыкты|шыгардым|тусед[а-я]*)(?![а-я])")
PERIODIC_RE = [
    (re.compile(r"в\s?день|в\s?сутки|ежедневно|каждый\s?день|кунине|кун\s?сайын|таулигине|ар\s?кун"), 1),
    (re.compile(r"в\s?недел\w*|еженедельно|каждую\s?неделю|аптасына|апта\s?сайын"), 7),
    (re.compile(r"в\s?месяц|ежемесячно|каждый\s?месяц|айына|ай\s?сайын"), 30),
]
DURATION_RE = re.compile(
    r"(?:за|через|спустя|на)?\s*(?P<n>\d+)\s*(?P<u>дн\w*|день|сут\w*|недел\w*|месяц\w*|час\w*|кун\w*|апта\w*|ай\w*)"
    r"|(?P<week>за\s+неделю|через\s+неделю|бир\s+аптада|аптада|бир\s+апта|неделю)"
    r"|(?P<month>за\s+месяц|через\s+месяц|бир\s+айда|айда)"
)
BONUS_CTX = re.compile(r"бонус|скидк|кешбэк|женилдик|к\s+депозит|на\s+(?:первый\s+)?депозит|фриспин|распродаж|кэшбек")
CASINO_CTX = re.compile(r"слот|казино|казик|ставк|бонус|фриспин|рулетк|занос|играю|выигр|авиатор|на красное")
RETURN_CTX = re.compile(r"доход|прибыл|доходност|табыс|пайда|начисл|получа|выплат|тусед|платит|профит")


@dataclass
class Hit:
    start: int
    end: int
    details: dict = field(default_factory=dict)


def _amounts(nt: NormalizedText) -> list[Entity]:
    out = [e for e in nt.by_kind("amount") if e.value and e.value >= 100 and "nstart" in e.meta]
    return sorted(out, key=lambda e: e.meta["nstart"])


def _duration_days(text: str) -> int | None:
    m = DURATION_RE.search(text)
    if not m:
        return None
    if m.group("week"):
        return 7
    if m.group("month"):
        return 30
    n = int(m.group("n"))
    unit = m.group("u")
    if unit.startswith(("час",)):
        return max(1, round(n / 24))
    if unit.startswith(("недел", "апта")):
        return n * 7
    if unit.startswith(("месяц", "ай")):
        return n * 30
    return n


def _periodic(text: str) -> int | None:
    for rx, days in PERIODIC_RE:
        if rx.search(text):
            return days
    return None


def _monthly(pct: float, days: int | None) -> float:
    if not days:
        return pct
    return pct * 30.0 / days


def unreal_return(nt: NormalizedText) -> list[Hit]:
    """Find promised returns and compute their monthly rate."""
    f = nt.folded
    hits: list[Hit] = []
    amounts = _amounts(nt)

    # 1) invested amount -> returned amount
    for i, a in enumerate(amounts):
        a_s, a_e = a.meta["nstart"], a.meta["nend"]
        for b in amounts[i + 1:]:
            b_s, b_e = b.meta["nstart"], b.meta["nend"]
            if b_s - a_e > 70:
                break
            between = f[a_e:b_s]
            arrow = bool(ARROW_RE.search(between)) and len(between.strip()) <= 14
            invest_m = INVEST_RE.search(f[max(0, a_s - 35):a_e + 15])
            return_ok = bool(RETURN_RE.search(between[-45:])) or bool(RETURN_RE.search(f[b_e:b_e + 22]))
            strong = bool(STRONG_RETURN_RE.search(between)) and len(between) <= 40
            reversed_pair = (
                bool(INVESTED_LABEL_RE.search(f[max(a_e, b_s - 25):b_s]))
                and bool(PAYOUT_BEFORE_RE.search(f[max(0, a_s - 30):a_s]))
            )
            if reversed_pair:
                a, b = b, a  # «+20 000 ... Вложено: 13 000» — the first amount is the payout
            elif not (arrow or strong or (invest_m and return_ok)):
                continue
            around_b = f[max(a_e, b_s - 25):b_s] + " " + f[b_e:b_e + 25]
            periodic = _periodic(around_b)
            if periodic and b.value < a.value:
                pct = b.value / a.value * 100
                days = periodic
            elif b.value > a.value:
                pct = (b.value / a.value - 1) * 100
                days = _duration_days(between) or _duration_days(f[max(a_e, b_e):max(a_e, b_e) + 40])
            else:
                continue
            monthly = _monthly(pct, days)
            if monthly >= MONTHLY_THRESHOLD:
                if reversed_pair:
                    a_s, b_e = b.meta["nstart"], a.meta["nend"]
                start = a_s
                if invest_m and not arrow:
                    start = min(a_s, max(0, a_s - 35) + invest_m.start())
                tail = TAIL_RE.match(f, b_e)
                end = tail.end() if tail else b_e
                hits.append(Hit(start, end, {
                    "kind": "amounts", "invest": a.value, "return": b.value,
                    "return_pct": round(pct, 1), "period_days": days, "monthly_pct": round(monthly, 1),
                }))
            break

    # 2) percent per period
    for p in nt.by_kind("percent"):
        if "nstart" not in p.meta:
            continue
        s, e = p.meta["nstart"], p.meta["nend"]
        ctx = f[max(0, s - 30):e + 30]
        if BONUS_CTX.search(ctx):
            continue
        days = p.meta.get("period_days")
        if days == 365:
            continue
        if days is None:
            if p.value < MONTHLY_THRESHOLD or not RETURN_CTX.search(f[max(0, s - 40):e + 40]):
                continue
        monthly = _monthly(p.value, days)
        if monthly >= MONTHLY_THRESHOLD:
            hits.append(Hit(s, e, {"kind": "percent", "return_pct": p.value, "period_days": days,
                                   "monthly_pct": round(monthly, 1)}))

    # 3) multipliers (х2 за неделю)
    for m in nt.by_kind("multiplier"):
        if "nstart" not in m.meta or m.value < 1.5:
            continue
        s, e = m.meta["nstart"], m.meta["nend"]
        if CASINO_CTX.search(f[max(0, s - 40):e + 40]):
            continue
        days = m.meta.get("period_days") or _duration_days(f[e:e + 25])
        ctx = f[max(0, s - 50):e + 50]
        if not (INVEST_RE.search(ctx) or RETURN_CTX.search(ctx) or days):
            continue
        pct = (m.value - 1) * 100
        monthly = _monthly(pct, days)
        if monthly >= MONTHLY_THRESHOLD:
            hits.append(Hit(s, e, {"kind": "multiplier", "multiplier": m.value, "return_pct": round(pct, 1),
                                   "period_days": days, "monthly_pct": round(monthly, 1)}))
    return hits


def bank_rate(nt: NormalizedText) -> list[Hit]:
    """Annual percents that look like ordinary bank rates (<= 20% per year)."""
    hits = []
    for p in nt.by_kind("percent"):
        if p.meta.get("period_days") == 365 and p.value is not None and p.value <= BANK_RATE_MAX and "nstart" in p.meta:
            hits.append(Hit(p.meta["nstart"], p.meta["nend"], {"annual_pct": p.value}))
    return hits


COMPUTED = {"unreal_return": unreal_return, "bank_rate": bank_rate}


def max_monthly_return(hits: list[Hit]) -> float:
    return max((h.details.get("monthly_pct", 0.0) for h in hits), default=0.0)
