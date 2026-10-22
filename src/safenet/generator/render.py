"""Render one synthetic record from a template."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from safenet.common import unique
from safenet.generator import fillers as F
from safenet.generator.obfuscate import LINK_HIDING, TEXT_OBFUSCATIONS, to_translit
from safenet.generator.screens import screen_spec
from safenet.generator.templates import SLOT_RE, Template, resolve_block

PH_OPEN, PH_CLOSE = "", ""
_PH_RE = re.compile(f"{PH_OPEN}(\\d+){PH_CLOSE}")

# slot name -> entities key
ENTITY_SLOTS = {
    "phone": "phones",
    "card": "cards",
    "promo": "promo_codes",
    "ref_link": "links",
    "short_link": "links",
    "domain": "domains",
    "shop_domain": "domains",
}
ENTITY_KEYS = ("cards", "phones", "domains", "promo_codes", "links")
TEAMS = ["Barys Esports", "Nomad Five", "Tulpar Team", "Steppe Wolves", "Altyn Squad", "Kok Bori"]
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍←-⇿■-◿]"
)


# Neutral additions used only when a record would duplicate an existing text.
VARIATIONS = {
    "ru": ["— {nick}", "{nick}", "пишу из города {city}", "({time})", "upd {time}", "{date}", "#{city}", "{nick}, {time}"],
    "kk": ["— {nick}", "{nick}", "{city} қаласынан жазамын", "({time})", "upd {time}", "{date}", "#{city}", "{nick}, {time}"],
}
STREAM_VARIATIONS = {
    "ru": ["спасибо {nick} за подписку", "{nick} привет", "привет {nick} из города {city}"],
    "kk": ["{nick} жазылғаның үшін рахмет", "{nick} сәлем", "{city} қаласынан {nick} сәлем"],
}


@dataclass
class Rendered:
    text: str
    gold_signals: list[str]
    obfuscations: list[str]
    entities: dict[str, list[str]]
    slot_values: dict[str, str] = field(default_factory=dict)


class SlotContext:
    """Consistent slot values for one record (numbers are shared across languages)."""

    def __init__(self, rng: random.Random, forced: dict[str, str] | None = None, mask_ok: bool = True):
        self.rng = rng
        self.forced = dict(forced or {})
        self.mask_ok = mask_ok
        self.entity_values: list[tuple[str, str]] = []
        self._entity_cache: dict[str, int] = {}
        self._text_cache: dict[tuple[str, str], str] = {}
        r = rng
        total = r.choice([50, 80, 100, 150, 200, 300, 500])
        sold = r.randint(int(total * 0.4), int(total * 0.95))
        invest = r.randrange(5000, 200001, 1000)
        mult = r.choice([2, 2, 2, 3, 3, 4, 5])
        self.num = {
            "amount": r.randrange(300, 5001, 100),
            "win": r.randrange(50_000, 2_500_001, 1000),
            "invest": invest,
            "ret": invest * mult if r.random() < 0.6 else int(invest * r.uniform(1.4, 3.5)) // 1000 * 1000,
            "mult": mult,
            "days": r.randint(2, 14),
            "pct_day": r.randint(2, 15),
            "pct_week": r.randrange(15, 61, 5),
            "pct_month": r.randrange(35, 301, 5),
            "pct_year": r.randint(10, 17),
            "total": total,
            "sold": sold,
            "left": total - sold,
            "n": r.randint(2, 60),
            "k": r.randint(2, 9),
            "fs": r.choice([50, 100, 150, 200, 250, 300]),
            "price": r.randrange(1500, 60001, 500),
            "amount_dep": r.choice([10_000, 50_000, 100_000, 500_000, 1_000_000]),
            "prize_pool": r.choice([100_000, 200_000, 300_000, 500_000, 1_000_000]),
            "income": r.choice([300_000, 500_000, 700_000, 1_000_000]),
            "salary": r.randrange(150_000, 400_001, 10_000),
            "temp": r.randint(10, 35),
        }
        self.num["n2"] = max(1, self.num["n"] - r.randint(1, 5))
        self.num["sum"] = sold * self.num["amount"]
        self.casino = r.choice(F.CASINOS)
        self.pick = {
            "nick": F.nick(r), "nick2": F.nick(r), "nick3": F.nick(r),
            "casino": self.casino, "project": r.choice(F.PROJECTS), "bank": r.choice(F.BANKS),
            "dep_name": r.choice(F.DEPOSIT_NAMES), "shop": r.choice(F.SHOPS), "game": r.choice(F.GAMES),
            "team": r.choice(TEAMS), "district": r.choice(F.DISTRICTS), "time": F.fmt_time(r),
        }
        self.idx = {key: r.randrange(1000) for key in ("prize", "prize_small", "item", "city", "day", "occasion")}
        self.date_day, self.date_month = r.randint(1, 28), r.randrange(12)

    # -- entities -----------------------------------------------------------
    def _new_entity(self, slot: str) -> str:
        r = self.rng
        if slot in self.forced:
            return self.forced[slot]
        if slot == "phone":
            return F.masked_phone(r) if self.mask_ok and r.random() < 0.2 else F.phone(r)
        if slot == "card":
            return F.masked_card(r) if self.mask_ok and r.random() < 0.35 else F.card(r)
        if slot == "promo":
            return F.promo(r)
        if slot == "domain":
            return F.casino_domain(r, self.casino) if r.random() < 0.7 else F.domain(r)
        if slot == "ref_link":
            dom = self.forced.get("ref_domain") or F.casino_domain(r, self.casino)
            return F.ref_link(r, dom)
        if slot == "short_link":
            return F.short_link(r)
        if slot == "shop_domain":
            return F.shop_domain(r)
        raise KeyError(slot)

    def entity(self, slot: str) -> str:
        if slot not in self._entity_cache:
            self._entity_cache[slot] = len(self.entity_values)
            self.entity_values.append((slot, self._new_entity(slot)))
        return f"{PH_OPEN}{self._entity_cache[slot]}{PH_CLOSE}"

    # -- text slots ---------------------------------------------------------
    def value(self, slot: str, lang: str) -> str:
        r = self.rng
        if slot in ENTITY_SLOTS:
            return self.entity(slot)
        if slot in ("amount", "win", "invest", "ret", "sum", "price", "amount_dep", "prize_pool", "income", "salary"):
            key = (slot, lang)
            if key not in self._text_cache:
                self._text_cache[key] = F.fmt_amount(self.num[slot], lang, r)
            return self._text_cache[key]
        if slot in self.num:
            return str(self.num[slot])
        if slot in self.pick:
            return self.pick[slot]
        if slot == "date":
            return f"{self.date_day} {F.MONTHS[lang][self.date_month]}"
        if slot == "prize":
            return F.PRIZES[lang][self.idx["prize"] % len(F.PRIZES[lang])]
        if slot == "prize_small":
            return F.SMALL_PRIZES[lang][self.idx["prize_small"] % len(F.SMALL_PRIZES[lang])]
        if slot == "item":
            return F.ITEMS[lang][self.idx["item"] % len(F.ITEMS[lang])]
        if slot == "city":
            return F.CITIES[lang][self.idx["city"] % len(F.CITIES[lang])]
        if slot == "day":
            return F.DAYS[lang][self.idx["day"] % 7]
        if slot == "occasion":
            return F.OCCASIONS[lang][self.idx["occasion"] % len(F.OCCASIONS[lang])]
        raise KeyError(f"unknown slot {slot!r}")

    def fill(self, text: str, lang: str) -> str:
        return SLOT_RE.sub(lambda m: self.value(m.group(1), lang), text)


# ------------------------------------------------------------------ layouts
def _punct(part: str, mark: str = ".") -> str:
    part = part.rstrip()
    if part and (part[-1].isalnum() or part[-1] in "»)" or part[-1] == PH_CLOSE):
        return part + mark
    return part


def _cap(part: str) -> str:
    return part[:1].upper() + part[1:] if part else part


def render_layout(layout: str, parts: list[str], deco: dict, rng: random.Random) -> str:
    """Join block texts according to the template layout."""
    if layout == "lines":
        bullet = deco["bullet"]
        return "\n".join([parts[0]] + [bullet + p for p in parts[1:]])
    if layout == "paragraph":
        sep = deco["separator"].strip()
        if sep == ",":
            return _punct(", ".join([parts[0]] + [p[:1].lower() + p[1:] for p in parts[1:]]))
        return " ".join(_cap(_punct(p, sep)) for p in parts)
    if layout == "emoji_lines":
        em = deco["emoji"]
        return "\n".join(f"{em[i % len(em)]} {p}" for i, p in enumerate(parts))
    if layout == "numbered":
        head = parts[0] if parts[0].endswith("?") else parts[0].rstrip(":!.") + ":"
        return "\n".join([head] + [f"{i}) {p}" for i, p in enumerate(parts[1:], 1)])
    if layout == "caps_lines":
        return "\n".join([parts[0].upper()] + parts[1:])
    if layout == "chat":
        sep = rng.choice([", ", " ", ". "])
        text = ""
        for i, p in enumerate(parts):
            if i:
                text += " " if text.endswith((":", "?")) else sep
            text += p.rstrip(".!") if not p.endswith("?") else p
        return text + rng.choice(["", ")", "!", "))", " 🙏", ""])
    if layout == "chat_lower":
        return " ".join(p.rstrip(".!") for p in parts).lower() + rng.choice(["", ")", "", "!!"])
    if layout == "stream":
        text = " ".join(parts).lower()
        text = _EMOJI_RE.sub(" ", text)
        text = re.sub(r"[«»\"“”!?.,:;()]", " ", text)
        return re.sub(r"\s+", " ", text).strip()
    if layout == "bio_pipe":
        return " | ".join(parts)
    if layout == "bio_lines":
        return "\n".join(parts)
    if layout == "bio_dot":
        return " • ".join(parts)
    if layout == "screen":
        return "\n".join(parts)
    raise ValueError(layout)


# ------------------------------------------------------------------ records
def render_record(
    template: Template,
    lang: str,
    rng: random.Random,
    forced: dict[str, str] | None = None,
    obfuscate_prob: float = 0.0,
    allow_link_hiding: bool = True,
    variation: int = 0,
) -> Rendered:
    """Render ``template`` in ``lang`` (``ru``/``kk``/``mixed``/``translit``)."""
    base = template.base_lang
    ctx = SlotContext(rng, forced)
    signals: list[str] = []
    obfuscations: list[str] = []

    if template.source_type == "screenshot":
        spec = screen_spec(template, base)
        parts = [ctx.fill(line, base) for line in spec["text"].split("\n")]
        signals.extend(spec["signals"])
    else:
        blocks = list(template.blocks)
        langs = [base] * len(blocks)
        if lang == "mixed":
            switch = [i for i in range(len(blocks)) if rng.random() < 0.45]
            if not switch:
                switch = [rng.randrange(len(blocks))]
            if len(switch) == len(blocks) and len(blocks) > 1:
                switch = switch[1:]
            for i in switch:
                langs[i] = "ru"
        parts = []
        for (block, rank), block_lang in zip(blocks, langs):
            text, sigs = resolve_block(template, block, rank, block_lang)
            parts.append(ctx.fill(text, block_lang))
            signals.extend(sigs)

    if variation:
        pool = STREAM_VARIATIONS if template.source_type == "stream_transcript" else VARIATIONS
        var_lang = "ru" if lang == "mixed" else base
        for _ in range(variation):
            extra = ctx.fill(rng.choice(pool[var_lang]), var_lang)
            parts = [extra, *parts] if rng.random() < 0.3 else [*parts, extra]
    text = render_layout(template.layout, parts, template.deco, rng)
    if template.layout not in ("stream", "screen") and template.deco.get("end_emoji") and rng.random() < 0.4:
        text = f"{text} {template.deco['end_emoji']}"

    # Link hiding works on placeholders, so it runs before entity substitution.
    entity_values = list(ctx.entity_values)
    hidden: set[int] = set()
    obf_lang = "ru" if lang in ("ru", "mixed") or (lang == "translit" and base == "ru") else "kk"
    text_obfuscations = [] if template.source_type == "screenshot" else list(TEXT_OBFUSCATIONS)
    if template.source_type == "stream_transcript":
        text_obfuscations = ["slang"]
    if lang == "translit":
        text_obfuscations = [k for k in text_obfuscations if k != "homoglyph"]
    if text_obfuscations and rng.random() < obfuscate_prob:
        link_ids = [i for i, (slot, _) in enumerate(entity_values) if slot in ("ref_link", "short_link")]
        kinds = list(text_obfuscations)
        if link_ids and allow_link_hiding and template.source_type != "stream_transcript":
            kinds.append("link_hiding")
        for kind in rng.sample(kinds, k=min(len(kinds), rng.choice([1, 1, 2]))):
            if kind == "link_hiding":
                for i in link_ids:
                    phrase, sig = rng.choice(LINK_HIDING[obf_lang])
                    text = text.replace(f"{PH_OPEN}{i}{PH_CLOSE}", phrase)
                    hidden.add(i)
                    signals.append(sig)
                obfuscations.append("link_hiding")
                continue
            new = TEXT_OBFUSCATIONS[kind](text, obf_lang, rng)
            if new is not None and new != text:
                text = new
                obfuscations.append(kind)

    if lang == "translit":
        text = to_translit(text, base, rng)
        obfuscations.append("translit")

    entities: dict[str, list[str]] = {key: [] for key in ENTITY_KEYS}
    slot_values: dict[str, str] = {}
    for i, (slot, value) in enumerate(entity_values):
        if i in hidden:
            continue
        entities[ENTITY_SLOTS[slot]].append(value)
        slot_values.setdefault(slot, value)
        if slot == "ref_link":
            host = re.sub(r"^https?://", "", value).split("/")[0]
            entities["domains"].append(host)
    text = _PH_RE.sub(lambda m: entity_values[int(m.group(1))][1], text)
    text = re.sub(r"[ \t]+", " ", text).replace(" \n", "\n").strip()

    present = {slot for i, (slot, _) in enumerate(entity_values) if i not in hidden}
    if "ref_link" not in present:
        signals = [s for s in signals if s != "K_REF_TRACKING"]
    entities = {k: unique(v) for k, v in entities.items()}
    return Rendered(text, unique(signals), unique(obfuscations), entities, slot_values)
