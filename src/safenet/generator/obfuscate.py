"""Language transforms and obfuscation techniques used by the generator.

Entity placeholders (private-use characters) are never touched, so card numbers,
phones, links and promo codes survive every transform unchanged.
"""

from __future__ import annotations

import random
import re

# ------------------------------------------------------------------ translit
_RU_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "з": "z", "и": "i", "к": "k",
    "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "ц": "ts", "ч": "ch", "ш": "sh", "ъ": "", "ы": "y", "э": "e",
}
_KK_SIMPLE = {"ә": "a", "ғ": "gh", "қ": "q", "ң": "n", "ө": "o", "ұ": "u", "ү": "u", "һ": "h", "і": "i"}
_KK_DIACRITIC = {"ә": "ä", "ғ": "ğ", "қ": "q", "ң": "ñ", "ө": "ö", "ұ": "ū", "ү": "ü", "һ": "h", "і": "i"}
_VOWELS = set("аеёиоуыэюяәөұүі")


def to_translit(text: str, lang: str, rng: random.Random) -> str:
    """Informal Cyrillic -> Latin transliteration (one random style per record)."""
    table = dict(_RU_LAT)
    table.update(_KK_SIMPLE)  # Kazakh letters may occur in names inside Russian texts
    table["ж"] = rng.choice(["zh", "j"]) if lang == "kk" else rng.choice(["zh", "zh", "j"])
    table["х"] = rng.choice(["h", "kh", "h"])
    table["щ"] = rng.choice(["sch", "sh"])
    table["ь"] = rng.choice(["", "", "'"])
    table["ё"] = rng.choice(["yo", "e"])
    table["ю"] = rng.choice(["yu", "iu"])
    table["я"] = rng.choice(["ya", "ia"])
    if lang == "kk":
        table.update(rng.choice([_KK_SIMPLE, _KK_SIMPLE, _KK_DIACRITIC]))
    out: list[str] = []
    prev = ""
    for i, ch in enumerate(text):
        low = ch.lower()
        if low == "й":
            rep = "y" if prev in _VOWELS else "i"
        elif low in table:
            rep = table[low]
        else:
            out.append(ch)
            prev = low
            continue
        if ch != low and rep:
            neighbours = text[max(0, i - 1):i] + text[i + 1:i + 2]
            all_caps = any(c.isupper() for c in neighbours)
            rep = rep.upper() if all_caps else rep[0].upper() + rep[1:]
        out.append(rep)
        prev = low
    return "".join(out)


# --------------------------------------------------------------- obfuscation
KEYWORDS = {
    "ru": [
        "казино", "каспи", "kaspi", "слоты", "слот", "ставки", "ставка", "депозит", "вывод",
        "выигрыш", "занос", "розыгрыш", "рандомайзер", "взнос", "перевод", "карту", "карта",
        "доход", "вложи", "гарантированно", "промокод", "зеркало", "бонус", "фриспин", "куратор",
        "выплата", "выплаты", "лотерея", "колесо", "прибыль",
    ],
    "kk": [
        "казино", "каспи", "kaspi", "слот", "ставка", "депозит", "вывод", "ұтыс", "рандомайзер",
        "жарна", "аударым", "карта", "табыс", "кепілді", "промокод", "айна", "зеркало", "бонус",
        "фриспин", "куратор", "төлем", "лотерея", "дөңгелек", "пайда", "салым",
    ],
}
_HOMOGLYPHS = {
    "а": ["a", "@"], "о": ["o", "0"], "е": ["e"], "с": ["c"], "р": ["p"], "к": ["k"],
    "х": ["x"], "у": ["y"], "з": ["3"], "и": ["u"],
}
WORD_EMOJI = {
    "ru": {"деньги": "💰", "казино": "🎰", "перевод": "💸", "приз": "🎁", "срочно": "🔥", "выигрыш": "💰"},
    "kk": {"ақша": "💰", "ақшаны": "💰", "казино": "🎰", "аударым": "💸", "жүлде": "🎁", "шұғыл": "🔥"},
}
DIVIDERS = ["💰💰💰", "➖➖➖", "✨✨", "💸💸", "⭐⭐⭐", "🎰🎰"]
SLANG = {
    "ru": [
        ("депозиту", "депчику"), ("депозит", "депчик"), ("казино", "казик"), ("в личку", "в лс"),
        ("в директ", "в лс"), ("в личных сообщениях", "в лс"), ("регистрируйтесь", "регайтесь"),
        ("деньги", "бабки"), ("промокод", "промик"), ("фриспинов", "фриков"), ("телеграм", "тг"),
    ],
    "kk": [
        ("депозитке", "депке"), ("депозит", "деп"), ("казино", "казик"), ("жекеге", "лс-ке"),
        ("директке", "лс-ке"), ("жеке хабарламада", "лс-те"), ("промокод", "промик"),
        ("тіркеліңдер", "регайтесь"), ("телеграм", "тг"),
    ],
}


def _keyword_matches(text: str, lang: str) -> list[re.Match]:
    words = "|".join(sorted(map(re.escape, KEYWORDS[lang]), key=len, reverse=True))
    return list(re.finditer(rf"(?<!\w)({words})", text, flags=re.IGNORECASE))


def apply_spacing(text: str, lang: str, rng: random.Random) -> str | None:
    """Split one or two keywords into letters: «к а з и н о», «к-а-с-п-и»."""
    matches = _keyword_matches(text, lang)
    if not matches:
        return None
    for m in sorted(rng.sample(matches, k=min(len(matches), rng.choice([1, 1, 2]))), key=lambda m: -m.start()):
        sep = rng.choice([" ", " ", "-", "."])
        text = text[:m.start()] + sep.join(m.group(0)) + text[m.end():]
    return text


def apply_homoglyph(text: str, lang: str, rng: random.Random) -> str | None:
    """Replace some Cyrillic letters of keywords with Latin look-alikes or digits."""
    matches = [m for m in _keyword_matches(text, lang) if m.group(0).lower() != "kaspi"]
    if not matches:
        return None
    changed = False
    for m in sorted(rng.sample(matches, k=min(len(matches), rng.choice([1, 2, 2, 3]))), key=lambda m: -m.start()):
        word = list(m.group(0))
        positions = [i for i, ch in enumerate(word) if ch.lower() in _HOMOGLYPHS]
        if not positions:
            continue
        for i in rng.sample(positions, k=min(len(positions), rng.choice([1, 2, 2, 3]))):
            sub = rng.choice(_HOMOGLYPHS[word[i].lower()])
            word[i] = sub.upper() if word[i].isupper() and sub.isalpha() else sub
        text = text[:m.start()] + "".join(word) + text[m.end():]
        changed = True
    return text if changed else None


def apply_emoji(text: str, lang: str, rng: random.Random) -> str | None:
    """Replace words with emoji and/or insert emoji dividers."""
    changed = False
    for word, emoji in WORD_EMOJI[lang].items():
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
        if pattern.search(text) and rng.random() < 0.8:
            text = pattern.sub(emoji, text, count=1)
            changed = True
    if "\n" in text and rng.random() < 0.7:
        divider = rng.choice(DIVIDERS)
        text = text.replace("\n", f"\n{divider}\n", 1)
        changed = True
    elif rng.random() < 0.6:
        divider = rng.choice(DIVIDERS)
        text = f"{divider} {text} {divider}"
        changed = True
    return text if changed else None


def apply_slang(text: str, lang: str, rng: random.Random) -> str | None:
    changed = False
    for word, slang in SLANG[lang]:
        pattern = re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE)
        if pattern.search(text):
            text = pattern.sub(slang, text)
            changed = True
    if not changed:
        # Amount slang: «5 000 тг» / «5000 тенге» -> «5к».
        def repl(m: re.Match) -> str:
            return f"{int(m.group(1).replace(' ', '')) // 1000}к"

        new = re.sub(r"(?<!\d)(\d{1,3}(?: ?000))(?: ?(?:тг|тенге|теңге|₸))", repl, text)
        if new != text:
            return new
        return None
    return text


TEXT_OBFUSCATIONS = {
    "spacing": apply_spacing,
    "homoglyph": apply_homoglyph,
    "emoji": apply_emoji,
    "slang": apply_slang,
}

LINK_HIDING = {
    "ru": [("(ссылка в био)", "C_SHORTLINK"), ("👉 ссылка в профиле", "C_SHORTLINK"),
           ("ссылку скину в лс", "C_DM_REDIRECT"), ("линк в шапке", "C_SHORTLINK")],
    "kk": [("(сілтеме биода)", "C_SHORTLINK"), ("👉 сілтеме профильде", "C_SHORTLINK"),
           ("сілтемені жекеге жіберемін", "C_DM_REDIRECT"), ("линк шапкада", "C_SHORTLINK")],
}
