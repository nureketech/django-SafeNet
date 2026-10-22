"""Character tables used by the normaliser."""

from __future__ import annotations

KK_LETTERS = set("әғқңөұүһі")
CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя") | KK_LETTERS
VOWELS_CYR = set("аеёиоуыэюяәөұүі")

# Folding Kazakh-specific letters (and ё/э/ъ) to their Russian look-alikes keeps the
# text length unchanged, so spans stay valid while «кепилди» == «кепілді».
FOLD = str.maketrans({
    "ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и",
    "ё": "е", "э": "е", "ъ": "ь",
})

DASHES = {"‐", "‑", "‒", "–", "—", "―", "−"}
QUOTES = {"«", "»", "“", "”", "„", "‟", "″"}
INVISIBLE = {"​", "‌", "‍", "⁠", "︎", "️", "­"}

# Latin look-alikes inside Cyrillic words (lowercase, after case folding).
HOMOGLYPHS = {
    "a": "а", "b": "в", "c": "с", "e": "е", "h": "н", "k": "к", "m": "м", "o": "о", "p": "р",
    "t": "т", "x": "х", "y": "у", "u": "и", "n": "п", "r": "г",
}
# Digits / symbols used as letters ("leet") inside Cyrillic words.
LEET_CYR = {"0": "о", "3": "з", "@": "а", "4": "ч", "6": "б", "$": "с"}
LEET_LAT = {"0": "o", "3": "z", "@": "a", "1": "i", "$": "s", "4": "ch"}

# Latin -> Cyrillic transliteration; longest match first.
TRANSLIT_MULTI = [
    ("shch", "щ"), ("sch", "щ"), ("zh", "ж"), ("kh", "х"), ("ch", "ч"), ("sh", "ш"), ("ts", "ц"),
    ("yu", "ю"), ("iu", "ю"), ("ya", "я"), ("ia", "я"), ("yo", "ё"), ("gh", "ғ"),
]
TRANSLIT_SINGLE = {
    "a": "а", "b": "б", "c": "с", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и",
    "j": "ж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р",
    "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "х", "z": "з", "'": "ь", "’": "ь",
    "ä": "ә", "ö": "ө", "ü": "ү", "ū": "ұ", "ğ": "ғ", "ş": "ш", "ç": "ч", "ı": "ы", "ñ": "ң",
    "ń": "ң", "á": "ә", "ó": "ө", "ú": "ұ", "ý": "у", "ǵ": "ғ",
}
TRANSLIT_KK_OVERRIDES = {"q": "қ", "i": "і", "w": "у", "'": "", "’": ""}
TRANSLIT_KK_MULTI = {"gh": "ғ"}
TRANSLIT_RU_MULTI = {"gh": "г"}
LATIN_VOWELS = set("aeiouyäöüūáóúı")
KK_TRANSLIT_CHARS = set("qğäöüūñşıńáóúǵ")
KK_TRANSLIT_STEMS = (
    "zhane", "jane", "ushin", "bugin", "tegin", "jeke", "zheke", "aqsh", "aksha", "utys", "qatys",
    "katys", "kepil", "tabys", "salym", "kerek", "rakhmet", "rahmet", "salem", "siltem", "tolem",
    "audar", "jenim", "zhenim", "zhuld", "juld", "oryn", "qal", "barlyq", "barlyk", "kazir", "qazir",
    "bolady", "jaz", "zhaz", "dosyn", "aqy", "kuni", "kunine", "aptasy", "ayyna",
)

# Pure-Latin words that are common in Russian/Kazakh scheme posts.
LATIN_WORDS = {
    "kaspi": "каспи", "casino": "казино", "kazino": "казино", "bonus": "бонус", "bonusy": "бонусы",
    "freespin": "фриспин", "freespins": "фриспины", "jackpot": "джекпот", "slot": "слот",
    "slots": "слоты", "promo": "промо", "promocode": "промокод", "promokod": "промокод",
    "promik": "промик", "deposit": "депозит", "depozit": "депозит", "roulette": "рулетка",
    "poker": "покер", "iphone": "айфон", "giveaway": "розыгрыш", "invest": "инвест",
    "profit": "профит", "vpn": "впн", "bio": "био", "link": "линк", "stream": "стрим",
    "ref": "реф", "vip": "вип", "bet": "бет", "win": "вин", "cash": "кэш", "money": "деньги",
}
# Stems that make a transliterated Latin word worth replacing inside a Cyrillic text.
KEYWORD_STEMS = (
    "казин", "каспи", "депоз", "промок", "бонус", "слот", "ставк", "вывод", "занос", "рандом",
    "розыгр", "перевод", "влож", "доход", "гарант", "зеркал", "ссылк", "фриспин", "лудик", "выигр",
    "кешбэк", "рулетк", "джекпот", "куратор", "выплат", "пирамид",
)

EMOJI_TOKENS = {
    "💰": ("деньги", "ақша"),
    "💵": ("деньги", "ақша"),
    "🤑": ("деньги", "ақша"),
    "🎰": ("казино", "казино"),
    "💸": ("перевод", "аударым"),
    "🎁": ("приз", "жүлде"),
    "🔥": ("срочно", "шұғыл"),
}

SHORTENERS = {
    "bit.ly", "clck.ru", "cutt.ly", "tinyurl.com", "goo.su", "t.ly", "is.gd", "vk.cc", "shorturl.at",
    "rb.gy", "ow.ly", "u.to", "tiny.cc",
}
TLDS = (
    "test", "example", "com", "ru", "kz", "net", "org", "io", "me", "ly", "cc", "app", "site",
    "online", "club", "bet", "casino", "top", "xyz", "pro", "info", "biz", "link", "live", "win",
    "vip", "su", "ua", "by", "uz", "kg", "co", "tv", "gg", "to", "at", "gd", "fun", "space", "store",
    "shop", "ai",
)

NUMBER_WORDS = {
    # Russian
    "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
    "семь": 7, "восемь": 8, "девять": 9, "десять": 10, "пятнадцать": 15, "двадцать": 20,
    "тридцать": 30, "сорок": 40, "пятьдесят": 50, "сто": 100, "двести": 200, "триста": 300,
    "пятьсот": 500, "тысяча": 1000, "тысячу": 1000,
    # Kazakh (folded spelling)
    "бир": 1, "еки": 2, "уш": 3, "торт": 4, "бес": 5, "алты": 6, "жети": 7, "сегиз": 8, "тогыз": 9,
    "он": 10, "жиырма": 20, "отыз": 30, "кырык": 40, "елу": 50, "жуз": 100, "мын": 1000,
}
