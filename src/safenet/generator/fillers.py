"""Random slot fillers for the synthetic generator.

All personal data is fictional: card numbers are generated so that they FAIL the
Luhn check, phone numbers are random, domains live in ``.test`` / ``.example``
zones, names are conventional nicknames and casino / project names are invented.
"""

from __future__ import annotations

import random
import string

from safenet.common import luhn_valid

NICK_A = [
    "lucky", "aru", "sultan", "dana", "aidos", "zhuldyz", "baha", "nurik", "kairat", "asel",
    "miras", "tomiris", "ernur", "madi", "dias", "aya", "sanzhar", "bota", "erlan", "gaukhar",
    "arman", "dilnaz", "beka", "alina", "temir", "saule", "rustem", "kamila", "yerzhan", "ainur",
    "aknur", "zarina", "olzhas", "meruert", "timur", "inkar", "daulet", "symbat", "azamat", "aisulu",
]
NICK_B = [
    "shop", "win", "gift", "kz", "pro", "life", "official", "777", "money", "invest", "bonus",
    "store", "almaty", "astana", "shym", "vip", "team", "club", "mama", "art", "home", "style",
]

CASINOS = [
    "Spinomad", "Altynslot", "Jackpotan", "Kumisbet", "Fortunix", "Slotaru", "Zharqyn Spin",
    "Betaryq", "Luckyshar", "Tulpar Slots", "Qazyna Spin", "Nomadbet",
]
PROJECTS = [
    "Tumar Profit", "Ulytau Coin", "Sapar Trade", "Kuat Invest", "Jarqyn Club", "Nomad Profit",
    "QazPro Invest", "Altyn Capital Club", "Saryarqa Money Team", "Ken Dala Fund",
]
BANKS = ["Тұмар Банк", "Көктем Банк", "Шұғыла Банк", "Жібек Финанс", "Сенім Капитал Банк"]
DEPOSIT_NAMES = ["Жинақ", "Болашақ", "Сенімді", "Отбасы плюс", "Комфорт", "Стандарт", "Мақсат"]
SHOPS = ["«Береке-маркет»", "«Ақниет»", "«Дала Store»", "«Томирис»", "«Мерей»", "«Нұр-Сауда»"]
PROMO_WORDS = [
    "SPIN", "LUCKY", "BARYS", "ALTYN", "BONUS", "WIN", "JACK", "NOMAD", "TULPAR", "VIP",
    "KZ", "QAZ", "MEGA", "FS", "ZHULDYZ", "STREAM", "GOLD", "TOP",
]
CODE_ALPHABET = string.ascii_letters + string.digits
SHORTENERS = ["bit.ly", "clck.ru", "cutt.ly"]
DOMAIN_WORDS = [
    "lucky", "spin", "bonus", "win", "slot", "play", "star", "gold", "nomad", "barys", "kumis",
    "fortune", "jackpot", "mega", "top", "profit", "invest", "club", "smart", "qaz", "dala", "tumar",
]
CARD_PREFIXES = ["4400 43", "4400 44", "5169 49", "4405 63", "5522 04", "4149 49"]
PHONE_CODES = ["700", "701", "702", "705", "707", "708", "747", "771", "775", "776", "777", "778"]

PRIZES = {
    "ru": [
        "iPhone 15", "айфон 16 Pro", "PlayStation 5", "500 000 тг", "1 000 000 тенге", "автомобиль",
        "MacBook Air", "Samsung Galaxy S24", "AirPods Pro", "золотое кольцо", "телевизор 55 дюймов",
        "путёвку в Дубай", "300 000 ₸", "холодильник", "электросамокат", "200 000 тенге",
    ],
    "kk": [
        "iPhone 15", "айфон 16 Pro", "PlayStation 5", "500 000 теңге", "1 000 000 теңге", "көлік",
        "MacBook Air", "Samsung Galaxy S24", "AirPods Pro", "алтын сақина", "55 дюймдық теледидар",
        "Дубайға жолдама", "300 000 ₸", "тоңазытқыш", "электросамокат", "200 000 теңге",
    ],
}
SMALL_PRIZES = {
    "ru": [
        "сертификат на 10 000 тг", "торт на заказ", "набор косметики", "книгу", "наушники",
        "букет цветов", "футболку с логотипом", "абонемент в зал", "кофе на месяц",
        "набор игрушек", "сертификат на маникюр", "фирменный термос",
    ],
    "kk": [
        "10 000 теңгелік сертификат", "тапсырыспен торт", "косметика жиынтығы", "кітап",
        "құлаққап", "гүл шоғы", "логотипі бар футболка", "спортзалға абонемент",
        "бір айлық кофе", "ойыншықтар жиынтығы", "маникюрге сертификат", "фирмалық термос",
    ],
}
ITEMS = {
    "ru": [
        "кроссовки Nike", "детские куртки", "домашние торты", "чак-чак", "казы", "парфюм",
        "чехлы для телефонов", "сумки", "постельное бельё", "мёд", "курт", "платья",
        "игрушки", "косметику", "посуду", "баурсаки на заказ",
    ],
    "kk": [
        "Nike кроссовкалары", "балалар күртешелері", "үй торттары", "шақ-шақ", "қазы", "иіссу",
        "телефон қаптары", "сөмкелер", "төсек жаймалары", "бал", "құрт", "көйлектер",
        "ойыншықтар", "косметика", "ыдыс-аяқ", "тапсырысқа бауырсақ",
    ],
}
CITIES = {
    "ru": [
        "Алматы", "Астана", "Шымкент", "Караганда", "Актобе", "Тараз", "Павлодар",
        "Усть-Каменогорск", "Семей", "Атырау", "Костанай", "Кызылорда", "Уральск", "Актау",
    ],
    "kk": [
        "Алматы", "Астана", "Шымкент", "Қарағанды", "Ақтөбе", "Тараз", "Павлодар", "Өскемен",
        "Семей", "Атырау", "Қостанай", "Қызылорда", "Орал", "Ақтау",
    ],
}
DAYS = {
    "ru": ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"],
    "kk": ["дүйсенбі", "сейсенбі", "сәрсенбі", "бейсенбі", "жұма", "сенбі", "жексенбі"],
}
MONTHS = {
    "ru": ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
           "сентября", "октября", "ноября", "декабря"],
    "kk": ["қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым", "шілде", "тамыз",
           "қыркүйек", "қазан", "қараша", "желтоқсан"],
}
GAMES = ["Dota 2", "CS2", "Mobile Legends", "PUBG Mobile", "Valorant", "FIFA 25", "Brawl Stars",
         "Standoff 2"]
OCCASIONS = {
    "ru": ["Наурыза", "дня рождения магазина", "10 000 подписчиков", "Нового года", "8 марта",
           "открытия филиала"],
    "kk": ["Наурыз мерекесі", "дүкеннің туған күні", "10 000 жазылушы", "Жаңа жыл", "8 наурыз",
           "жаңа филиалдың ашылуы"],
}
DISTRICTS = ["Самал", "Алмагуль", "Нуркент", "Жетысу", "Аксай", "Орбита", "Каратал", "Сайран"]


def nick(rng: random.Random) -> str:
    sep = rng.choice(["_", ".", ""])
    tail = rng.choice(["", str(rng.randint(1, 99)), "_kz"])
    return f"@{rng.choice(NICK_A)}{sep}{rng.choice(NICK_B)}{tail}"


def card(rng: random.Random) -> str:
    """A 16-digit card number that deliberately fails the Luhn check."""
    prefix = rng.choice(CARD_PREFIXES).replace(" ", "")
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(16 - len(prefix)))
    if luhn_valid(body):
        last = (int(body[-1]) + rng.randint(1, 9)) % 10
        body = body[:-1] + str(last)
    assert not luhn_valid(body)
    sep = rng.choice([" ", " ", "-", ""])
    return sep.join(body[i:i + 4] for i in range(0, 16, 4))


def masked_card(rng: random.Random) -> str:
    raw = card(rng).replace(" ", "").replace("-", "")
    return rng.choice([
        f"{raw[:4]} {raw[4:6]}** **** {raw[-4:]}",
        f"{raw[:4]} **** **** {raw[-4:]}",
        f"**** **** **** {raw[-4:]}",
    ])


def phone(rng: random.Random) -> str:
    code = rng.choice(PHONE_CODES)
    a, b, c = rng.randint(100, 999), rng.randint(10, 99), rng.randint(10, 99)
    return rng.choice([
        f"+7 {code} {a} {b} {c}",
        f"8{code}{a}{b}{c}",
        f"+7{code}{a}{b}{c}",
        f"8 ({code}) {a}-{b}-{c}",
        f"+7 ({code}) {a} {b} {c}",
    ])


def masked_phone(rng: random.Random) -> str:
    code = rng.choice(PHONE_CODES)
    return rng.choice([f"+7 {code} *** ** {rng.randint(10, 99)}", f"8 {code} *** ** {rng.randint(10, 99)}"])


def _code(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(CODE_ALPHABET) for _ in range(n))


def domain(rng: random.Random, words: list[str] | None = None) -> str:
    pool = words or DOMAIN_WORDS
    tld = rng.choice(["test", "example"])
    style = rng.random()
    if style < 0.45:
        return f"{rng.choice(pool)}-{rng.choice(DOMAIN_WORDS)}{rng.randint(1, 99)}.{tld}"
    if style < 0.75:
        return f"{rng.choice(pool)}{rng.choice(DOMAIN_WORDS)}.{tld}"
    return f"{_code(rng, 6).lower()}.{tld}"


def casino_domain(rng: random.Random, casino_name: str) -> str:
    slug = casino_name.lower().replace(" ", "")
    tld = rng.choice(["test", "example"])
    return rng.choice([
        f"{slug}{rng.randint(1, 99)}.{tld}",
        f"{slug}-mirror{rng.randint(1, 30)}.{tld}",
        f"{slug}-kz.{tld}",
        f"play-{slug}.{tld}",
    ])


def promo(rng: random.Random) -> str:
    word = rng.choice(PROMO_WORDS)
    return rng.choice([
        f"{word}{rng.randint(10, 999)}",
        f"{word}{rng.choice(PROMO_WORDS)}{rng.randint(1, 99)}",
        f"{word}-{rng.randint(100, 9999)}",
    ])


def ref_link(rng: random.Random, dom: str) -> str:
    code = _code(rng, rng.randint(4, 8))
    return rng.choice([
        f"https://{dom}/?ref={code}",
        f"https://{dom}/reg?utm_source=tg&utm_campaign={code.lower()}",
        f"https://{dom}/r/{code}",
        f"{dom}/?ref={code}",
        f"https://{dom}/?utm_source=instagram&utm_medium=bio&utm_content={code.lower()}",
        f"{dom}/r/{code.lower()}",
    ])


def short_link(rng: random.Random) -> str:
    host = rng.choice(SHORTENERS)
    prefix = rng.choice(["", "https://"])
    return f"{prefix}{host}/{_code(rng, rng.randint(5, 7))}"


def shop_domain(rng: random.Random) -> str:
    return f"{rng.choice(['dala', 'mereke', 'nur', 'aru', 'tomiris', 'shanyrak'])}-shop{rng.randint(1, 50)}.example"


def money_value(rng: random.Random, low: int, high: int, step: int) -> int:
    return rng.randrange(low, high + 1, step)


def fmt_amount(value: int, lang: str, rng: random.Random, allow_short: bool = True) -> str:
    """Format a tenge amount in one of many realistic styles."""
    spaced = f"{value:,}".replace(",", " ")
    styles = ["plain", "spaced", "tg", "sign", "word"]
    if allow_short and value % 1000 == 0 and value >= 1000:
        styles += ["k", "thousand"]
    style = rng.choice(styles)
    if style == "plain":
        return str(value)
    if style == "spaced":
        return spaced
    if style == "tg":
        return f"{rng.choice([spaced, str(value)])} тг"
    if style == "sign":
        return rng.choice([f"{spaced} ₸", f"{value}₸"])
    if style == "word":
        return f"{spaced} {'теңге' if lang == 'kk' else 'тенге'}"
    thousands = value // 1000
    if style == "k":
        return f"{thousands}{rng.choice(['к', 'k', 'к'])}"
    if lang == "kk":
        return rng.choice([f"{thousands} мың", f"{thousands} мың теңге"])
    return rng.choice([f"{thousands} тыс", f"{thousands} тыс. тг", f"{thousands} тысяч"])


def fmt_time(rng: random.Random) -> str:
    return f"{rng.randint(10, 22)}:{rng.choice(['00', '30', '00', '15', '45'])}"


def fmt_date(rng: random.Random, lang: str) -> str:
    return f"{rng.randint(1, 28)} {rng.choice(MONTHS[lang])}"
