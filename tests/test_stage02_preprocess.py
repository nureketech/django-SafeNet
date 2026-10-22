"""Stage 2: normalisation, position mapping, entity extraction, language detection."""

from __future__ import annotations

import time

import pandas as pd
import pytest

from safenet import paths
from safenet.preprocess import fold, normalize
from tests.conftest import require


# ------------------------------------------------------------ normalisation
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("к@зинo", "казино"),
        ("к а с п и", "каспи"),
        ("к-а-с-п-и", "каспи"),
        ("К.А.З.И.Н.О", "казино"),
        ("kaзинo", "казино"),
        ("ка3ино", "казино"),
        ("казин0", "казино"),
        ("дох0д", "доход"),
        ("р.а.н.д.о.м.а.й.з.е.р", "рандомайзер"),
        ("aqsha sal", "ақша сал"),
        ("jeke jaz", "жеке жаз"),
        ("tegin utys", "тегін утыс"),
        ("jüldeni ūtyp al", "жүлдені ұтып ал"),
        ("ağa, rakhmet", "аға, рахмет"),
        ("vlozhi 5000 poluchi 10000", "вложи 5000 получи 10000"),
        ("rozygrysh", "розыгрыш"),
        ("randomayzer", "рандомайзер"),
        ("R o z y g r y sh", "розыгрыш"),
        ("Kaspi Gold", "каспи голд"),
        ("Перевод на Kaspi", "перевод на каспи"),
        ("Заходи в kazino", "заходи в казино"),
        ("BONUS 200%", "бонус 200%"),
        ("ГАРАНТИРОВАННО — без риска", "гарантированно - без риска"),
        ("  много   пробелов \n\n строка ", "много пробелов\nстрока"),
        ("к а з и н о сегодня", "казино сегодня"),
    ],
)
def test_normalized_text(raw, expected):
    assert normalize(raw).text == expected


@pytest.mark.parametrize(
    "raw,token",
    [
        ("💰 есть тема", "деньги"),
        ("🎰 топ", "казино"),
        ("🔥🔥🔥 акция", "срочно"),
        ("💸 жібер, жүлде бар 🎁", "жүлде"),
        ("💸 скинь на карту", "перевод"),
    ],
)
def test_emoji_tokens(raw, token):
    text = normalize(raw).text
    assert token in text
    assert text.count(token) <= 2  # repeated emoji collapse into one token


def test_folding_keeps_length_and_unifies_spelling():
    assert fold("кепілді табыс") == fold("кепилди табыс")
    assert fold("всё") == "все"
    nt = normalize("Қатысу тегін, ұтыс бүгін")
    assert len(nt.folded) == len(nt.text)


def test_protected_entities_are_not_transliterated():
    nt = normalize("Регистрация https://Spin.test/?ref=AbC1 промокод SPIN200")
    assert "spin.test/?ref=abc1" in nt.text
    assert [e.text for e in nt.by_kind("promo_code")] == ["SPIN200"]
    assert [e.text for e in nt.by_kind("ref_param")] == ["?ref=AbC1"]


# -------------------------------------------------------------- amounts etc.
@pytest.mark.parametrize(
    "raw,amounts",
    [
        ("vlozhi 5k poluchi 10k", [5000, 10000]),
        ("5 мың теңге", [5000]),
        ("10 000 тг", [10000]),
        ("5000₸", [5000]),
        ("1,5 млн тг", [1_500_000]),
        ("20 тыс", [20000]),
        ("вложи 5к получи 10к", [5000, 10000]),
        ("1 myn tenge", [1000]),
        ("бонус 200% к депозиту", []),
        ("в 21:00 эфир", []),
    ],
)
def test_amounts(raw, amounts):
    assert normalize(raw).amounts() == amounts


@pytest.mark.parametrize(
    "raw,value,period",
    [
        ("доход 30% в день", 30, 1),
        ("депозит 15% годовых", 15, 365),
        ("күніне 10% түседі", 10, 1),
        ("25% в неделю", 25, 7),
        ("айына 40% пайда", 40, 30),
        ("двести процентов", 200, None),
    ],
)
def test_percents_with_period(raw, value, period):
    pcts = normalize(raw).by_kind("percent")
    assert len(pcts) == 1
    assert pcts[0].value == value
    assert pcts[0].meta["period_days"] == period


@pytest.mark.parametrize(
    "raw,value",
    [("х2 за неделю", 2), ("Сделал x10 на слотах", 10), ("в 3 раза больше", 3), ("екі есе", 2)],
)
def test_multipliers(raw, value):
    mults = normalize(raw).by_kind("multiplier")
    assert [m.value for m in mults] == [value]


def test_esports_format_is_not_multiplier():
    assert not normalize("турнир 5x5").by_kind("multiplier")


# ----------------------------------------------------------- raw entities
@pytest.mark.parametrize(
    "raw,kind,text",
    [
        ("ссылка bit.ly/3Xk9aQ", "shortener", "bit.ly/3Xk9aQ"),
        ("жми https://clck.ru/3BxYz", "shortener", "https://clck.ru/3BxYz"),
        ("cutt.ly/k7Pzq2 👈", "shortener", "cutt.ly/k7Pzq2"),
        ("https://lucky.test/?ref=abc", "ref_param", "?ref=abc"),
        ("?ref=abc", "ref_param", "?ref=abc"),
        ("https://x.test/reg?utm_source=tg&utm_campaign=q1", "ref_param", "?utm_source=tg"),
        ("https://x.test/r/abc12", "ref_param", "/r/abc12"),
        ("зеркало spin-arena77.test работает", "domain", "spin-arena77.test"),
        ("карта 4400 43** **** 1234", "card", "4400 43** **** 1234"),
        ("карта 4400 4302 1234 5678", "card", "4400 4302 1234 5678"),
        ("карта 4400-4302-1234-5678", "card", "4400-4302-1234-5678"),
        ("звони +7 701 234 56 78", "phone", "+7 701 234 56 78"),
        ("звони 87012345678", "phone", "87012345678"),
        ("каспи 8 (705) 738-62-71", "phone", "8 (705) 738-62-71"),
        ("номер +7 707 *** ** 12", "phone", "+7 707 *** ** 12"),
        ("промокод SPIN200 даёт бонус", "promo_code", "SPIN200"),
        ("по моему промику JACK50", "promo_code", "JACK50"),
        ("промокод: QAZ500", "promo_code", "QAZ500"),
        ("BARYS-898 промокоды", "promo_code", "BARYS-898"),
        ("Используй FS-8446, чтобы забрать бонус", "promo_code", "FS-8446"),
        ("оплата на Kaspi", "kaspi", "Kaspi"),
        ("каспиге аударыңыз", "kaspi", "каспиге"),
        ("k a s p i перевод", "kaspi", "k a s p i"),
    ],
)
def test_entities(raw, kind, text):
    found = [e.text for e in normalize(raw).by_kind(kind)]
    assert text in found, found


@pytest.mark.parametrize(
    "raw,kind",
    [
        ("Каспий теңізі", "kaspi"),
        ("iPhone 15 и PS5", "promo_code"),
        ("турнир CS2, формат BO3", "promo_code"),
        ("вклад 4400 тг", "card"),
        ("сегодня в 20:00", "phone"),
    ],
)
def test_no_false_entities(raw, kind):
    assert not normalize(raw).by_kind(kind)


def test_card_numbers_inside_text_are_not_amounts():
    nt = normalize("перевод на карту 4400 4302 1234 5678, взнос 1000 тг")
    assert nt.amounts() == [1000]


# ------------------------------------------------------------- position map
@pytest.mark.parametrize(
    "raw,fragment,original",
    [
        ("Заходи в к@зинo сегодня", "казино", "к@зинo"),
        ("Это к а з и н о топ", "казино", "к а з и н о"),
        ("vlozhi 5k poluchi 10k", "вложи", "vlozhi"),
        ("vlozhi 5k poluchi 10k", "получи 10к", "poluchi 10k"),
        ("💰 есть тема", "деньги", "💰"),
        ("Ставь лайк — ГАРАНТИРОВАННО!", "гарантированно", "ГАРАНТИРОВАННО"),
        ("Kaspi-ге аудар", "каспи", "Kaspi"),
        ("Zhenimpazdy r.a.n.d.o.m.a.i.z.e.rmen", "рандомаізермен", "r.a.n.d.o.m.a.i.z.e.rmen"),
        ("aqsha   sal", "ақша сал", "aqsha   sal"),
        ("Кепілді табыс", "кепилди", "Кепілді"),
    ],
)
def test_position_map(raw, fragment, original):
    nt = normalize(raw)
    haystack = nt.folded if fragment == fold(fragment) and fragment not in nt.text else nt.text
    idx = haystack.find(fragment)
    assert idx >= 0, (nt.text, fragment)
    assert nt.original_fragment(idx, idx + len(fragment)) == original


def test_position_map_entities_point_to_original():
    raw = "Вложи 5к, получи 15 000 тг за 3 дня! Каспи +7 701 234 56 78"
    nt = normalize(raw)
    for ent in nt.entities:
        assert raw[ent.start:ent.end] == ent.text


# ---------------------------------------------------------- language detect
LANG_CASES = {
    "ru": [
        "Всем привет! Кто идёт сегодня на футбол?",
        "Продаю детскую коляску, состояние отличное",
        "Розыгрыш айфона, участие 1000 тг, итоги в пятницу",
        "Вложи 5000 и получи 10000 через неделю",
        "Ссылка в профиле, промокод даёт бонус",
        "Напоминаю, завтра собрание в школе в 18:00",
        "Новое зеркало: старое не открывается",
        "Подскажите хорошего стоматолога в Алматы",
        "Куратор всё объяснит, пишите в личку",
        "Сегодня вечером стрим, не пропустите",
    ],
    "kk": [
        "Бүгін кешке футболға кім барады?",
        "Қатысу тегін, тек парақшаға жазылыңыз",
        "Жеңімпазды рандомайзермен таңдаймыз",
        "Ертең мектепте ата-аналар жиналысы болады",
        "Төлем Kaspi арқылы, жеткізу қала бойынша",
        "Күніне 10% табыс, тәуекел жоқ",
        "Сілтеме биода, тіркеліңдер",
        "Балалар, үй тапсырмасы қандай?",
        "Бәріңізге сәтті күн тілеймін",
        "Жүлде қоры бір миллион теңге",
    ],
    "mixed": [
        "Ребята, ертең футбол болады ма? Погода вроде норм",
        "Всем привет! Бүгін кешке ұтыс болады, участие 1000 тг",
        "Кто ищет подработку? Күніне 10% түседі, жеке жаз",
        "Сәлем всем, сегодня вечером эфир бар ма?",
        "Розыгрыш продолжается! Қатысқысы келгендер жазыңыз",
        "Мамочки в декрете! Үйден шықпай табыс табыңыз",
        "Жаңа бонус: регистрация жасасаң бонус к депозиту, ссылка в био",
        "Девочки, кім барады ертең? Я тоже хочу пойти",
        "Спасибо всем, баршаңызға рахмет за поддержку",
        "Привет, ты где? Мен үйдемін, приходи",
    ],
    "translit": [
        "vlozhi 5k poluchi 10k",
        "aqsha sal, tabys kepildi",
        "privet vsem, kto idet zavtra na koncert?",
        "zanos dnya! podnyal 780k na slotah",
        "uchastie 500 tg, perevod na kaspi",
        "jeke jaz, bari tusindiremin",
        "kazino platit, ssylka v bio",
        "bugin keshke utys bolady",
        "prodam iphone 13, oplata kaspi",
        "Tegin syilyq, tek jazylyp, dosyndy belgile",
    ],
}


@pytest.mark.parametrize("lang,text", [(lang, t) for lang, items in LANG_CASES.items() for t in items])
def test_language_detection(lang, text):
    assert normalize(text).lang == lang


# --------------------------------------------------------------- dataset run
def test_whole_dataset_runs_fast():
    require(paths.DATASET_PATH)
    df = pd.read_parquet(paths.DATASET_PATH, columns=["text", "lang"])
    start = time.perf_counter()
    results = [normalize(t) for t in df["text"]]
    avg_ms = (time.perf_counter() - start) / len(df) * 1000
    assert avg_ms < 5, f"{avg_ms:.2f} ms per record"
    assert all(r.text for r in results)
    assert all(len(r.starts) == len(r.text) == len(r.ends) == len(r.folded) for r in results)
    accuracy = sum(r.lang == lang for r, lang in zip(results, df["lang"])) / len(df)
    assert accuracy >= 0.9, accuracy


def test_dataset_entities_recovered():
    from safenet.common import to_entities

    require(paths.DATASET_PATH)
    df = pd.read_parquet(paths.DATASET_PATH, columns=["text", "entities"])
    for key in ("cards", "phones", "links"):
        expected = found = 0
        for text, ents in zip(df["text"], df["entities"].map(to_entities)):
            got = set(normalize(text).entity_dict()[key])
            expected += len(ents[key])
            found += len(set(ents[key]) & got)
        assert found / expected >= 0.98, key
