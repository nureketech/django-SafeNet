"""Stage 3: signals dictionary and evidence-based detector."""

from __future__ import annotations

import re

import pandas as pd
import pytest
import yaml

from safenet import paths
from safenet.signals import SignalDetector, detect
from safenet.signals.config import load_raw, validate
from tests.conftest import require

MINIMUM_SET = {
    "C_PAY_TRANSFER", "C_PERSONAL_CARD", "C_URGENCY", "C_REFERRAL", "C_SHORTLINK", "C_DM_REDIRECT",
    "L_PAID_ENTRY", "L_RANDOM_DRAW", "L_ORGANIZER_PROFIT", "L_SERIES",
    "K_GAMBLING_TERMS", "K_WIN_BRAG", "K_PROMO_BONUS", "K_REF_TRACKING", "K_MIRROR", "K_STREAM_CTA",
    "P_UNREAL_RETURN", "P_GUARANTEE", "P_REFERRAL_TIERS", "P_PAYOUT_PROOF", "P_ORG_STRUCTURE",
    "N_FREE_ENTRY", "N_WARNING_CONTEXT", "N_COMMERCE", "N_BANK_RATE",
}
CORE = {"L_PAID_ENTRY", "L_RANDOM_DRAW", "K_GAMBLING_TERMS", "P_UNREAL_RETURN"}
_KK = re.compile("[әғқңөұүһі]", re.I)

# For every signal: positive examples (must fire) and negative look-alikes (must not fire).
CASES: dict[str, dict[str, list[str]]] = {
    "C_PAY_TRANSFER": {
        "pos": ["Оплата на Kaspi +7 701 234 56 78", "Реквизиты для оплаты скину в личку",
                "Каспиге аударыңыз, чекті жіберіңіз", "Перевод на мою карту, потом скрин"],
        "neg": ["Перевели встречу на завтра", "Ребята, кто идёт на футбол?", "Сыйлықты курьер жеткізеді"],
    },
    "C_PERSONAL_CARD": {
        "pos": ["Кидай на мою карту", "Карта: 4400 43** **** 1234", "Менің картама салыңдар",
                "Номер карты 5169 4911 3083 5733"],
        "neg": ["Карта города обновилась", "Скидочная карта магазина", "Кеше картаны жоғалттым"],
    },
    "C_URGENCY": {
        "pos": ["Осталось 3 места!", "Успей до 20:00", "Тек бүгін жеңілдік", "5 орын қалды",
                "Торопись, набор закрывается"],
        "neg": ["Сегодня хорошая погода", "Бүгін кешке кездесеміз", "Мы не торопимся, всё обсудим"],
    },
    "C_REFERRAL": {
        "pos": ["Приведи друга и получи бонус", "Досыңды әкел — сыйлық ал", "Реферальная программа для всех",
                "Пригласил 3 друзей — получил скидку"],
        "neg": ["Отметь друга в комментариях", "Пошли с другом в кино", "Досыммен кездестім"],
    },
    "C_SHORTLINK": {
        "pos": ["Ссылка: bit.ly/3Xk9aQ", "Ссылка в био", "Сілтеме биода", "Все ссылки во втором канале"],
        "neg": ["Ссылка на документ в почте", "Сайт магазина shop-1.example", "Сілтемені кейін жіберемін"],
    },
    "C_DM_REDIRECT": {
        "pos": ["Пиши в лс", "Подробности в директ", "Жеке жаз", "Жекеге жазыңыздар", "Запись через лс"],
        "neg": ["Пишите в комментариях", "Директор школы сообщил", "Жазғы демалыс басталды"],
    },
    "L_PAID_ENTRY": {
        "pos": ["Участие 1000 тг", "Слот стоит 500", "Қатысу — 700₸", "Жарна 1500 ғана",
                "Платное участие, подробности в лс"],
        "neg": ["Участие бесплатное", "Тегін розыгрыш, жазылып, досыңды белгіле", "Номер телефона 87011234567",
                "Қатысу тегін", "Ставка 1000 на красное"],
    },
    "L_RANDOM_DRAW": {
        "pos": ["Победителя выберем рандомайзером", "Крутим колесо в прямом эфире", "Жеңімпаз кездейсоқ таңдалады",
                "Тікелей эфирде дөңгелек айналдырамыз"],
        "neg": ["Победитель определяется по результатам матчей", "Жеңімпаз ұпай бойынша анықталады",
                "Выбираем лучшую работу голосованием"],
    },
    "L_ORGANIZER_PROFIT": {
        "pos": ["Продано 80 из 100 слотов", "Уже собрано 150 000", "50 слоттың 30 сатылды",
                "Всего 100 номеров по 1000"],
        "neg": ["Осталось всего 3 места", "Продано! Спасибо всем", "Собрание в пятницу"],
    },
    "L_SERIES": {
        "pos": ["Розыгрыш №14", "Это уже 5-й розыгрыш", "Ұтыс №7", "Каждую пятницу новый розыгрыш"],
        "neg": ["Заказ №14 оформлен", "Каждую пятницу у нас йога", "Ұтыс ойынына қатыстым"],
    },
    "K_GAMBLING_TERMS": {
        "pos": ["Играю в казино каждый день", "Слоты и рулетка", "Спортқа ставкалар", "к@зинo топ",
                "🎰 платит без задержек", "Spinomad-тың слоттарында ойнадым"],
        "neg": ["Ставка вознаграждения 15% годовых", "Слот стоит 500 тенге", "Депозит в банке под 14%",
                "Бір слот 1000 теңге"],
    },
    "K_WIN_BRAG": {
        "pos": ["Мой занос за вечер", "Поднял 300к за час", "Вывод пришёл", "Кеше 500 мың ұттым"],
        "neg": ["Вывод из статьи простой", "Ұтыс ойыны тегін", "Баланс телефона пополнил"],
    },
    "K_PROMO_BONUS": {
        "pos": ["Промокод SPIN200", "Бонус 200% к депозиту", "BARYS-898 промокоды арқылы", "100 фриспинов по коду"],
        "neg": ["Скидка 10% на всё", "Бонусная карта магазина", "Код домофона 123"],
    },
    "K_REF_TRACKING": {
        "pos": ["https://spin.test/?ref=abc", "Регистрация: x.test/r/abc12", "Тіркелу: spin.test/?utm_source=tg",
                "Это моя реф ссылка"],
        "neg": ["Сайт магазина: shop.example", "https://news.test/article/123", "Реформа образования"],
    },
    "K_MIRROR": {
        "pos": ["Рабочее зеркало: spin.test", "Айна VPN-сіз ашылады", "Обход блокировки через зеркало",
                "Жұмыс істейтін айна"],
        "neg": ["Продаю зеркало в прихожую", "Барабан айналдырамыз", "Айнаш апай келді"],
    },
    "K_STREAM_CTA": {
        "pos": ["Регайтесь по моей ссылке в описании", "Сілтеме арқылы тіркеліңдер", "Ссылка под видео",
                "Переходите по ссылке"],
        "neg": ["Регистрация команд на турнир по ссылке", "Видео смотрите до конца", "Турнирге тіркелу профильде"],
    },
    "P_UNREAL_RETURN": {
        "pos": ["вложи 5000 получи 10000 за 3 дня", "Доход 3% в день", "х2 за неделю",
                "5 мың салсаң апта сайын 2 мың табыс түседі", "Күніне 10% табыс",
                "С 69 000 тенге делаем 138 000 тенге за неделю"],
        "neg": ["депозит 15% годовых", "Скидка 50% только сегодня", "Сделал x10 на слотах",
                "С каждого приглашённого 10%", "Жылдық 14% мөлшерлеме"],
    },
    "P_GUARANTEE": {
        "pos": ["Гарантированно!", "Без риска", "Кепілді табыс", "Тәуекелсіз"],
        "neg": ["Депозиты гарантированы КФГД", "Гарантия на работу 6 месяцев",
                "Абсолютно гарантированного дохода не бывает"],
    },
    "P_REFERRAL_TIERS": {
        "pos": ["1-я линия 10%, 2-я линия 5%", "1-деңгей 10%, 2-деңгей 5%", "С каждого приглашённого — 10%",
                "Маркетинг: 7 линий партнёрки"],
        "neg": ["Әлеуметтік желілерде жаңалық", "Уровень английского B2", "Линия метро закрыта"],
    },
    "P_PAYOUT_PROOF": {
        "pos": ["Скрины выплат в комментариях", "Выплата пришла, всё честно", "Төлем түсті, бәрі әділ",
                "Пруфы в канале"],
        "neg": ["Зарплату задерживают", "Төлемді ертең жасаймын", "Выписка из банка"],
    },
    "P_ORG_STRUCTURE": {
        "pos": ["Куратор всё объяснит", "Наставник поможет", "Командаға қосыл", "Жабық чатқа қосамын"],
        "neg": ["Команда выиграла матч", "Учитель объяснит тему", "Біздің команда жеңді"],
    },
    "N_FREE_ENTRY": {
        "pos": ["Участие бесплатное", "Тегін розыгрыш, жазылып, досыңды белгіле",
                "Подпишись, поставь лайк и отметь друга", "Қатысу тегін"],
        "neg": ["Тегін фриспиндер керек пе?", "Обучение бесплатно, наставник 24/7", "Участие 1000 тг"],
    },
    "N_WARNING_CONTEXT": {
        "pos": ["Осторожно, новая пирамида!", "АФМ предупреждает", "Абай болыңыз!", "Не ведитесь на обещания"],
        "neg": ["Это не пирамида, всё честно", "Пирамида емес, бәрі ашық", "Предлагаю работу на дому"],
    },
    "N_COMMERCE": {
        "pos": ["Продаю коляску, самовывоз", "Доставка по городу бесплатно", "Тапсырыс қабылдаймыз, жеткізу бар",
                "При покупке от 5000 — купон"],
        "neg": ["Взнос 1000 на каспи", "Чек скиньте в лс", "Вложи 5000 получи 10000"],
    },
    "N_BANK_RATE": {
        "pos": ["депозит 15% годовых", "Жылдық 14% мөлшерлеме", "Ставка вознаграждения до 16%",
                "ГОДОВАЯ СТАВКА — 13%"],
        "neg": ["Доход 3% в день", "120% годовых гарантированно", "Айына 40% пайда"],
    },
    "N_SKILL_EVENT": {
        "pos": ["Турнир по Dota 2", "Победитель определяется по результатам матчей", "Шахмат турнирі сенбіде",
                "Плей-офф ертең басталады"],
        "neg": ["Розыгрыш через рандомайзер", "Кім ұтты, кездейсоқ таңдаймыз", "Колесо фортуны решит"],
    },
    "N_OFFICIAL_LOTTERY": {
        "pos": ["Государственная лотерея: новый тираж", "Мемлекеттік лотерея билеттері",
                "Билеты в официальных пунктах продаж", "Ресми оператор хабарлайды"],
        "neg": ["Лотерея для своих, слот 500", "Ұтыс ойнатамыз", "Официальный магазин бренда"],
    },
}


# ------------------------------------------------------------------ config
def test_signals_yaml_is_valid():
    raw = load_raw()
    validate(raw)
    ids = [s["id"] for s in raw]
    assert len(ids) == len(set(ids))
    for s in raw:
        for field in ("name_ru", "name_kk", "weight", "track"):
            assert s.get(field) not in (None, ""), (s["id"], field)
        assert float(s["weight"]) > 0


def test_minimum_signal_set_and_core_flags():
    raw = {s["id"]: s for s in load_raw()}
    assert MINIMUM_SET <= set(raw)
    assert {sid for sid, s in raw.items() if s.get("core")} == CORE
    negatives = {sid for sid, s in raw.items() if s["track"] == "negative"}
    assert {"N_FREE_ENTRY", "N_WARNING_CONTEXT", "N_COMMERCE", "N_BANK_RATE"} <= negatives


def test_every_signal_has_cases():
    ids = {s["id"] for s in load_raw()}
    assert set(CASES) == ids
    for sid, case in CASES.items():
        assert len(case["pos"]) >= 3 and len(case["neg"]) >= 3, sid
        assert any(_KK.search(t) for t in case["pos"]), f"{sid}: needs a Kazakh positive example"
        assert any(not _KK.search(t) for t in case["pos"]), f"{sid}: needs a Russian positive example"


# --------------------------------------------------------- per-signal cases
@pytest.mark.parametrize("sid,text", [(sid, t) for sid, c in CASES.items() for t in c["pos"]])
def test_signal_fires(sid, text):
    fired = {h.id for h in detect(text)}
    assert sid in fired, f"{sid} expected in {fired}"


@pytest.mark.parametrize("sid,text", [(sid, t) for sid, c in CASES.items() for t in c["neg"]])
def test_signal_does_not_fire(sid, text):
    fired = {h.id for h in detect(text)}
    assert sid not in fired, f"{sid} must not fire; fired={fired}"


# -------------------------------------------------------------- evidence
@pytest.mark.parametrize("text", [t for c in CASES.values() for t in c["pos"]])
def test_evidence_spans_point_to_original(text):
    for hit in detect(text):
        assert hit.evidence, hit.id
        for ev in hit.evidence:
            assert 0 <= ev.start < ev.end <= len(text)
            assert text[ev.start:ev.end] == ev.text
            assert ev.source == "text"


def test_evidence_spans_on_dataset_sample():
    require(paths.DATASET_PATH)
    df = pd.read_parquet(paths.DATASET_PATH, columns=["text"]).sample(300, random_state=3)
    detector = SignalDetector()
    for text in df["text"]:
        for hit in detector.detect(text):
            for ev in hit.evidence:
                assert text[ev.start:ev.end] == ev.text and ev.text.strip()


def test_evidence_highlights_obfuscated_original():
    hits = {h.id: h for h in detect("Играй в к@зинo, вывод пришёл")}
    assert hits["K_GAMBLING_TERMS"].evidence[0].text == "к@зинo"


# ------------------------------------------------------- TZ-specific checks
def test_unreal_return_is_computed():
    hits = {h.id: h for h in detect("вложи 5000 получи 10000 за 3 дня")}
    assert "P_UNREAL_RETURN" in hits
    details = hits["P_UNREAL_RETURN"].details
    assert details["return_pct"] == 100.0
    assert details["period_days"] == 3
    assert details["monthly_pct"] >= 30


def test_bank_rate_is_not_unreal_return():
    fired = {h.id for h in detect("депозит 15% годовых")}
    assert "P_UNREAL_RETURN" not in fired
    assert "N_BANK_RATE" in fired


def test_free_giveaway_is_not_paid_entry():
    fired = {h.id for h in detect("Тегін розыгрыш, жазылып, досыңды белгіле")}
    assert "L_PAID_ENTRY" not in fired
    assert "N_FREE_ENTRY" in fired


@pytest.mark.parametrize(
    "text,low",
    [
        ("Вложил 100 000, через месяц получил 110 000", True),   # +10% per month: realistic
        ("Вложил 100 000, через месяц получил 140 000", False),  # +40% per month: unreal
        ("вложи 10к получи 13к через 2 месяца", True),           # +30% in 2 months = 15%/month
    ],
)
def test_unreal_return_threshold(text, low):
    fired = {h.id for h in detect(text)}
    assert ("P_UNREAL_RETURN" not in fired) == low


def test_negative_signals_are_marked():
    hits = {h.id: h for h in detect("Участие бесплатное, АФМ предупреждает, депозит 15% годовых")}
    for sid in ("N_FREE_ENTRY", "N_WARNING_CONTEXT", "N_BANK_RATE"):
        assert hits[sid].negative


# ----------------------------------------------------------- precision table
def test_core_signal_precision_on_train():
    require(paths.split_path("train"))
    from safenet.signals.report import CORE_MIN_PRECISION, write_report

    stats = write_report()["train"]
    core = stats[stats["core"]]
    assert set(core["signal"]) == CORE
    for row in core.itertuples():
        assert row.precision_track >= CORE_MIN_PRECISION, (row.signal, row.precision_track)
    report = (paths.REPORTS_DIR / "signals_precision.md").read_text(encoding="utf-8")
    for sid in CORE:
        assert sid in report


def test_signals_yaml_is_plain_yaml():
    data = yaml.safe_load(paths.SIGNALS_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(data["signals"], list)
