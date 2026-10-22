"""Stage 5: risk scoring, AnalysisResult, explanations and thresholds."""

from __future__ import annotations

import itertools
import json
import random

import pandas as pd
import pytest

from safenet import DISCLAIMER_KK, DISCLAIMER_RU, paths
from safenet.common import find_forbidden
from safenet.pipeline import analyze, analyze_many
from safenet.scoring import compute_score, load_config
from safenet.signals import Evidence, SignalHit, get_detector
from tests.conftest import require

TZ_FIELDS = {
    "id", "risk_score", "risk_level", "tracks", "signals", "negative_signals", "ml", "network",
    "explanation_ru", "explanation_kk", "disclaimer_kk", "disclaimer_ru",
}

CLASSIC = {
    "lottery": [
        "🎉 РОЗЫГРЫШ iPhone 15! Участие — 1500 тг, оплата на Kaspi +7 701 234 56 78. Победителя выберем "
        "рандомайзером. Продано 80 из 100 слотов!",
        "Ұтыс! Жүлде — көлік. Қатысу 2000 теңге, каспиге аударыңыз. Тікелей эфирде дөңгелек айналдырамыз. "
        "100 нөмірдің 70 сатылды",
    ],
    "casino": [
        "Мой занос — 340 тыс за вечер! Играю в казино Spinomad, промокод SPIN200 — бонус 200% к депозиту, "
        "регистрация https://spinomad.test/?ref=ab1",
        "Кеше 500 мың ұттым 😎 Altynslot казиносында ойнаймын, ALTYN777 промокоды — депозитке 150% бонус, "
        "сілтеме биода",
    ],
    "pyramid": [
        "Вложи 10 000 — через неделю получишь 30 000! Гарантированно, без риска. С каждого приглашённого 10%. "
        "Осталось 3 места, пиши в лс",
        "50 мың сал — 7 күннен кейін 150 мың ал! Кепілді табыс, 1-деңгей 10%, 2-деңгей 5%. Тек бүгін, жеке жаз",
    ],
}
FREE_GIVEAWAYS = [
    "РОЗЫГРЫШ 🎁 Разыгрываем торт! Условия: подпишись, поставь лайк, отметь друга. Участие бесплатное, "
    "победителя выберем рандомайзером",
    "Тегін розыгрыш! Жазылып, досыңды белгіле, жеңімпазды рандомайзермен таңдаймыз",
    "Конкурс в честь Наурыза: подписка + лайк, ничего платить не нужно. Итоги в пятницу через random.org",
]
COMMERCE = [
    "Продаю торты на заказ, 5000 тг, оплата Kaspi QR, доставка по городу",
    "Балалар күртешелері қолда бар, бағасы 12 000 тг, Kaspi аударым, жеткізу бар",
    "Новое поступление: кроссовки, 25 000 тг. Оплата Kaspi перевод или наличными, самовывоз",
]
WARNINGS = [
    "АФМ предупреждает: в WhatsApp обещают «вложи 10 000 — получи 30 000 за неделю» и просят перевести деньги "
    "на личную карту. Не ведитесь!",
    "Абай болыңыз! Жаңа пирамида күніне 10% табыс уәде етеді, кепілді деп алдайды. Ақша аудармаңыз",
    "Осторожно, финансовая пирамида: обещают х2 за неделю, гарантированно и без риска. Сообщайте в полицию",
]
WEAK_NON_CORE = ["Пиши в лс", "Ссылка в био", "Осталось 3 места!", "Куратор всё объяснит", "Тек бүгін!"]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def sample_results():
    require(paths.DATASET_PATH)
    require(paths.MODEL_PATH, "make train")
    df = pd.read_parquet(paths.DATASET_PATH).sample(200, random_state=11)
    items = [{"id": r.id, "text": r.text, "source_type": r.source_type} for r in df.itertuples()]
    return analyze_many(items)


def _all_strings(obj, skip=("text",)):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield from _all_strings(v, skip)
    elif isinstance(obj, list):
        for v in obj:
            yield from _all_strings(v, skip)
    elif isinstance(obj, str):
        yield obj


# --------------------------------------------------------------- format
def test_result_has_tz_fields():
    require(paths.MODEL_PATH, "make train")
    data = analyze(CLASSIC["pyramid"][0]).model_dump()
    assert TZ_FIELDS <= set(data)
    sig = data["signals"][0]
    assert {"id", "name_ru", "name_kk", "contribution", "evidence"} <= set(sig)
    assert {"text", "start", "end", "source"} <= set(sig["evidence"][0])
    assert {"top_class", "probability", "contribution"} <= set(data["ml"])
    assert {"cluster_id", "cluster_size", "contribution"} <= set(data["network"])
    json.dumps(data, ensure_ascii=False)


def test_scores_in_range_and_levels_consistent(sample_results, cfg):
    for r in sample_results:
        assert 0 <= r.risk_score <= 100
        expected = "high" if r.risk_score >= cfg.high else "medium" if r.risk_score >= cfg.medium else "low"
        assert r.risk_level == expected


def test_medium_and_high_have_evidence(sample_results):
    flagged = [r for r in sample_results if r.risk_level in ("medium", "high")]
    assert flagged
    for r in flagged:
        assert r.signals, r.id
        for s in r.signals:
            assert s.evidence, (r.id, s.id)
            for e in s.evidence:
                assert r.text[e.start:e.end] == e.text


def test_disclaimer_always_present(sample_results):
    for r in sample_results:
        assert r.disclaimer_kk == DISCLAIMER_KK
        assert r.disclaimer_ru == DISCLAIMER_RU


def test_no_forbidden_words_in_results(sample_results):
    for r in sample_results:
        for s in _all_strings(r.model_dump(), skip=()):
            assert not find_forbidden(s), s


def test_no_forbidden_words_even_if_input_has_them():
    require(paths.MODEL_PATH, "make train")
    r = analyze("Мошенники обещают 30% в день, это незаконно! Вложи 5000 получи 10000 за 3 дня")
    for s in _all_strings(r.model_dump()):
        assert not find_forbidden(s), s


def test_contributions_sum_to_score(sample_results):
    for r in sample_results:
        assert abs(r.contributions_total() - r.risk_score) <= 1.0, (r.id, r.contributions_total(), r.risk_score)


# ----------------------------------------------------------- invariants
def _hits(ids: list[str]) -> list[SignalHit]:
    det = get_detector()
    out = []
    for sid in ids:
        s = det.by_id[sid]
        out.append(SignalHit(s.id, s.track, s.name_ru, s.name_kk, s.weight, s.core, s.negative,
                             [Evidence("x", 0, 1)]))
    return out


def test_adding_positive_signal_never_lowers_score(cfg):
    det = get_detector()
    positive = [s.id for s in det.signals if not s.negative]
    negative = [s.id for s in det.signals if s.negative]
    rng = random.Random(0)
    probas = [None, {"lottery": 0.1, "casino": 0.7, "pyramid": 0.1, "none": 0.1},
              {"lottery": 0.0, "casino": 0.0, "pyramid": 0.05, "none": 0.95}]
    for _ in range(300):
        base = rng.sample(positive, rng.randint(0, 5))
        negs = rng.sample(negative, rng.randint(0, 2))
        extra = rng.choice([s for s in positive if s not in base])
        proba = rng.choice(probas)
        net = rng.choice([0.0, 5.0, 10.0])
        before = compute_score(_hits(base + negs), proba, net, cfg).score
        after = compute_score(_hits(base + [extra] + negs), proba, net, cfg).score
        assert after >= before - 1e-9, (base, extra, negs, before, after)


def test_no_core_signal_is_never_high(cfg):
    det = get_detector()
    non_core = [s.id for s in det.signals if not s.negative and not s.core]
    proba = {"lottery": 1.0, "casino": 0.0, "pyramid": 0.0, "none": 0.0}
    for k in (1, 3, len(non_core)):
        for combo in itertools.islice(itertools.combinations(non_core, k), 50):
            b = compute_score(_hits(list(combo)), proba, 10.0, cfg)
            assert b.level != "high" and b.score <= cfg.no_core_cap


@pytest.mark.parametrize("text", FREE_GIVEAWAYS)
def test_free_giveaway_is_low(text):
    require(paths.MODEL_PATH, "make train")
    assert analyze(text).risk_level == "low"


@pytest.mark.parametrize("text", COMMERCE)
def test_kaspi_commerce_is_low(text):
    require(paths.MODEL_PATH, "make train")
    assert analyze(text).risk_level == "low"


@pytest.mark.parametrize("text", WARNINGS)
def test_pyramid_warning_is_not_high(text):
    require(paths.MODEL_PATH, "make train")
    assert analyze(text).risk_level != "high"


@pytest.mark.parametrize("text", WEAK_NON_CORE)
def test_single_weak_signal_is_not_high(text):
    require(paths.MODEL_PATH, "make train")
    r = analyze(text)
    assert r.risk_level != "high"
    assert len(r.signals) <= 1


@pytest.mark.parametrize("track,text", [(t, x) for t, xs in CLASSIC.items() for x in xs])
def test_classic_cases_are_high(track, text):
    require(paths.MODEL_PATH, "make train")
    r = analyze(text)
    assert r.risk_level == "high", (r.risk_score, r.explanation_ru)
    assert r.primary_track == track


def test_no_signals_means_low_even_if_ml_is_sure(cfg):
    b = compute_score([], {"lottery": 0.99, "casino": 0.0, "pyramid": 0.0, "none": 0.01}, 0.0, cfg)
    assert b.level == "low"
    assert any(a.id == "no_evidence" for a in b.adjustments)


def test_works_without_ml_and_marks_transcript_evidence():
    r = analyze({"text": "регайтесь по моей ссылке в описании промокод SPIN200 бонус 200% к депозиту казино",
                 "source_type": "stream_transcript"}, use_ml=False)
    assert r.ml.available is False and r.ml.contribution == 0
    assert r.signals and all(e.source == "transcript" for s in r.signals for e in s.evidence)


def test_explanations_are_bilingual():
    require(paths.MODEL_PATH, "make train")
    r = analyze(CLASSIC["lottery"][0])
    assert "Обнаружены признаки" in r.explanation_ru and "Уровень риска" in r.explanation_ru
    assert "Анықталған белгілер" in r.explanation_kk and "Тәуекел деңгейі" in r.explanation_kk
    low = analyze("Кто идёт завтра на футбол?")
    assert "не обнаружено" in low.explanation_ru and "табылған жоқ" in low.explanation_kk


def test_card_numbers_are_masked_in_result():
    r = analyze("Перевод на мою карту 4400 4302 1234 5678, взнос 1000 тг", use_ml=False)
    assert r.entities["cards"] == ["4400 43** **** 5678"]


# ------------------------------------------------------------ thresholds
def test_thresholds_file(cfg):
    require(paths.THRESHOLDS_PATH, "make train")
    data = json.loads(paths.THRESHOLDS_PATH.read_text(encoding="utf-8"))
    assert data["source"] == "val"
    assert cfg.medium == data["medium"] and cfg.high == data["high"]
    assert cfg.no_core_cap < cfg.high
    assert 0 < cfg.medium < cfg.high <= 100
    assert (paths.REPORTS_DIR / "thresholds.md").exists()


def test_val_precision_and_hard_negative_fpr_with_tuned_thresholds():
    require(paths.split_path("val"))
    require(paths.THRESHOLDS_PATH, "make train")
    val = pd.read_parquet(paths.split_path("val"))
    results = analyze_many({"text": t, "source_type": s} for t, s in zip(val["text"], val["source_type"]))
    flagged = pd.Series([r.risk_level != "low" for r in results], index=val.index)
    positive = val["label"] != "none"
    precision = (flagged & positive).sum() / flagged.sum()
    hn = val["is_hard_negative"]
    hn_fpr = (flagged & hn).sum() / hn.sum()
    assert precision >= 0.85, precision
    assert hn_fpr <= 0.10, hn_fpr
