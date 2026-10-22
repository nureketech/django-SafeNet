"""Human-readable explanations in Russian and Kazakh.

Wording follows the TZ: the system reports *signs* («признаки», «белгілер»), a risk
level and the need for a specialist's review — never a verdict.
"""

from __future__ import annotations

from safenet.common import find_forbidden
from safenet.scoring.score import Breakdown

TRACK_NAMES = {
    "ru": {
        "lottery": "платный розыгрыш (лотерея)",
        "casino": "скрытая реклама онлайн-казино",
        "pyramid": "финансовая пирамида",
        "none": "схема не определена",
    },
    "kk": {
        "lottery": "ақылы ұтыс (лотерея)",
        "casino": "онлайн-казиноның жасырын жарнамасы",
        "pyramid": "қаржы пирамидасы",
        "none": "схема анықталмады",
    },
}
LEVEL_NAMES = {
    "ru": {"low": "низкий", "medium": "средний", "high": "высокий"},
    "kk": {"low": "төмен", "medium": "орташа", "high": "жоғары"},
}
ADJUSTMENT_NAMES = {
    "rule_cap": ("Ограничение суммы правил (максимум 60)", "Ережелер ұпайының шегі (ең көбі 60)"),
    "clip_low": ("Оценка не может быть меньше 0", "Ұпай 0-ден төмен болмайды"),
    "clip_high": ("Оценка не может быть больше 100", "Ұпай 100-ден аспайды"),
    "no_core": ("Нет ключевого признака: уровень «высокий» не присваивается",
                "Негізгі белгі жоқ: «жоғары» деңгей берілмейді"),
    "no_evidence": ("Нет признаков с доказательством: уровень ниже «среднего»",
                    "Дәлелі бар белгі жоқ: деңгей «орташадан» төмен"),
}


def adjustment_names(adj_id: str) -> tuple[str, str]:
    return ADJUSTMENT_NAMES.get(adj_id, (adj_id, adj_id))


def _join(names: list[str]) -> str:
    return "; ".join(names)


def explain(b: Breakdown, lang: str) -> str:
    """Explanation of ``b`` in ``lang`` (``ru`` or ``kk``)."""
    ru = lang == "ru"
    ordered = sorted(b.positive, key=lambda h: -h.weight)
    names = [h.name_ru if ru else h.name_kk for h in ordered]
    neg_names = [h.name_ru if ru else h.name_kk for h in b.negative]
    combo_names = [c["name_ru"] if ru else c["name_kk"] for c in b.combos]
    level = LEVEL_NAMES[lang][b.level]
    track = TRACK_NAMES[lang][b.primary_track]
    parts: list[str] = []
    if names:
        parts.append(f"{'Обнаружены признаки' if ru else 'Анықталған белгілер'} ({len(names)}): {_join(names)}.")
        if combo_names:
            parts.append(f"{'Сочетание признаков' if ru else 'Белгілердің үйлесімі'}: {_join(combo_names)}.")
        if b.primary_track != "none" and b.level != "low":
            tail = ""
            if b.ml_available:
                prob = round(b.track_probability.get(b.primary_track, 0.0) * 100)
                tail = f" ({'вероятность модели' if ru else 'модель ықтималдығы'} {prob}%)"
            parts.append(f"{'Наиболее вероятный тип схемы' if ru else 'Схеманың ең ықтимал түрі'}: {track}{tail}.")
    else:
        parts.append("Существенных признаков схем не обнаружено." if ru else "Схемалардың елеулі белгілері табылған жоқ.")
    if neg_names:
        parts.append(f"{'Снижающие факторы' if ru else 'Тәуекелді төмендететін факторлар'}: {_join(neg_names)}.")
    for adj in b.adjustments:
        if adj.id in ("no_core", "no_evidence"):
            parts.append(adjustment_names(adj.id)[0 if ru else 1] + ".")
    if b.network_contribution > 0:
        parts.append("Учтены связи с другими аккаунтами (общие реквизиты)." if ru
                     else "Басқа аккаунттармен байланыстар ескерілді (ортақ реквизиттер).")
    parts.append(f"{'Уровень риска' if ru else 'Тәуекел деңгейі'}: {level} ({b.rounded}/100).")
    text = " ".join(parts)
    assert not find_forbidden(text), text
    return text
