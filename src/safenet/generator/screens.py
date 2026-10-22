"""Explicit screenshot templates (payment / withdrawal screens, banners, chats, receipts).

Each spec is ``(text, gold_signals, strength)``; spec ``i`` belongs to phrase family
``i`` so every split gets its own screens. Texts contain no emoji because the
rendering font cannot draw them and the image must show exactly ``text``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from safenet.generator.templates import Template

Spec = tuple[str, tuple[str, ...], str]

SCREENS: dict[str, dict[str, list[Spec]]] = {
    "ru": {
        "lottery": [
            ("РОЗЫГРЫШ {prize}\nСлот — {amount}\nKaspi: {phone}\nКолесо в прямом эфире {date} в {time}\nОсталось {k} мест!",
             ("L_PAID_ENTRY", "C_PAY_TRANSFER", "L_RANDOM_DRAW", "C_URGENCY"), "high"),
            ("ЛОТЕРЕЯ №{n}\nПриз: {prize}\nНомер: {amount}\nПродано {sold} из {total}\nИтоги через рандомайзер",
             ("L_SERIES", "L_PAID_ENTRY", "L_ORGANIZER_PROFIT", "L_RANDOM_DRAW"), "high"),
            ("{nick}: а как участвовать?\nАдмин: взнос {amount} на карту {card}\nАдмин: потом крутим барабан в эфире\n{nick}: ок, скинула",
             ("L_PAID_ENTRY", "C_PAY_TRANSFER", "C_PERSONAL_CARD", "L_RANDOM_DRAW"), "high"),
            ("Список участников\n1. {nick} - оплачено\n2. {nick2} - оплачено\n3. {nick3} - ждём оплату\nВзнос {amount}\nРозыгрыш {prize} в {day}",
             ("L_PAID_ENTRY",), "medium"),
            ("Kaspi Перевод\nСумма: {amount}\nПолучатель: {nick}\nКомментарий: за слот №{n}\nСтатус: выполнен",
             ("C_PAY_TRANSFER", "L_PAID_ENTRY"), "medium"),
        ],
        "casino": [
            ("Казино {casino}\nВывод средств\nСумма: {win}\nСтатус: Выполнено\nПромокод для подписчиков: {promo}",
             ("K_GAMBLING_TERMS", "K_WIN_BRAG", "K_PROMO_BONUS"), "high"),
            ("Баланс: {win}\nСтавка: {amount}\nВыигрыш x{mult}\nСлот Golden Barys\nЗеркало: {domain}",
             ("K_GAMBLING_TERMS", "K_WIN_BRAG", "K_MIRROR"), "high"),
            ("БОНУС 200% К ДЕПОЗИТУ\nПромокод {promo}\n{fs} фриспинов\nРегистрация: {short_link}",
             ("K_PROMO_BONUS", "K_GAMBLING_TERMS", "C_SHORTLINK"), "high"),
            ("История операций\nПополнение -{amount}\nВывод +{win}\nСтатус: зачислено",
             ("K_WIN_BRAG",), "medium"),
            ("{nick}: брат, где играешь?\n{nick2}: в {casino}, ссылка в био\n{nick2}: вчера поднял {win}",
             ("K_WIN_BRAG", "C_SHORTLINK"), "medium"),
        ],
        "pyramid": [
            ("Выплата получена!\n+{ret}\nВложено: {invest}\nСрок: {days} дней\nПроект {project}",
             ("P_PAYOUT_PROOF", "P_UNREAL_RETURN"), "high"),
            ("Куратор: вложи {invest}, через {days} дней получишь {ret}\nЯ: а это точно?\nКуратор: гарантированно, без риска\nКуратор: карта {card}",
             ("P_ORG_STRUCTURE", "P_UNREAL_RETURN", "P_GUARANTEE", "C_PAY_TRANSFER", "C_PERSONAL_CARD"), "high"),
            ("ТАРИФЫ\nСтарт: вложи {invest} - получи {ret}\nДоход {pct_day}% в день\n1-я линия 10%, 2-я линия 5%\nОсталось {k} мест",
             ("P_UNREAL_RETURN", "P_REFERRAL_TIERS", "C_URGENCY"), "high"),
            ("Отзывы участников\n- Вывела уже второй раз ({nick})\n- Куратор помог, всё честно ({nick2})\n- Пассивный доход реально работает ({nick3})",
             ("P_PAYOUT_PROOF", "P_ORG_STRUCTURE"), "medium"),
            ("Kaspi Перевод\nСумма: {invest}\nПолучатель: {nick}\nКомментарий: вклад в команду\nСтатус: выполнен",
             ("C_PAY_TRANSFER", "P_ORG_STRUCTURE"), "medium"),
        ],
        "receipt": [
            ("ЧЕК №{n}\n{item} - {price}\nИтого: {price}\nОплата: Kaspi QR\nСпасибо за покупку!",
             ("N_COMMERCE", "C_PAY_TRANSFER"), "low"),
            ("Магазин {shop}\nТовар: {item}\nКол-во: 1\nСумма: {price}\nОплачено картой\nДоставка: {city}",
             ("N_COMMERCE",), "low"),
            ("Kaspi Перевод\nСумма: {price}\nПолучатель: {nick}\nКомментарий: за {item}\nСтатус: выполнен",
             ("C_PAY_TRANSFER", "N_COMMERCE"), "low"),
            ("Заказ №{n} оформлен\nТовар: {item}\nСумма к оплате: {price}\nСпособ оплаты: Kaspi QR\nДоставка курьером",
             ("N_COMMERCE", "C_PAY_TRANSFER"), "low"),
            ("Ваш депозит\nСумма: {amount_dep}\nСтавка: {pct_year}% годовых\nВознаграждение начисляется ежемесячно",
             ("N_BANK_RATE",), "low"),
        ],
        "contest": [
            ("КОНКУРС\nПриз: {prize_small}\nУчастие бесплатное\nПодпишись + лайк + отметь друга\nИтоги {date} рандомайзером",
             ("N_FREE_ENTRY", "L_RANDOM_DRAW"), "low"),
            ("РОЗЫГРЫШ\n{prize_small}\nУсловия:\n1. Подписка на {nick}\n2. Лайк\n3. Отметить друга\nБесплатно!",
             ("N_FREE_ENTRY",), "low"),
            ("Итоги конкурса\nПобедитель: {nick}\nВыбран через random.org\nСпасибо всем за участие!",
             ("L_RANDOM_DRAW",), "low"),
            ("Акция {shop}\nПри покупке от {price} - купон\nРозыгрыш {prize_small} {date}\nДополнительной оплаты нет",
             ("N_COMMERCE", "N_FREE_ENTRY"), "low"),
            ("Турнир по {game}\nПризовой фонд {prize_pool}\nРегистрация до {date}\nПобедитель по результатам матчей",
             ("N_SKILL_EVENT",), "low"),
        ],
    },
    "kk": {
        "lottery": [
            ("ҰТЫС {prize}\nСлот — {amount}\nKaspi: {phone}\nТікелей эфирде дөңгелек {date}, сағат {time}\n{k} орын қалды!",
             ("L_PAID_ENTRY", "C_PAY_TRANSFER", "L_RANDOM_DRAW", "C_URGENCY"), "high"),
            ("ЛОТЕРЕЯ №{n}\nЖүлде: {prize}\nНөмір: {amount}\n{total} нөмірдің {sold} сатылды\nҚорытынды рандомайзер арқылы",
             ("L_SERIES", "L_PAID_ENTRY", "L_ORGANIZER_PROFIT", "L_RANDOM_DRAW"), "high"),
            ("{nick}: қалай қатысуға болады?\nӘкімші: жарна {amount}, картаға {card}\nӘкімші: кейін эфирде барабан айналдырамыз\n{nick}: жарайды, салдым",
             ("L_PAID_ENTRY", "C_PAY_TRANSFER", "C_PERSONAL_CARD", "L_RANDOM_DRAW"), "high"),
            ("Қатысушылар тізімі\n1. {nick} - төленді\n2. {nick2} - төленді\n3. {nick3} - төлем күтілуде\nЖарна {amount}\n{prize} ұтысы {day} күні",
             ("L_PAID_ENTRY",), "medium"),
            ("Kaspi Аударым\nСомасы: {amount}\nАлушы: {nick}\nТүсініктеме: №{n} слот үшін\nМәртебесі: орындалды",
             ("C_PAY_TRANSFER", "L_PAID_ENTRY"), "medium"),
        ],
        "casino": [
            ("{casino} казиносы\nҚаражат шығару\nСомасы: {win}\nМәртебесі: Орындалды\nЖазылушыларға промокод: {promo}",
             ("K_GAMBLING_TERMS", "K_WIN_BRAG", "K_PROMO_BONUS"), "high"),
            ("Баланс: {win}\nСтавка: {amount}\nҰтыс x{mult}\nGolden Barys слоты\nАйна: {domain}",
             ("K_GAMBLING_TERMS", "K_WIN_BRAG", "K_MIRROR"), "high"),
            ("ДЕПОЗИТКЕ 200% БОНУС\nПромокод {promo}\n{fs} фриспин\nТіркелу: {short_link}",
             ("K_PROMO_BONUS", "K_GAMBLING_TERMS", "C_SHORTLINK"), "high"),
            ("Операциялар тарихы\nТолықтыру -{amount}\nВывод +{win}\nМәртебесі: есептелді",
             ("K_WIN_BRAG",), "medium"),
            ("{nick}: аға, қайда ойнайсыз?\n{nick2}: {casino} сайтында, сілтеме биода\n{nick2}: кеше {win} ұттым",
             ("K_WIN_BRAG", "C_SHORTLINK"), "medium"),
        ],
        "pyramid": [
            ("Төлем алынды!\n+{ret}\nСалынды: {invest}\nМерзімі: {days} күн\n{project} жобасы",
             ("P_PAYOUT_PROOF", "P_UNREAL_RETURN"), "high"),
            ("Куратор: {invest} сал, {days} күннен кейін {ret} аласың\nМен: бұл нақты ма?\nКуратор: кепілді, тәуекелсіз\nКуратор: карта {card}",
             ("P_ORG_STRUCTURE", "P_UNREAL_RETURN", "P_GUARANTEE", "C_PAY_TRANSFER", "C_PERSONAL_CARD"), "high"),
            ("ТАРИФТЕР\nСтарт: {invest} сал - {ret} ал\nКүніне {pct_day}% табыс\n1-деңгей 10%, 2-деңгей 5%\n{k} орын қалды",
             ("P_UNREAL_RETURN", "P_REFERRAL_TIERS", "C_URGENCY"), "high"),
            ("Қатысушылар пікірі\n- Екінші рет шығардым ({nick})\n- Куратор көмектесті, бәрі әділ ({nick2})\n- Пассивті табыс шынымен бар ({nick3})",
             ("P_PAYOUT_PROOF", "P_ORG_STRUCTURE"), "medium"),
            ("Kaspi Аударым\nСомасы: {invest}\nАлушы: {nick}\nТүсініктеме: командаға салым\nМәртебесі: орындалды",
             ("C_PAY_TRANSFER", "P_ORG_STRUCTURE"), "medium"),
        ],
        "receipt": [
            ("ЧЕК №{n}\n{item} - {price}\nБарлығы: {price}\nТөлем: Kaspi QR\nСатып алғаныңызға рахмет!",
             ("N_COMMERCE", "C_PAY_TRANSFER"), "low"),
            ("{shop} дүкені\nТауар: {item}\nСаны: 1\nСомасы: {price}\nКартамен төленді\nЖеткізу: {city}",
             ("N_COMMERCE",), "low"),
            ("Kaspi Аударым\nСомасы: {price}\nАлушы: {nick}\nТүсініктеме: {item} үшін\nМәртебесі: орындалды",
             ("C_PAY_TRANSFER", "N_COMMERCE"), "low"),
            ("№{n} тапсырыс рәсімделді\nТауар: {item}\nТөлеуге: {price}\nТөлем тәсілі: Kaspi QR\nКурьермен жеткізу",
             ("N_COMMERCE", "C_PAY_TRANSFER"), "low"),
            ("Сіздің депозитіңіз\nСомасы: {amount_dep}\nМөлшерлеме: жылдық {pct_year}%\nСыйақы ай сайын есептеледі",
             ("N_BANK_RATE",), "low"),
        ],
        "contest": [
            ("БАЙҚАУ\nЖүлде: {prize_small}\nҚатысу тегін\nЖазыл + лайк + досыңды белгіле\nҚорытынды {date} рандомайзермен",
             ("N_FREE_ENTRY", "L_RANDOM_DRAW"), "low"),
            ("ҰТЫС\n{prize_small}\nШарттары:\n1. {nick} парақшасына жазылу\n2. Лайк\n3. Досыңды белгілеу\nТегін!",
             ("N_FREE_ENTRY",), "low"),
            ("Байқау қорытындысы\nЖеңімпаз: {nick}\nrandom.org арқылы таңдалды\nҚатысқандардың бәріне рахмет!",
             ("L_RANDOM_DRAW",), "low"),
            ("{shop} акциясы\n{price} бастап сатып алсаңыз - купон\n{prize_small} ұтысы {date}\nҚосымша төлем жоқ",
             ("N_COMMERCE", "N_FREE_ENTRY"), "low"),
            ("{game} бойынша турнир\nЖүлде қоры {prize_pool}\nТіркелу {date} дейін\nЖеңімпаз матч нәтижесі бойынша",
             ("N_SKILL_EVENT",), "low"),
        ],
    },
}


def screen_key(template: "Template") -> str:
    if template.label != "none":
        return template.label
    return "receipt" if (template.screen_index or 0) % 2 == 0 else "contest"


def screen_spec(template: "Template", lang: str) -> dict:
    """Return ``{"text", "signals", "strength"}`` for a screenshot template."""
    specs = SCREENS[lang][screen_key(template)]
    text, signals, strength = specs[template.fold % len(specs)]
    return {"text": text, "signals": signals, "strength": strength}
