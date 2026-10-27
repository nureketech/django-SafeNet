"""Stage 16 (outside the TZ): links that an ordinary analyst can use.

* one field: a t.me link is enough — the post's text, hidden links, channel and time are read from Telegram;
* «Арнаны толық тексеру» checks a whole channel (latest posts + description) in one click;
* the case card and the network page say in plain words who is linked with whom and why;
* the queue marks cases that belong to a network and can show only them.

Telegram pages are replaced by small hand-written pages of the same structure: the tests never go online.
"""

from __future__ import annotations

import re
import urllib.error

import pytest
from django.urls import reverse

from app.dashboard import telegram as tg
from app.dashboard.models import Case
from safenet import paths
from safenet.graph.entities import graph_entities
from safenet.preprocess import normalize
from tests.conftest import require

SHARED = "https://spin-route-77.click/c4eda236c?utm_source=smm&utm_medium=telegram&country=kz"


def message(channel: str, number: int, text_html: str | None, *, buttons=(), photo: bool = False,
            when: str = "2026-10-22T14:02:00+00:00") -> str:
    parts = [f'<div class="tgme_widget_message" data-post="{channel}/{number}">']
    if photo:
        parts.append('<a class="tgme_widget_message_photo_wrap" style="background-image:url(\'x.jpg\')"></a>')
    if text_html is not None:
        parts.append(f'<div class="tgme_widget_message_text js-message_text" dir="auto">{text_html}</div>')
    parts += [f'<a class="tgme_widget_message_inline_button url_button" href="{b}"><span>Play</span></a>'
              for b in buttons]
    parts.append(f'<a class="tgme_widget_message_date" href="https://t.me/{channel}/{number}">'
                 f'<time datetime="{when}" class="time">14:02</time></a></div>')
    return "".join(parts)


def channel_page(channel: str, title: str, description: str, messages: list[str]) -> str:
    return (f'<html><div class="tgme_channel_info"><div class="tgme_channel_info_header_title">'
            f'<span dir="auto">{title}</span></div><div class="tgme_channel_info_description">{description}</div>'
            f'</div>{"".join(messages)}</html>')


CASINO_A = ("🎰 Бонус-код <b>WILD100</b> добавит 100 FS к депозиту!<br>Депозит от 5000 ₸, ставка 40 ₸. "
            f'<a href="{SHARED}&amp;bonus-code=WILD100">Забрать бонус</a>')
CASINO_B = ("Фриспины каждый день, вывод моментальный! Промокод <b>LUCKY77</b>, "
            f'<a href="{SHARED.replace("c4eda236c", "c6d7237b1")}">играть</a>')
PAGES = {
    "https://t.me/alpha_casino/12?embed=1&mode=tme": message(
        "alpha_casino", 12, CASINO_A, buttons=["https://t.me/alpha_casino/12", SHARED], photo=True),
    "https://t.me/alpha_casino/13?embed=1&mode=tme": '<div class="tgme_widget_message_error">Post not found</div>',
    "https://t.me/alpha_casino/14?embed=1&mode=tme": message("alpha_casino", 14, None, photo=True),
    "https://t.me/s/alpha_casino": channel_page(
        "alpha_casino", "ALPHA Official 🇰🇿", 'Актуальное зеркало: <a href="https://alpha-mirror-01.com/x">тут</a>',
        [message("alpha_casino", 12, CASINO_A), message("alpha_casino", 14, None, photo=True),
         message("alpha_casino", 15, "Розыгрыш 3000 FS для подписчиков! Регистрируйся и вводи код SPIN15")]),
    "https://t.me/s/beta_casino": channel_page(
        "beta_casino", "BETA Casino", "Бонусы и зеркала", [message("beta_casino", 7, CASINO_B)]),
    "https://t.me/s/nobody_here": "<html><title>Telegram: Contact @nobody_here</title></html>",
}


@pytest.fixture(scope="module", autouse=True)
def _artifacts():
    require(paths.MODEL_PATH, "make train")


@pytest.fixture(autouse=True)
def _offline(monkeypatch, settings):
    settings.SAFENET_FETCH_ENABLED = True

    def fake_get(url: str) -> str:
        if url not in PAGES:
            raise tg.FetchError("fetch_not_found")
        return PAGES[url]

    monkeypatch.setattr(tg, "_get", fake_get)


@pytest.fixture
def signed_in(client, analyst):
    client.force_login(analyst)
    return client


def html(response) -> str:
    return response.content.decode("utf-8")


def post_check(client, link: str, action: str = "check", **extra):
    return client.post(reverse("check"), {"source_url": link, "text": "", "source_type": "post", "save": "1",
                                          "action": action, **extra})


# ------------------------------------------------------------------ reading t.me
@pytest.mark.parametrize("link,expected", [
    ("https://t.me/alpha_casino/12", ("alpha_casino", 12)),
    ("t.me/s/Alpha_Casino/12?single", ("alpha_casino", 12)),
    ("telegram.me/alpha_casino", ("alpha_casino", None)),
])
def test_parse_link(link, expected):
    assert tg.parse_link(link) == expected


@pytest.mark.parametrize("link", ["https://t.me/joinchat/abc", "https://t.me/+AbCd", "https://evil.example/t.me/a/1",
                                  "https://instagram.com/p/x", "t.me/abc/1"])
def test_parse_link_rejects(link):
    with pytest.raises(tg.FetchError) as exc:
        tg.parse_link(link)
    assert exc.value.code == "fetch_bad_link"


def test_post_text_hidden_links_and_time():
    post = tg.fetch_post("alpha_casino", 12)
    assert post.visible_text.startswith("🎰 Бонус-код WILD100 добавит 100 FS") and "\n" in post.visible_text
    # the link under «Забрать бонус» and the button; utm_* removed, Telegram's own link dropped, no duplicates
    assert post.links == ["https://spin-route-77.click/c4eda236c?country=kz&bonus-code=WILD100",
                          "https://spin-route-77.click/c4eda236c?country=kz"]
    assert post.text.endswith("🔗 https://spin-route-77.click/c4eda236c?country=kz")
    assert post.published_at.isoformat() == "2026-10-22T14:02:00+00:00" and post.has_photo
    assert (post.account, post.url, post.case_id) == ("@alpha_casino", "https://t.me/alpha_casino/12",
                                                      "tg-alpha_casino-12")
    kinds = {(e.kind, e.label) for e in graph_entities(normalize(post.text))}
    assert ("domain", "spin-route-77.click") in kinds and ("promo", "WILD100") in kinds


@pytest.mark.parametrize("post_id,code", [(13, "fetch_not_found"), (14, "fetch_empty"), (99, "fetch_not_found")])
def test_post_errors(post_id, code):
    with pytest.raises(tg.FetchError) as exc:
        tg.fetch_post("alpha_casino", post_id)
    assert exc.value.code == code


def test_channel_page():
    channel = tg.fetch_channel("alpha_casino")
    assert channel.title == "ALPHA Official 🇰🇿" and channel.description == "Актуальное зеркало: тут"
    assert channel.description_links == ["https://alpha-mirror-01.com/x"]
    assert [p.post_id for p in channel.posts] == [12, 15]  # the photo-only post has nothing to read
    with pytest.raises(tg.FetchError) as exc:
        tg.fetch_channel("nobody_here")
    assert exc.value.code == "fetch_not_found"


def test_network_failure_and_switch(monkeypatch, settings):
    monkeypatch.undo()
    settings.SAFENET_FETCH_ENABLED = True

    def offline(*args, **kwargs):
        raise urllib.error.URLError("no route")

    monkeypatch.setattr(tg.urllib.request, "urlopen", offline)
    with pytest.raises(tg.FetchError) as exc:
        tg.fetch_post("alpha_casino", 12)
    assert exc.value.code == "fetch_failed"
    settings.SAFENET_FETCH_ENABLED = False
    with pytest.raises(tg.FetchError) as exc:
        tg.fetch_channel("alpha_casino")
    assert exc.value.code == "fetch_disabled"


# ------------------------------------------------------------------ one link on /check/
def test_check_form_leads_with_the_link(signed_in):
    page = html(signed_in.get(reverse("check")))
    assert page.index('data-role="link-field"') < page.index('name="text"')
    assert '<details class="context" data-role="context">' in page  # account and chat folded away
    assert 'data-role="check-channel"' in page


def test_check_by_link_only(signed_in):
    r = post_check(signed_in, "https://t.me/alpha_casino/12")
    assert r.status_code == 200
    page = html(r)
    assert 'data-role="fetched"' in page and "WILD100" in page
    case = Case.objects.get()
    assert case.id == "tg-alpha_casino-12" and case.account_id == "@alpha_casino" and case.platform == "telegram"
    assert case.source_url == "https://t.me/alpha_casino/12" and case.published_at.year == 2026
    assert "spin-route-77.click" in case.text
    page = html(signed_in.get(reverse("case", args=[case.id])))
    assert "22.10.2026" in page  # publication time on the card
    # the same post checked again stays one case
    post_check(signed_in, "t.me/s/alpha_casino/12")
    assert Case.objects.count() == 1


@pytest.mark.parametrize("link,status,key", [
    ("https://t.me/alpha_casino", 422, "fetch_need_post"),
    ("https://t.me/alpha_casino/13", 422, "fetch_not_found"),
    ("https://t.me/alpha_casino/14", 422, "fetch_empty"),
])
def test_check_by_link_errors(signed_in, link, status, key):
    from app.dashboard.i18n import UI

    r = post_check(signed_in, link)
    assert r.status_code == status and UI["kk"][key] in html(r)
    assert not Case.objects.exists()


def test_pasted_text_wins_over_fetching(signed_in, monkeypatch):
    monkeypatch.setattr(tg, "_get", lambda url: pytest.fail("must not fetch when text is given"))
    post_check(signed_in, "https://t.me/alpha_casino/12", text="Промокод MANUAL1 в описании канала")
    case = Case.objects.get()
    assert case.id == "tg-alpha_casino-12" and "MANUAL1" in case.text


def test_link_check_absorbs_an_earlier_pasted_copy(signed_in):
    from app.dashboard.models import Feedback

    visible = tg.fetch_post("alpha_casino", 12).visible_text
    signed_in.post(reverse("check"), {"text": visible.replace("\n", "\r\n"), "source_type": "post", "save": "1"})
    manual = Case.objects.get()
    signed_in.post(reverse("case_feedback", args=[manual.id]), {"verdict": "confirmed", "comment": "алдында"})
    page = html(post_check(signed_in, "https://t.me/alpha_casino/12"))
    assert list(Case.objects.values_list("id", flat=True)) == ["tg-alpha_casino-12"]
    assert Feedback.objects.get().case_id == "tg-alpha_casino-12"
    assert 'data-role="cluster"' not in page  # a post is not a network with its own copy


# ------------------------------------------------------------------ whole channel
def test_check_whole_channel(signed_in):
    r = post_check(signed_in, "https://t.me/alpha_casino/12", action="channel")
    assert r.status_code == 200
    page = html(r)
    assert 'data-role="channel-report"' in page and "ALPHA Official" in page
    assert set(Case.objects.values_list("id", "source_type")) == {
        ("tg-alpha_casino-12", "post"), ("tg-alpha_casino-15", "post"), ("tg-alpha_casino-about", "account_bio")}
    assert set(Case.objects.values_list("account_id", flat=True)) == {"@alpha_casino"}
    assert "alpha-mirror-01.com" in Case.objects.get(pk="tg-alpha_casino-about").text
    assert 'data-role="no-network"' in page  # nothing to compare with yet


def test_channel_check_needs_a_telegram_link(signed_in):
    r = post_check(signed_in, "", action="channel")
    assert r.status_code == 422 and not Case.objects.exists()
    r = post_check(signed_in, "https://t.me/nobody_here", action="channel")
    assert r.status_code == 422


def test_observer_cannot_check_a_channel(client, observer):
    client.force_login(observer)
    assert 'data-role="check-channel"' not in html(client.get(reverse("check")))
    r = client.post(reverse("check"), {"source_url": "https://t.me/alpha_casino", "action": "channel"})
    assert r.status_code == 403 and not Case.objects.exists()


# ------------------------------------------------------------------ plain words
def test_two_channels_sharing_a_site(signed_in):
    post_check(signed_in, "https://t.me/alpha_casino", action="channel")
    page = html(post_check(signed_in, "https://t.me/beta_casino", action="channel"))
    # the second channel's report already names the network it joined
    assert re.search(r'<b class="net-name">@alpha_casino, @beta_casino</b>', page)
    assert "Бір сайт: <b>spin-route-77.click</b>" in page and "2 аккаунтта" in page

    page = html(signed_in.get(reverse("case", args=["tg-beta_casino-7"])))
    assert "🔗 Бұл пост басқа 1 аккаунтпен байланысты: @alpha_casino." in page
    assert "Неге байланысты:" in page and "кластер" not in page.lower()

    page = html(signed_in.get(reverse("case", args=["tg-beta_casino-7"]), {"lang": "ru"}))
    assert "Этот пост связан ещё с 1 аккаунтом: @alpha_casino." in page and "Общий сайт" in page

    # queue: every case of both channels is marked (the channels are linked), counter and filter
    in_network = {"tg-alpha_casino-12", "tg-alpha_casino-15", "tg-alpha_casino-about", "tg-beta_casino-7",
                  "tg-beta_casino-about"}
    page = html(signed_in.get(reverse("queue")))
    assert page.count('data-role="net-badge"') == len(in_network)
    linked = re.search(r'data-counter="linked">\s*<span class="counter-n">(\d+)</span>', page)
    assert linked and linked.group(1) == str(len(in_network))
    page = html(signed_in.get(reverse("queue"), {"linked": "1"}))
    assert set(re.findall(r'<td class="snippet"><a href="/case/([^/]+)/">', page)) == in_network

    # network page: named cards; the drawing only on request
    page = html(signed_in.get(reverse("network")))
    assert "@alpha_casino, @beta_casino" in page and '<details class="graph-details" data-role="graph">' in page


def test_why_there_is_no_link(signed_in):
    page = html(signed_in.post(reverse("check"), {"text": "Сегодня в Алматы солнечно, всем хорошего дня",
                                                  "source_type": "post", "save": "1"}))
    assert "Байланыс іздеуге дерек жоқ" in page and "Кеңес: посттың сілтемесін қойсаңыз" in page
    page = html(post_check(signed_in, "https://t.me/alpha_casino/12"))
    assert re.search(r"Байланыс табылмады: посттағы .*промокод WILD100.* басқа аккаунттарда әзірге кездеспеді", page)
    assert "Кеңес:" not in page


# ------------------------------------------------------------------ recognisers
def test_links_with_any_tld_after_a_scheme():
    kinds = {(e.kind, e.label) for e in graph_entities(normalize("https://twingo.best/r/abc и caburka.click"))}
    assert ("domain", "twingo.best") in kinds and ("link", "twingo.best/r/abc") in kinds
    assert not any(label == "caburka.click" for _, label in kinds)  # bare name: a known TLD is still required


def test_big_platforms_are_not_shared_sites():
    text = "https://vk.ru/club123 https://apps.apple.com/app/x https://linktr.ee/someone"
    assert [(e.kind, e.label) for e in graph_entities(normalize(text))] == [("link", "linktr.ee/someone")]
