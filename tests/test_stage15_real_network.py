"""Stage 15 (outside the TZ): the platform works on real data only.

* checked content carries its context: link to the post, account / channel, chat / group;
* links come from the live graph of real saved cases (checks and API), never from the synthetic one;
* synthetic cases are not loaded by ``make app`` and cannot be loaded in production.
"""

from __future__ import annotations

import re

import pytest
from django.core.management import CommandError, call_command
from django.urls import reverse

from app.dashboard import network as live
from app.dashboard.models import Case
from app.dashboard.services import save_case
from app.dashboard.sources import ContextError, account_from_url, build_context
from safenet import paths
from safenet.pipeline import analyze
from tests.conftest import require

PYRAMID = ("Вложи 10 000 — через неделю получишь 30 000! Гарантированно, без риска. С каждого приглашённого "
           "10%. Осталось 3 места, пиши в лс.")
CASINO = "Регистрируйся на spinwin-kz.top и вводи промокод ZOLOTO77 — бонус 200% к депозиту, фриспины каждый день"
OPEN_CARD = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")
OPEN_PHONE = re.compile(r"\+7[ (]*7\d\d[) ]*\d{3}[ -]?\d{2}[ -]?\d{2}")


@pytest.fixture(scope="module", autouse=True)
def _artifacts():
    require(paths.MODEL_PATH, "make train")


@pytest.fixture(autouse=True)
def _signed_in(client, analyst):
    client.force_login(analyst)


def html(response) -> str:
    return response.content.decode("utf-8")


def check(client, text: str, **context):
    return client.post(reverse("check"), {"text": text, "source_type": "post", "save": "1", **context})


def saved(origin: str, case_id: str, text: str, account: str = "", group: str = "") -> Case:
    result = analyze({"id": case_id, "text": text, "source_type": "post", "account_id": account or None},
                     network_lookup=None)
    return save_case(result, origin=origin, group_id=group)


# ------------------------------------------------------------------ context
@pytest.mark.parametrize("url,expected", [
    ("https://t.me/airolejon/15156", ("telegram", "@airolejon")),
    ("t.me/s/AiroLejon/15156", ("telegram", "@airolejon")),
    ("https://t.me/c/123/45", ("telegram", "")),
    ("https://t.me/+AbCdEf", ("telegram", "")),
    ("https://www.instagram.com/p/XYZ/", ("instagram", "")),
    ("https://instagram.com/some.user/", ("instagram", "@some.user")),
    ("https://www.tiktok.com/@user1/video/1", ("tiktok", "@user1")),
    ("https://vk.com/wall-12345_67", ("vk", "vk:-12345")),
    ("https://youtube.com/@Chan/videos", ("youtube", "@chan")),
    ("https://wa.me/77011234567", ("whatsapp", "")),
    ("https://example.com/post/1", ("", "")),
])
def test_account_from_link(url, expected):
    assert account_from_url(url) == expected


def test_build_context_normalises_names():
    assert build_context(account="AiroLejon", platform="telegram").account_id == "@airolejon"
    assert build_context(account="https://t.me/AiroLejon").account_id == "@airolejon"
    ctx = build_context(source_url="t.me/airolejon/15156", group="t.me/matikagents")
    assert (ctx.account_id, ctx.group_id, ctx.platform) == ("@airolejon", "@matikagents", "telegram")
    assert ctx.source_url == "https://t.me/airolejon/15156"
    assert build_context(account="@Other", source_url="https://t.me/airolejon/1").account_id == "@other"
    for bad in ("javascript:alert(1)", "not a link", "https://" + "x" * 600 + ".kz"):
        with pytest.raises(ContextError):
            build_context(source_url=bad)


def test_check_form_has_context_fields(client, db):
    page = html(client.get(reverse("check")))
    assert 'data-role="context"' in page
    for name in ("source_url", "account", "group"):
        assert f'name="{name}"' in page


def test_check_saves_context(client, db):
    r = check(client, PYRAMID, source_url="t.me/airolejon/15156", group="@matikagents")
    case = Case.objects.get(origin="check")
    assert (case.account_id, case.group_id, case.platform) == ("@airolejon", "@matikagents", "telegram")
    assert case.source_url == "https://t.me/airolejon/15156"
    assert reverse("case", args=[case.id]) in html(r)
    page = html(client.get(reverse("case", args=[case.id])))
    assert re.search(r'<a href="https://t.me/airolejon/15156" target="_blank" rel="noopener noreferrer nofollow" '
                     r'data-role="source-url">', page)
    assert 'data-role="account">@airolejon</a>' in page
    # the queue can be narrowed to the account
    page = html(client.get(reverse("queue"), {"account": "@airolejon"}))
    assert 'data-role="context-filter"' in page and case.id in page


def test_bad_source_url_is_rejected(client, db):
    r = check(client, PYRAMID, source_url="javascript:alert(1)")
    assert r.status_code == 422 and 'data-role="error"' in html(r)
    assert not Case.objects.exists()


def test_same_text_from_two_accounts_is_two_cases(client, db):
    check(client, CASINO, account="@chan_one", platform="telegram")
    check(client, CASINO, account="@chan_two", platform="telegram")
    check(client, CASINO, account="@chan_one", platform="telegram")
    assert sorted(Case.objects.values_list("account_id", flat=True)) == ["@chan_one", "@chan_two"]


# --------------------------------------------------------------- live graph
def test_posts_sharing_a_promo_code_form_a_cluster(client, db):
    check(client, CASINO)  # no account: the post is its own graph node
    first = Case.objects.get(origin="check")
    page = html(check(client, f"Залетай, промокод ZOLOTO77, выводы моментальные. {PYRAMID}"))
    second = Case.objects.exclude(pk=first.pk).get()
    assert 'data-role="cluster"' in page and reverse("case", args=[first.id]) in page

    page = html(client.get(reverse("case", args=[first.id])))
    cluster_id = re.search(r'data-network="(N[0-9A-F]{6}X*)"', page).group(1)  # stage 16: a network card
    assert reverse("case", args=[second.id]) in page and "ZOLOTO77" in page

    page = html(client.get(reverse("network")))
    assert f'data-cluster="{cluster_id}"' in page
    for case in (first, second):
        assert reverse("case", args=[case.id]) in page
    r = client.get(reverse("network_graph", args=[cluster_id]))
    assert r.status_code == 200 and r["X-Frame-Options"] == "SAMEORIGIN"
    assert client.get(reverse("network_graph", args=["C001"])).status_code == 404


def test_other_cases_of_the_same_account(client, db):
    check(client, PYRAMID, account="@airolejon", platform="telegram")
    check(client, "Новый проект, заходим со старта, детали в закрепе", account="@airolejon", platform="telegram")
    first, second = Case.objects.order_by("analysed_at")
    page = html(client.get(reverse("case", args=[first.id])))
    assert 'data-role="account-cases"' in page and reverse("case", args=[second.id]) in page
    assert 'data-role="cluster"' not in page  # one account is not a network


def test_shared_chat_links_accounts(db):
    saved("check", "g1", "Пишите в лс, все подробности в чате", account="@acc_a", group="@schemechat")
    saved("api", "g2", "Сегодня стрим, подключайтесь", account="@acc_b", group="@schemechat")
    model = live.live_graph()
    (cluster,) = model.clusters.values()
    assert cluster.accounts == ["@acc_a", "@acc_b"]
    assert [(e.kind, e.label) for e in cluster.shared] == [("group", "@schemechat")]


def test_synthetic_cases_never_enter_the_live_graph(db):
    saved("dataset", "syn-1", CASINO, account="acc1")
    saved("demo", "syn-2", CASINO, account="acc2")
    assert live.live_graph() is None
    saved("check", "real-1", CASINO, account="@real")
    model = live.live_graph()
    assert model.clusters == {} and set(model.account_cluster) == set()
    assert {n for n, d in model.graph.nodes(data=True) if d["kind"] == "account"} == {"acc:@real"}


def test_cluster_ids_are_stable_and_the_graph_is_cached(db):
    saved("check", "s1", CASINO, account="@one")
    saved("check", "s2", f"промокод ZOLOTO77 {PYRAMID}", account="@two")
    model = live.live_graph()
    assert live.live_graph() is model  # nothing changed: no rebuild
    (cid,) = model.clusters
    saved("check", "s3", "Обычный пост про погоду в Алматы", account="@three")
    rebuilt = live.live_graph()
    assert rebuilt is not model and list(rebuilt.clusters) == [cid]


def test_network_bonus_comes_from_the_live_graph(client, db):
    from safenet.graph.lookup import load_graph

    synthetic = load_graph()
    promo = next((e.label for c in synthetic.clusters.values() for e in c.shared if e.kind == "promo"), None) \
        if synthetic else None
    if promo:  # a promo code of a synthetic network links nothing on the platform
        r = check(client, f"{PYRAMID} Промокод {promo}")
        assert 'data-role="cluster"' not in html(r)
        Case.objects.all().delete()
    saved("check", "b1", f"{PYRAMID} Промокод ZOLOTO77", account="@b_one")
    saved("check", "b2", f"{PYRAMID} Код ZOLOTO77, пиши в лс", account="@b_two")
    result = analyze({"text": f"{PYRAMID} Промокод ZOLOTO77"}, network_lookup=live.live_lookup())
    assert result.network.cluster_id and result.network.contribution > 0
    page = html(check(client, f"{PYRAMID} Промокод ZOLOTO77 ещё раз"))
    assert re.search(r'class="bar-label" title="Байланыстар \(N[0-9A-F]{6}X*\)"', page)


def test_cards_and_phones_stay_masked(client, db):
    text = f"{PYRAMID} Переводи на +7 701 234 56 78, карта 4400 4301 2345 6789"
    check(client, text, account="+7 702 111 22 33")
    check(client, f"Оплата только на +7 701 234 56 78. {PYRAMID}", account="@payer")
    for case in Case.objects.all():
        page = html(client.get(reverse("case", args=[case.id])))
        meta = page.split('class="meta"', 1)[1].split("</p>", 1)[0]
        assert not OPEN_PHONE.search(meta) and not OPEN_CARD.search(meta)
    page = html(client.get(reverse("network")))
    assert 'data-cluster="' in page
    links = page.split('class="grid-net"', 1)[1]
    assert not OPEN_PHONE.search(links) and not OPEN_CARD.search(links)
    assert "+7 702 *** ** 33" in links and "account=%2B7" not in links


# ------------------------------------------------------------------- API
def test_api_saves_context(client, api_token, db):
    client.logout()
    headers = {"HTTP_AUTHORIZATION": f"Bearer {api_token}"}
    body = {"text": PYRAMID, "source_url": "https://t.me/airolejon/15156", "group_id": "@matikagents", "save": True}
    r = client.post("/analyze", body, content_type="application/json", **headers)
    assert r.status_code == 200 and r.json()["account_id"] == "@airolejon"
    case = Case.objects.get(origin="api")
    assert (case.account_id, case.group_id, case.source_url) == (
        "@airolejon", "@matikagents", "https://t.me/airolejon/15156")
    r = client.post("/analyze", {"text": PYRAMID, "source_url": "javascript:alert(1)"},
                    content_type="application/json", **headers)
    assert r.status_code == 422
    assert client.get("/health", **headers).json()["graph_loaded"] is True


# ------------------------------------------------------- no synthetic data
def test_make_app_does_not_load_synthetic_cases(monkeypatch):
    import tasks

    commands = []
    monkeypatch.setattr(tasks, "run", lambda *args: commands.append(args[1:]) or 0)
    monkeypatch.setattr(tasks.webbrowser, "open", lambda url: None)
    assert tasks.target_app({"OPEN": "0"}) == 0
    assert all(cmd[:2] != ("manage.py", "load_cases") for cmd in commands)
    assert ("manage.py", "runserver", "127.0.0.1:8000") in commands


@pytest.mark.parametrize("command", ["load_cases", "load_demo"])
def test_synthetic_loaders_refuse_in_production(settings, db, command):
    settings.PRODUCTION = True
    with pytest.raises(CommandError, match="synthetic"):
        call_command(command)
    assert not Case.objects.exists()
