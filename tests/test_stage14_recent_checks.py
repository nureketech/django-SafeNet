"""Stage 14 (outside the TZ): checked content can be found again — recent checks on /check/,
a «Тексерілгендер» counter and a «newest first» order in the queue."""

from __future__ import annotations

import re
from datetime import timedelta

import pytest
from django.urls import reverse
from django.utils import timezone

from app.dashboard.models import Case
from app.dashboard.services import save_case
from safenet import paths
from safenet.pipeline import analyze
from tests.conftest import require

PYRAMID = ("Вложи 10 000 — через неделю получишь 30 000! Гарантированно, без риска. С каждого приглашённого "
           "10%. Осталось 3 места, пиши в лс")
NEUTRAL = "Ертең сағат 19:00-де кітап клубының кездесуі өтеді, барлығын шақырамыз."
ROW_ID = re.compile(r'<td class="snippet"><a href="/case/([^/]+)/">')


@pytest.fixture(scope="module", autouse=True)
def _artifacts():
    require(paths.MODEL_PATH, "make train")


@pytest.fixture(autouse=True)
def _signed_in(client, analyst):
    client.force_login(analyst)


def html(response) -> str:
    return response.content.decode("utf-8")


def recent_ids(page: str) -> list[str]:
    block = page.split('data-role="recent-checks"', 1)[1]
    return ROW_ID.findall(block)


def check(client, text: str, **extra):
    return client.post(reverse("check"), {"text": text, "source_type": "post", "save": "1", **extra})


def test_check_form_saves_by_default(client, db):
    page = html(client.get(reverse("check")))
    assert re.search(r'name="save" value="1" checked', page)
    assert 'data-role="recent-checks"' not in page  # nothing checked yet


def test_recent_checks_newest_first(client, db):
    check(client, PYRAMID)
    first = Case.objects.get(origin="check")
    Case.objects.filter(pk=first.pk).update(analysed_at=timezone.now() - timedelta(minutes=5))
    page = html(check(client, NEUTRAL))
    second = Case.objects.filter(origin="check").exclude(pk=first.pk).get()
    assert recent_ids(page) == [second.id, first.id]
    assert re.search(rf'class="clickable on" onclick="location.href=\'/case/{second.id}/\'', page)
    # re-checking the same content refreshes the same case and moves it up
    page = html(check(client, PYRAMID))
    assert Case.objects.filter(origin="check").count() == 2
    assert recent_ids(page) == [first.id, second.id]
    assert f'href="{reverse("queue")}?origin=check&sort=new"' in page


def test_unsaved_check_is_not_listed(client, db):
    page = html(client.post(reverse("check"), {"text": PYRAMID, "source_type": "post"}))
    assert not Case.objects.exists() and 'data-role="recent-checks"' not in page
    assert re.search(r'name="save" value="1">', page)  # the analyst's choice is kept


def test_recent_checks_show_only_checks(client, db):
    save_case(analyze({"id": "api-1", "text": PYRAMID, "source_type": "post"}), origin="api")
    check(client, NEUTRAL)
    assert recent_ids(html(client.get(reverse("check")))) == [Case.objects.get(origin="check").id]


def test_queue_counter_and_newest_first(client, db):
    old = save_case(analyze({"id": "ds-high", "text": PYRAMID, "source_type": "post"}), origin="dataset")
    Case.objects.filter(pk=old.pk).update(analysed_at=timezone.now() - timedelta(days=1))
    check(client, NEUTRAL)
    checked = Case.objects.get(origin="check")
    assert checked.risk_score < old.risk_score

    page = html(client.get(reverse("queue")))
    assert ROW_ID.findall(page) == [old.id, checked.id]  # default: by risk score
    assert re.search(r'href="\?origin=check&sort=new" data-counter="check">\s*<span class="counter-n">1</span>', page)
    assert 'class="time"' not in page

    page = html(client.get(reverse("queue"), {"sort": "new"}))
    assert ROW_ID.findall(page) == [checked.id, old.id]
    assert '<option value="new" selected>' in page and 'class="time"' in page

    page = html(client.get(reverse("queue"), {"origin": "check", "sort": "new"}))
    assert ROW_ID.findall(page) == [checked.id]
    assert 'class="counter counter-check on"' in page


def test_newest_first_survives_paging(client, db):
    from app.dashboard.views import PAGE_SIZE

    Case.objects.bulk_create([
        Case(id=f"c{i:03d}", risk_score=i % 90, risk_level="low", primary_track="none", result={}, origin="check")
        for i in range(PAGE_SIZE + 5)
    ])
    page = html(client.get(reverse("queue"), {"origin": "check", "sort": "new"}))
    assert re.search(r'href="\?origin=check&(amp;)?sort=new&(amp;)?page=2"', page)
