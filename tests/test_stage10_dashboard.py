"""Stage 10: analyst dashboard (Django views) — pages, check, feedback, masks, wording.

Since the production hardening every page needs a signed-in user: the tests run as an analyst
(see tests/test_stage12_production.py for access control itself).
"""

from __future__ import annotations

import csv
import re

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.urls import reverse

from app.dashboard.models import Case, Feedback
from safenet import DISCLAIMER_KK, DISCLAIMER_RU, paths
from safenet.common import find_forbidden
from safenet.multimodal import ocr as ocr_module
from safenet.pipeline import analyze, analyze_many
from tests.conftest import require

PYRAMID = ("Вложи 10 000 — через неделю получишь 30 000! Гарантированно, без риска. С каждого приглашённого "
           "10%. Осталось 3 места, пиши в лс")
FEEDBACK_HEADER = ["timestamp", "case_id", "verdict", "risk_score", "risk_level", "primary_track", "source_type",
                   "lang", "comment", "user"]
# removed from the interface for production (customer request)
TAGLINES = ("Схемалардың белгілерін анықтау — шешімді маман қабылдайды",
            "Выявление признаков схем — решение принимает специалист")
# open (unmasked) card / phone numbers
OPEN_CARD = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")
OPEN_PHONE = re.compile(r"\+7[ (]*7\d\d[) ]*\d{3}[ -]?\d{2}[ -]?\d{2}")


@pytest.fixture(scope="module", autouse=True)
def _artifacts():
    require(paths.MODEL_PATH, "make train")
    require(paths.split_path("test"), "make data")


@pytest.fixture(scope="module")
def analysed():
    """Results for a small, varied slice of the test split (computed once per module)."""
    from safenet.graph.lookup import load_graph

    df = pd.read_parquet(paths.split_path("test"))
    picked = [df[df.source_type != "screenshot"].head(40), df[df.source_type == "screenshot"].head(2)]
    graph = load_graph()
    if graph is not None and graph.clusters:
        first = next(iter(graph.clusters.values()))
        picked.append(df[df.account_id.isin(first.accounts) & (df.source_type != "screenshot")].head(5))
    df = pd.concat(picked).drop_duplicates("id")
    texts, shots = df[df.source_type != "screenshot"], df[df.source_type == "screenshot"]
    out = [(r, "") for r in analyze_many(
        {"id": r.id, "text": r.text, "source_type": r.source_type, "platform": r.platform,
         "account_id": r.account_id} for r in texts.itertuples())]
    ocr_ok = ocr_module.available_engine()[0] is not None
    for r in shots.itertuples():
        item = {"id": r.id, "text": "", "source_type": "screenshot", "platform": r.platform,
                "account_id": r.account_id}
        result = analyze(item, image=paths.ROOT / r.image_path) if ocr_ok else analyze({**item, "text": r.text})
        out.append((result, r.image_path))
    return out


@pytest.fixture
def cases(db, analysed):
    Case.objects.bulk_create([Case.from_result(r, image_path=img) for r, img in analysed])
    return Case.objects.all()


@pytest.fixture(autouse=True)
def _signed_in(client, analyst):
    client.force_login(analyst)


@pytest.fixture
def feedback_csv(settings, tmp_path):
    settings.SAFENET_FEEDBACK_PATH = str(tmp_path / "feedback" / "feedback.csv")
    return tmp_path / "feedback" / "feedback.csv"


def html(response) -> str:
    return response.content.decode("utf-8")


def assert_clean(page: str, result: bool = False) -> None:
    """No verdict wording and no tagline; every result card carries the disclaimer."""
    assert not find_forbidden(page), find_forbidden(page)
    assert not any(tagline in page for tagline in TAGLINES)
    assert 'data-role="disclaimer"' not in page  # the per-page footer disclaimer is gone
    if result:
        assert 'class="disclaimer inline"' in page and (DISCLAIMER_KK in page or DISCLAIMER_RU in page)


# ------------------------------------------------------------------ pages
def test_every_page_renders(client, cases):
    high = cases.filter(risk_level="high").first()
    case_url = reverse("case", args=[high.id])
    urls = [reverse("queue"), reverse("check"), reverse("network"), reverse("metrics"), case_url,
            reverse("check") + "?example=lottery"]
    for lang in ("kk", "ru"):
        for url in urls:
            r = client.get(url, {"lang": lang})
            assert r.status_code == 200, url
            page = html(r)
            assert_clean(page, result=url == case_url)
            if url == case_url:
                assert (DISCLAIMER_KK if lang == "kk" else DISCLAIMER_RU) in page
            assert f'<html lang="{lang}">' in page


def test_language_switch_sets_cookie(client, db):
    r = client.get(reverse("set_lang", args=["ru"]), HTTP_REFERER=reverse("metrics"))
    assert r.status_code == 302 and r["Location"] == reverse("metrics")
    assert r.cookies["safenet_lang"].value == "ru"
    assert "Очередь кейсов" in html(client.get(reverse("queue")))
    client.get(reverse("set_lang", args=["kk"]))
    assert "Кейстер кезегі" in html(client.get(reverse("queue")))
    r = client.get(reverse("set_lang", args=["en"]))  # unknown language: no cookie change
    assert "safenet_lang" not in r.cookies


def test_empty_queue_message(client, db):
    page = html(client.get(reverse("queue")))
    assert "Кезекте кейс жоқ." in page and "load_cases" not in page
    assert_clean(page)


# ------------------------------------------------------------------ queue
def test_queue_sorted_with_counters(client, cases):
    page = html(client.get(reverse("queue")))
    scores = [int(s) for s in re.findall(r'<td class="num">(\d+)</td>', page)]
    assert scores and scores == sorted(scores, reverse=True)
    for level in ("high", "medium", "low"):
        n = cases.filter(risk_level=level).count()
        assert re.search(rf'data-counter="{level}">\s*<span class="counter-n">{n}</span>', page)
    assert cases.filter(risk_level="high").exists() and "badge-high" in page
    # every row opens its case card with one click
    first = cases.first()
    assert f"location.href='{reverse('case', args=[first.id])}'" in page


@pytest.mark.parametrize("param,field", [("level", "risk_level"), ("track", "primary_track"), ("lang", "lang"),
                                         ("platform", "platform"), ("source", "source_type")])
def test_queue_filters(client, cases, param, field):
    value = cases.values_list(field, flat=True).first()
    page = html(client.get(reverse("queue"), {param: value}))
    expected = cases.filter(**{field: value})
    ids = re.findall(r'<td class="snippet"><a href="/case/([^/]+)/">', page)
    assert ids and set(ids) == set(expected.values_list("id", flat=True)[:50])


def test_queue_filter_without_match(client, cases):
    page = html(client.get(reverse("queue"), {"platform": "no-such-platform"}))
    assert 'class="empty"' in page and "load_cases" not in page


# ------------------------------------------------------------------- case
def test_case_card_highlights_evidence(client, cases):
    case = cases.filter(risk_level="high").exclude(text="").first()
    page = html(client.get(reverse("case", args=[case.id])))
    result = case.analysis
    assert f'data-level="{case.risk_level}"' in page
    assert re.search(r'<mark class="ev ev-(lottery|casino|pyramid|common)"', page)
    for s in result.signals:
        assert f'data-signal="{s.id}"' in page
    assert 'data-role="contributions"' in page and 'class="bar-row total"' in page
    # TZ button labels
    for label in ("Расталды", "Жалған іске қосылу", "Өткізіп жіберу"):
        assert f">{label}</button>" in page
    assert_clean(page, result=True)


def test_contribution_bars_add_up(client, cases):
    """Bars show every component of the score; they sum to the shown score."""
    for case in cases.exclude(risk_level="low")[:10]:
        page = html(client.get(reverse("case", args=[case.id])))
        block = page.split('data-role="contributions"', 1)[1].split('class="bar-row total"', 1)[0]
        values = [float(v) for v in re.findall(r'<span class="bar-value">\+?(-?\d+)</span>', block)]
        assert abs(sum(values) - case.risk_score) <= len(values) * 0.5 + 0.01, case.id


def test_screenshot_case_shows_image_and_ocr(client, cases):
    case = cases.exclude(image_path="").first()
    if case is None:
        pytest.skip("тест бөлігінде скриншот жоқ")
    page = html(client.get(reverse("case", args=[case.id])))
    image_url = reverse("case_image", args=[case.id])
    assert f'src="{image_url}"' in page
    if case.result.get("ocr_text"):
        assert 'class="ocr"' in page
    r = client.get(image_url)
    assert r.status_code == 200 and r["Content-Type"] == "image/png"
    assert b"".join(r.streaming_content)[:8] == b"\x89PNG\r\n\x1a\n"


def test_case_image_only_from_images_dir(client, cases):
    case = cases.first()
    Case.objects.filter(pk=case.pk).update(image_path="app/settings.py")
    assert client.get(reverse("case_image", args=[case.id])).status_code == 404
    Case.objects.filter(pk=case.pk).update(image_path="")
    assert client.get(reverse("case_image", args=[case.id])).status_code == 404


def test_unknown_case_404(client, db):
    assert client.get(reverse("case", args=["nope"])).status_code == 404


# --------------------------------------------------------------- feedback
def test_false_positive_appends_feedback_csv(client, cases, feedback_csv):
    case = cases.filter(risk_level="high").first()
    url = reverse("case_feedback", args=[case.id])
    r = client.post(url, {"verdict": "false_positive", "comment": "дүкен, жай сату"})
    assert r.status_code == 302 and r["Location"].endswith("?fb=1")
    with feedback_csv.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == FEEDBACK_HEADER
    assert len(rows) == 2
    row = dict(zip(rows[0], rows[1]))
    assert row["case_id"] == case.id and row["verdict"] == "false_positive"
    assert row["risk_level"] == case.risk_level and row["comment"] == "дүкен, жай сату"
    assert row["user"] == "analyst1"

    client.post(url, {"verdict": "confirmed"})
    with feedback_csv.open(encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 3  # appended, header written once
    assert Feedback.objects.filter(case_ref=case.id).count() == 2

    assert set(Feedback.objects.values_list("username", flat=True)) == {"analyst1"}

    page = html(client.get(r["Location"]))
    assert 'data-role="feedback-saved"' in page and "дүкен, жай сату" in page and "analyst1" in page


def test_feedback_validation(client, cases, feedback_csv):
    case = cases.first()
    url = reverse("case_feedback", args=[case.id])
    assert client.post(url, {"verdict": "guilty"}).status_code == 400
    assert client.get(url).status_code == 405
    assert client.post(reverse("case_feedback", args=["nope"]), {"verdict": "confirmed"}).status_code == 404
    assert not feedback_csv.exists() and not Feedback.objects.exists()


# ------------------------------------------------------------------ check
def test_check_pyramid_is_high_with_signals(client, db):
    r = client.post(reverse("check"), {"text": PYRAMID, "source_type": "chat_message"})
    assert r.status_code == 200
    page = html(r)
    assert 'data-level="high"' in page and 'data-track="pyramid"' in page
    assert 'data-role="signals"' in page and 'data-signal="P_UNREAL_RETURN"' in page
    assert re.search(r'<mark class="ev ev-pyramid"', page)
    assert_clean(page, result=True)
    assert not Case.objects.exists()  # not saved unless asked


def test_check_examples_only_in_demo_mode(client, settings):
    settings.SAFENET_DEMO_MODE = False
    page = html(client.get(reverse("check"), {"example": "pyramid"}))
    assert 'class="chip"' not in page and "Вложи 10 000" not in page
    settings.SAFENET_DEMO_MODE = True
    page = html(client.get(reverse("check"), {"example": "pyramid"}))
    assert 'class="chip"' in page and "Вложи 10 000" in page and "data-level" not in page


def test_check_save_adds_case_to_queue(client, db):
    r = client.post(reverse("check"), {"text": PYRAMID, "source_type": "post", "platform": "telegram",
                                       "save": "1"})
    case = Case.objects.get(origin="check")
    assert case.risk_level == "high" and case.platform == "telegram"
    assert reverse("case", args=[case.id]) in html(r)
    client.post(reverse("check"), {"text": PYRAMID, "source_type": "post", "save": "1"})
    assert Case.objects.filter(origin="check").count() == 1  # same content -> same case
    assert case.id in html(client.get(reverse("queue")))


def test_check_rejects_empty_and_broken_input(client, db):
    r = client.post(reverse("check"), {"text": "   "})
    assert r.status_code == 422 and 'data-role="error"' in html(r)
    bad = SimpleUploadedFile("x.png", b"not an image", content_type="image/png")
    r = client.post(reverse("check"), {"text": "", "image": bad})
    assert r.status_code == 422 and 'data-role="error"' in html(r)


def test_check_with_screenshot(client, db):
    if ocr_module.available_engine()[0] is None:
        pytest.skip("OCR қозғалтқышы жоқ — make tessdata")
    df = pd.read_parquet(paths.split_path("test"))
    shot = df[(df.source_type == "screenshot") & (df.label != "none")].iloc[0]
    image = SimpleUploadedFile("shot.png", (paths.ROOT / shot.image_path).read_bytes(), content_type="image/png")
    r = client.post(reverse("check"), {"text": "", "image": image})
    assert r.status_code == 200
    page = html(r)
    assert 'src="data:image/png;base64,' in page and 'class="ocr"' in page
    assert 'data-level="' in page
    assert_clean(page, result=True)


# ---------------------------------------------------------------- network
def test_network_page_and_graph(client, cases):
    from safenet.graph.lookup import load_graph

    graph = load_graph()
    if graph is None:
        pytest.skip("граф жоқ — make train")
    cluster = next(iter(graph.clusters.values()))
    page = html(client.get(reverse("network"), {"cluster": cluster.cluster_id}))
    assert f'data-cluster="{cluster.cluster_id}"' in page
    assert f'src="{reverse("network_graph", args=[cluster.cluster_id])}"' in page
    linked = cases.filter(account_id__in=cluster.accounts)
    for case in linked[:5]:
        assert reverse("case", args=[case.id]) in page
    r = client.get(reverse("network_graph", args=[cluster.cluster_id]))
    assert r.status_code == 200 and r["X-Frame-Options"] == "SAMEORIGIN"
    assert client.get(reverse("network_graph", args=["C999"])).status_code == 404


def test_network_shows_only_masked_cards_and_phones(client, db):
    from safenet.graph.lookup import load_graph

    graph = load_graph()
    if graph is None:
        pytest.skip("граф жоқ — make train")
    for cluster_id in graph.clusters:
        page = html(client.get(reverse("network"), {"cluster": cluster_id}))
        assert not OPEN_CARD.search(page), cluster_id
        assert not OPEN_PHONE.search(page), cluster_id


def test_case_entities_are_masked(client, cases):
    for case in cases:
        ents = case.result.get("entities", {})
        for value in ents.get("cards", []) + ents.get("phones", []):
            assert "*" in value, (case.id, value)


# ---------------------------------------------------------------- metrics
def test_metrics_page(client, db):
    require(paths.REPORTS_DIR / "metrics.json", "make eval")
    page = html(client.get(reverse("metrics")))
    assert page.count('class="metric-card"') == 2
    for name in ("ablation.png", "confusion_test.png", "confusion_gold.png", "pr_curve.png"):
        url = reverse("report_file", args=[name])
        assert f'src="{url}"' in page
        r = client.get(url)
        assert r.status_code == 200 and r["Content-Type"] == "image/png"
    assert client.get(reverse("report_file", args=["errors.md"])).status_code == 200
    for name in ("metrics.json", "..%2Fdb.sqlite3", "thresholds.md"):
        assert client.get(f"/reports/{name}").status_code == 404
    assert_clean(page)


# ----------------------------------------------------------- load command
def test_load_cases_command(db, capsys):
    call_command("load_cases", "--split", "test", "--limit", "12")
    assert Case.objects.count() == 12
    assert Case.objects.filter(origin="dataset").count() == 12
    call_command("load_cases", "--split", "test", "--limit", "20", "--if-empty")
    assert Case.objects.count() == 12
    call_command("load_cases", "--split", "test", "--limit", "5", "--replace")
    assert Case.objects.count() == 5
