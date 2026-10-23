"""Stage 11: documentation, demo cases and pitch."""

from __future__ import annotations

import re

import numpy as np
import pytest
import yaml
from django.core.management import call_command
from PIL import Image

import tasks
from app.dashboard.models import Case
from safenet import DISCLAIMER_KK, LABELS, docs, paths
from safenet.common import find_forbidden, luhn_valid
from safenet.demo import load_cases, render_receipt
from safenet.multimodal import ocr as ocr_module
from tests.conftest import require

DOCS = ["README.md", "docs/METHODOLOGY.md", "docs/ARCHITECTURE.md", "docs/ROADMAP.md", "docs/PITCH.md"]
# metric-like numbers: shares with three decimals (0.972) and percentages with one decimal (6.4%)
NUMBER_RE = re.compile(r"(?<![\w.])(?:0\.\d{3}|\d{1,3}\.\d%)(?!\d)")
LINK_RE = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")


def read(name: str) -> str:
    return (paths.ROOT / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- documents
@pytest.mark.parametrize("name", DOCS)
def test_document_exists_and_not_empty(name):
    path = paths.ROOT / name
    assert path.exists(), name
    text = read(name)
    assert len(text) > 2000 and text.lstrip().startswith("# ")


@pytest.mark.parametrize("name", DOCS)
def test_document_wording(name):
    assert not find_forbidden(read(name)), find_forbidden(read(name))


@pytest.mark.parametrize("name", DOCS)
def test_relative_links_resolve(name):
    base = (paths.ROOT / name).parent
    for target in LINK_RE.findall(read(name)):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (base / target).exists(), f"{name}: {target}"


def test_readme_quick_start_and_screenshots():
    text = read("README.md")
    assert "make install data train eval app" in text
    assert "make.bat install data train eval app" in text
    assert "make demo" in text
    shots = re.findall(r"\((docs/img/[\w.]+\.png)\)", text)
    assert len(set(shots)) >= 4
    for shot in shots:
        with Image.open(paths.ROOT / shot) as img:
            assert img.width >= 1000 and img.height >= 600
    assert DISCLAIMER_KK in text


def test_methodology_lists_every_signal_and_combo():
    text = read("docs/METHODOLOGY.md")
    signals = yaml.safe_load(paths.SIGNALS_CONFIG.read_text(encoding="utf-8"))["signals"]
    missing = [s["id"] for s in signals if f"`{s['id']}`" not in text]
    assert not missing, missing
    combos = yaml.safe_load(paths.SCORING_CONFIG.read_text(encoding="utf-8"))["combos"]
    assert all(f"`{c['id']}`" in text for c in combos)
    for part in ("risk_score", "no_core_cap", "Шектер", "Жалған іске қосылулармен", "Шектеулер", "Этика",
                 "үкім емес, белгілер"):
        assert part in text, part


def test_architecture_scaling_plan():
    text = read("docs/ARCHITECTURE.md")
    for part in ("Kafka", "каскад", "OCR", "Көлденең масштабтау", "Өткізу қабілетін"):
        assert part in text, part
    require(paths.REPORTS_DIR / "benchmark.json", "make bench")
    assert "мс/жазба" in text and "<!-- auto:bench -->" in text


def test_roadmap_phases_in_order():
    text = read("docs/ROADMAP.md")
    phases = ["Пилот", "Платформалармен интеграция", "Кері байланыс бойынша қайта оқыту", "Бейне мен аудио",
              "Ведомствоаралық алмасу"]
    positions = [text.find(f"## {i}. {p}") for i, p in enumerate(phases, start=1)]
    assert all(p > 0 for p in positions) and positions == sorted(positions)
    assert "карта хэштері" in text and "домендер" in text


def test_pitch_plan_is_five_minutes():
    text = read("docs/PITCH.md")
    plan = [("Мәселе", "30 с"), ("Тәсіл", "1 мин"), ("Тірі демо", "2 мин"), ("Метрикалар", "40 с"),
            ("Масштаб және жоспар", "50 с")]
    seconds = 0
    for title, duration in plan:
        assert re.search(rf"\| {title} \| {duration} \|", text), title
        value, unit = duration.split()
        seconds += int(value) * (60 if unit == "мин" else 1)
    assert seconds == 300
    headings = re.findall(r"^## \d\. (.+?) — ", text, flags=re.M)
    assert headings == [t for t, _ in plan]
    for case in load_cases():
        if case.id != "demo-3-pyramid":  # shown through the «Пирамида» example of the check page
            assert case.id in text, case.id


# ------------------------------------------------------------------ numbers
def test_generated_blocks_are_up_to_date():
    require(paths.REPORTS_DIR / "metrics.json", "make eval")
    require(paths.THRESHOLDS_PATH, "make train")
    assert docs.sync(check=True) == [], "run `make docs`"


@pytest.mark.parametrize("name", ["README.md", "docs/PITCH.md"])
def test_metric_numbers_match_reports(name):
    require(paths.REPORTS_DIR / "metrics.md", "make eval")
    reports = read("reports/metrics.md")
    bench = paths.REPORTS_DIR / "benchmark.md"
    if bench.exists():
        reports += bench.read_text(encoding="utf-8")
    numbers = set(NUMBER_RE.findall(read(name)))
    assert numbers, name
    missing = sorted(n for n in numbers if n not in reports)
    assert not missing, f"{name}: {missing} not in reports/metrics.md"


@pytest.mark.parametrize("name", ["README.md", "docs/PITCH.md"])
def test_metric_table_equals_metrics_md(name):
    require(paths.REPORTS_DIR / "metrics.md", "make eval")
    main = read("reports/metrics.md").split("## Негізгі нәтижелер", 1)[1].split("\n## ", 1)[0]
    expected = {row.split("|")[1].strip(): row for row in main.splitlines() if row.startswith(("| test", "| gold"))}
    text = read(name)
    for split, row in expected.items():
        assert row in text, (name, split)


# --------------------------------------------------------------------- demo
def test_demo_spec_is_synthetic():
    cases = load_cases()
    assert len(cases) == 5 and len({c.id for c in cases}) == 5
    tracks = {c.expected_track for c in cases if c.expected_level == "high"}
    assert tracks == set(LABELS) - {"none"}
    assert any(c.expected_level == "low" and "тегін" in c.text.lower() for c in cases)
    assert any(c.source_type == "screenshot" and c.image for c in cases)
    for c in cases:
        text = c.text + " " + c.image_text
        for digits in re.findall(r"\d[\d ]{11,22}\d", text):
            assert not luhn_valid(digits.replace(" ", "")), (c.id, digits)
        for phone in re.findall(r"\+7[\d ()-]{9,16}", text):
            assert re.fullmatch(r"\+7 7\d\d \d{3} \d\d \d\d", phone.strip()), (c.id, phone)
        assert not re.search(r"https?://(?!\S+\.(test|example)\b)", text), c.id


def test_demo_screenshot_shows_its_text(tmp_path):
    case = next(c for c in load_cases() if c.image)
    fresh = render_receipt(case.image_text, tmp_path / "receipt.png")
    with Image.open(case.image_path) as committed, Image.open(fresh) as rendered:
        assert committed.size == rendered.size
        assert np.array_equal(np.asarray(committed.convert("RGB")), np.asarray(rendered.convert("RGB")))


@pytest.mark.django_db
def test_load_demo_puts_five_cases_with_expected_levels():
    require(paths.MODEL_PATH, "make train")
    call_command("load_demo")
    cases = {c.id: c for c in load_cases()}
    stored = Case.objects.filter(origin="demo")
    assert stored.count() == 5 and set(stored.values_list("id", flat=True)) == set(cases)
    ocr_ok = ocr_module.available_engine()[0] is not None
    for case in stored:
        spec = cases[case.id]
        assert case.risk_level == spec.expected(ocr_ok), (case.id, case.risk_score, case.risk_level)
        if spec.expected_track and case.risk_level != "low":
            assert case.primary_track == spec.expected_track, case.id
        result = case.analysis
        assert result.disclaimer_kk == DISCLAIMER_KK
        assert not find_forbidden(result.explanation_kk + result.explanation_ru)
    shot = stored.get(source_type="screenshot")
    assert shot.image_path == cases[shot.id].image
    if ocr_ok:
        assert shot.analysis.ocr_text and any(e.source == "ocr" for s in shot.analysis.signals for e in s.evidence)
    # loading again replaces the demo cases instead of duplicating them
    call_command("load_demo")
    assert Case.objects.filter(origin="demo").count() == 5


@pytest.mark.django_db
def test_demo_cases_open_in_dashboard(client):
    require(paths.MODEL_PATH, "make train")
    call_command("load_demo")
    page = client.get("/", {"origin": "demo"}).content.decode()
    assert 'data-counter="demo"' in page
    for case in Case.objects.filter(origin="demo"):
        assert f"/case/{case.id}/" in page
        assert client.get(f"/case/{case.id}/").status_code == 200
    shot = Case.objects.get(origin="demo", source_type="screenshot")
    r = client.get(f"/case/{shot.id}/image/")
    assert r.status_code == 200 and r["Content-Type"] == "image/png"


def test_make_demo_runs_load_demo_and_opens_queue(monkeypatch):
    commands, served = [], {}
    monkeypatch.setattr(tasks, "run_all", lambda *cmds: commands.extend(cmds) or 0)
    monkeypatch.setattr(tasks, "_serve", lambda port, open_browser, **kw: served.update(port=port, **kw) or 0)
    assert tasks.main(["demo", "PORT=8123", "OPEN=0"]) == 0
    steps = [c[2:] for c in commands]
    assert ("load_cases", "--if-empty") in steps and ("load_demo",) in steps
    assert steps.index(("load_cases", "--if-empty")) < steps.index(("load_demo",))
    assert served == {"port": "8123", "load_cases": False, "page": "/?origin=demo"}


def test_make_runs_several_targets_in_order(monkeypatch):
    calls = []
    fake = {name: (lambda n: lambda p: calls.append(n) or (1 if n == "eval" else 0))(name)
            for name in ("install", "data", "train", "eval", "app")}
    monkeypatch.setattr(tasks, "TARGETS", fake)
    assert tasks.main(["install", "data", "train", "eval", "app"]) == 1
    assert calls == ["install", "data", "train", "eval"]  # stops at the first failure
