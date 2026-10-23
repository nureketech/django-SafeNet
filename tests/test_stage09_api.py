"""Stage 9: HTTP API (django-ninja) — 200 on valid input, 422 on invalid, schema, batch speed."""

from __future__ import annotations

import glob
import json
import time

import pandas as pd
import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from safenet import DISCLAIMER_KK, paths
from safenet.common import find_forbidden
from safenet.scoring.models import AnalysisResult
from tests.conftest import require

PYRAMID = "Вложи 10 000 — через неделю получишь 30 000! Гарантированно, без риска. Пиши в лс"


@pytest.fixture(scope="module", autouse=True)
def _model():
    require(paths.MODEL_PATH, "make train")


def post_json(client, url, payload):
    body = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return client.post(url, data=body, content_type="application/json")


# ---------------------------------------------------------------- service
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok" and data["model_loaded"] is True
    assert set(data["thresholds"]) == {"medium", "high"}


def test_openapi_docs(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    paths_ = schema.json()["paths"]
    assert {"/analyze", "/analyze/image", "/batch", "/health"} <= set(paths_)
    assert "AnalysisResult" in json.dumps(schema.json())


# ---------------------------------------------------------------- /analyze
def test_analyze_valid(client):
    r = post_json(client, "/analyze", {"text": PYRAMID, "source_type": "chat_message", "platform": "whatsapp",
                                       "account_id": "acc_test"})
    assert r.status_code == 200
    result = AnalysisResult.model_validate(r.json())
    assert result.risk_level == "high" and result.primary_track == "pyramid"
    assert result.signals and all(s.evidence for s in result.signals)
    assert result.disclaimer_kk == DISCLAIMER_KK
    assert result.platform == "whatsapp" and result.account_id == "acc_test"
    assert not any(find_forbidden(s) for s in (result.explanation_ru, result.explanation_kk))


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"text": ""},
        {"text": 123},
        {"text": "x", "source_type": "blog"},
        {"text": "x", "platform": "myspace"},
        {"text": "x" * 20001},
        "not json",
        "[]",
    ],
)
def test_analyze_invalid(client, body):
    assert post_json(client, "/analyze", body).status_code == 422


# ---------------------------------------------------------- /analyze/image
def test_analyze_image_valid(client):
    require(paths.DATASET_PATH)
    png = sorted(glob.glob(str(paths.IMAGES_DIR / "syn-*.png")))[0]
    with open(png, "rb") as f:
        r = client.post("/analyze/image", data={"image": f, "text": "Смотрите, что пришло", "platform": "telegram"})
    assert r.status_code == 200
    result = AnalysisResult.model_validate(r.json())
    assert result.text == "Смотрите, что пришло"
    if result.ocr_engine:  # OCR available: the screenshot text is analysed with source "ocr"
        assert result.ocr_text
        for s in result.signals:
            for e in s.evidence:
                base = result.ocr_text if e.source == "ocr" else result.text
                assert base[e.start:e.end] == e.text
    else:
        assert any("OCR unavailable" in w for w in result.warnings)


def test_analyze_image_without_text(client):
    require(paths.DATASET_PATH)
    png = sorted(glob.glob(str(paths.IMAGES_DIR / "syn-*.png")))[1]
    with open(png, "rb") as f:
        r = client.post("/analyze/image", data={"image": f})
    assert r.status_code == 200
    assert AnalysisResult.model_validate(r.json()).source_type == "screenshot"


@pytest.mark.parametrize(
    "data",
    [
        {"text": "нет файла"},
        {"image": SimpleUploadedFile("x.png", b"not an image", content_type="image/png")},
        {"image": SimpleUploadedFile("x.txt", "просто текст".encode(), content_type="text/plain")},
    ],
)
def test_analyze_image_invalid(client, data):
    assert client.post("/analyze/image", data=data).status_code == 422


def test_analyze_image_invalid_source_type(client):
    require(paths.DATASET_PATH)
    png = sorted(glob.glob(str(paths.IMAGES_DIR / "syn-*.png")))[0]
    with open(png, "rb") as f:
        r = client.post("/analyze/image", data={"image": f, "source_type": "blog"})
    assert r.status_code == 422


# ----------------------------------------------------------------- /batch
def test_batch_valid(client):
    items = [{"text": PYRAMID}, {"text": "Кто идёт завтра на футбол?", "id": "b2"},
             {"text": "Тегін розыгрыш, жазылып, досыңды белгіле", "source_type": "post"}]
    r = post_json(client, "/batch", items)
    assert r.status_code == 200
    results = [AnalysisResult.model_validate(x) for x in r.json()]
    assert len(results) == 3
    assert results[1].id == "b2"
    assert [x.risk_level for x in results] == ["high", "low", "low"]


@pytest.mark.parametrize("body", ["[]", {"text": "x"}, [{"text": ""}], [{"txt": "x"}], "{bad"])
def test_batch_invalid(client, body):
    assert post_json(client, "/batch", body).status_code == 422


def test_batch_of_100_is_fast(client):
    require(paths.DATASET_PATH)
    df = pd.read_parquet(paths.DATASET_PATH, columns=["id", "text", "source_type"]).sample(100, random_state=5)
    items = [{"id": r.id, "text": r.text, "source_type": r.source_type} for r in df.itertuples()]
    post_json(client, "/batch", items[:2])  # warm-up (model and graph loading)
    start = time.perf_counter()
    r = post_json(client, "/batch", items)
    elapsed = time.perf_counter() - start
    assert r.status_code == 200 and len(r.json()) == 100
    assert elapsed < 3.0, elapsed
