"""Stage 6: OCR of screenshots, transcripts, joint analysis of a post + screenshot."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from safenet import paths
from safenet.multimodal import benchmark, ocr, transcribe
from safenet.pipeline import analyze
from tests.conftest import require

SCHEME_SCREEN = "Казино Kumisbet\nВывод средств\nСумма: 1 281 000 ₸\nСтатус: Выполнено\nПромокод: QAZ-2184"


def _engine_or_skip():
    engine, reason = ocr.available_engine()
    if engine is None:
        pytest.skip(f"OCR engine unavailable — {reason}")
    return engine


@pytest.fixture()
def rendered(tmp_path):
    """A dark, noisy, rotated synthetic screenshot."""
    from safenet.generator.images import render_screenshot

    for seed in range(300):
        meta = render_screenshot(SCHEME_SCREEN, tmp_path / "shot.png", random.Random(seed))
        if meta["theme"] == "dark" and abs(meta["angle"]) > 1.5:
            return tmp_path / "shot.png", meta
    pytest.fail("could not render a dark rotated screenshot")


# -------------------------------------------------------------- preprocessing
def test_preprocessing_gives_black_text_on_white(rendered):
    path, meta = rendered
    clean, info = ocr.preprocess_image(path)
    arr = np.asarray(clean)
    assert set(np.unique(arr)) <= {0, 255}
    assert (arr == 255).mean() > 0.8  # mostly white background
    assert info["inverted"] is True
    assert abs(info["skew"] + meta["angle"]) <= 1.0  # rotation detected and undone


def test_otsu_threshold_separates_two_levels():
    gray = np.array([[20] * 50 + [230] * 50] * 10, dtype=np.uint8)
    t = ocr.otsu_threshold(gray)
    assert 20 <= t < 230


def test_load_image_accepts_bytes_and_paths(rendered):
    path, _ = rendered
    assert ocr.load_image(path.read_bytes()).size == Image.open(path).size
    assert ocr.load_image(str(path)).size == Image.open(path).size


# --------------------------------------------------------- unavailable engine
def test_missing_tesseract_gives_clear_message(monkeypatch, rendered):
    path, _ = rendered
    monkeypatch.setenv("TESSERACT_CMD", r"Z:\nowhere\tesseract.exe")
    monkeypatch.setenv("SAFENET_OCR_ENGINE", "tesseract")
    ocr.tesseract_status.cache_clear()
    try:
        with pytest.raises(ocr.OCRUnavailable) as exc:
            ocr.ocr_image(path)
        assert "Tesseract" in str(exc.value) and "install" in str(exc.value).lower()
    finally:
        ocr.tesseract_status.cache_clear()


def test_pipeline_falls_back_to_text_without_ocr(monkeypatch):
    def unavailable(_image):
        raise ocr.OCRUnavailable("no engine (test)")

    monkeypatch.setattr(ocr, "ocr_image", unavailable)
    result = analyze({"text": "Вложи 10 000 — через неделю получишь 30 000, гарантированно",
                      "image_path": "data/images/missing.png"}, use_ml=False)
    assert result.ocr_text is None
    assert any("OCR unavailable" in w for w in result.warnings)
    assert {s.id for s in result.signals} >= {"P_UNREAL_RETURN", "P_GUARANTEE"}


def test_audio_transcriber_reports_missing_package():
    ok, reason = transcribe.available()
    if ok:
        pytest.skip("faster-whisper is installed")
    with pytest.raises(transcribe.TranscriberUnavailable, match="faster-whisper"):
        transcribe.transcribe_audio("stream.mp3")


# ------------------------------------------------- joint post + screenshot
def test_post_and_screenshot_are_analysed_together(monkeypatch):
    def fake_ocr(_image):
        return ocr.OCRResult(text=SCHEME_SCREEN, confidence=0.93, engine="stub")

    monkeypatch.setattr(ocr, "ocr_image", fake_ocr)
    post = "Ребята, смотрите какой вывод пришёл 🔥 ссылка в био"
    result = analyze({"text": post, "image_path": "any.png"}, use_ml=False)
    sources = {e.source for s in result.signals for e in s.evidence}
    assert sources == {"text", "ocr"}
    for s in result.signals:
        for e in s.evidence:
            base = result.ocr_text if e.source == "ocr" else result.text
            assert base[e.start:e.end] == e.text
    assert "K_GAMBLING_TERMS" in {s.id for s in result.signals}  # only in the screenshot
    assert "C_SHORTLINK" in {s.id for s in result.signals}       # only in the post
    assert result.entities["promo_codes"] == ["QAZ-2184"]
    assert result.ocr_engine == "stub" and result.ocr_confidence == 0.93


def test_transcript_evidence_is_labelled():
    result = analyze({"text": "так ребята регайтесь по моей ссылке в описании казино бонус 200% к депозиту",
                      "source_type": "stream_transcript"}, use_ml=False)
    assert result.signals
    assert {e.source for s in result.signals for e in s.evidence} == {"transcript"}


# ---------------------------------------------- screenshots, perfect OCR
def test_screenshot_texts_are_scored_correctly_with_perfect_ocr():
    require(paths.DATASET_PATH)
    require(paths.MODEL_PATH, "make train")
    shares = benchmark.oracle_levels()
    assert shares["scheme_at_least_medium"] >= 0.75, shares
    assert shares["hard_negative_low"] >= 0.85, shares


def test_key_tokens():
    tokens = benchmark.key_tokens("Kaspi Перевод\nСумма: 12 500 ₸\nПромокод SPIN200\nТөлем түсті")
    assert {"amount:12500", "word:каспи", "word:промокод", "word:толем"} <= tokens
    assert benchmark.token_recovery("Сумма: 12 500 ₸, каспи", "Сумма: 12 500 Т, каспи") == 1.0


# --------------------------------------------------- screenshots, real OCR
@pytest.fixture(scope="module")
def real_ocr():
    engine, reason = ocr.available_engine()
    if engine is None:
        pytest.skip(f"OCR engine unavailable — {reason}")
    require(paths.DATASET_PATH)
    require(paths.MODEL_PATH, "make train")
    return benchmark.run_benchmark()


def test_ocr_recovers_key_tokens(real_ocr):
    assert real_ocr["mean_token_recovery"] >= 0.8, real_ocr["mean_token_recovery"]


def test_scheme_screenshots_reach_medium(real_ocr):
    assert real_ocr["scheme_at_least_medium"] >= 0.75


def test_hard_negative_screenshots_stay_low(real_ocr):
    assert real_ocr["hard_negative_low"] >= 0.85


def test_ocr_on_rendered_screenshot(rendered):
    _engine_or_skip()
    path, _ = rendered
    result = ocr.ocr_image(path)
    tokens = benchmark.key_tokens(result.text)
    assert "word:промокод" in tokens and "amount:1281000" in tokens, result.text
    assert pd.notna(result.confidence)
