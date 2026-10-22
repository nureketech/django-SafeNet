"""Stage 4: ML classifier (TF-IDF + calibrated logistic regression)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from sklearn.metrics import f1_score

from safenet import LABELS, paths
from tests.conftest import require


def _make_train_command() -> list[str]:
    if shutil.which("make"):
        return ["make", "train", f"PY={sys.executable}"]
    if os.name == "nt":
        return ["cmd", "/c", str(paths.ROOT / "make.bat"), "train"]
    return [sys.executable, str(paths.ROOT / "tasks.py"), "train"]


@pytest.fixture(scope="module")
def trained_model():
    """Train through `make train` once for the module and load the result."""
    require(paths.split_path("train"))
    before = paths.MODEL_PATH.stat().st_mtime if paths.MODEL_PATH.exists() else 0.0
    proc = subprocess.run(_make_train_command(), cwd=paths.ROOT, capture_output=True, text=True, timeout=1800,
                          env={**os.environ, "PYTHONUTF8": "1"})
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    assert paths.MODEL_PATH.exists()
    assert paths.MODEL_PATH.stat().st_mtime > before
    from safenet.ml.model import load_model

    load_model.cache_clear()
    return load_model()


@pytest.fixture(scope="module")
def val() -> pd.DataFrame:
    require(paths.split_path("val"))
    return pd.read_parquet(paths.split_path("val"))


def test_make_train_creates_model(trained_model):
    assert paths.MODEL_PATH.exists()
    assert list(trained_model.classes_) == sorted(LABELS)
    for report in ("model_report.md", "model_top_features.md"):
        assert (paths.REPORTS_DIR / report).exists()


def test_probabilities_sum_to_one(trained_model):
    from safenet.ml import predict_proba

    texts = ["Вложи 5000 получи 10000 за 3 дня", "Тегін ұтыс, тек жазылыңыз", "", "Регайтесь по ссылке в био"]
    proba = predict_proba(texts, trained_model)
    assert list(proba.columns) == list(LABELS)
    assert proba.shape == (4, 4)
    np.testing.assert_allclose(proba.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert ((proba.to_numpy() >= 0) & (proba.to_numpy() <= 1)).all()


def test_val_macro_f1(trained_model, val):
    pred = trained_model.predict(val["text"])
    macro_f1 = f1_score(val["label"], pred, average="macro")
    assert macro_f1 >= 0.80, macro_f1


def test_val_hard_negatives(trained_model, val):
    hn = val[val["is_hard_negative"]]
    pred = trained_model.predict(hn["text"])
    rate = float((pred != "none").mean())
    assert rate <= 0.15, rate


def test_single_record_inference_is_fast(trained_model, val):
    texts = val["text"].tolist()[:40]
    trained_model.predict_proba(pd.Series(texts[:2]))
    start = time.perf_counter()
    for text in texts:
        trained_model.predict_proba(pd.Series([text]))
    avg_ms = (time.perf_counter() - start) / len(texts) * 1000
    assert avg_ms < 20, avg_ms


def test_classes_are_sensible(trained_model):
    from safenet.ml import predict_proba

    proba = predict_proba([
        "Розыгрыш iPhone! Участие 1000 тг на Kaspi, победителя выберем рандомайзером",
        "Мой занос 500к в казино, промокод SPIN200 даёт бонус 200% к депозиту",
        "Вложи 10 000 и через неделю получи 30 000, гарантированно, куратор поможет",
        "Кто идёт завтра на футбол в 18:00?",
    ], trained_model)
    assert proba.idxmax(axis=1).tolist() == ["lottery", "casino", "pyramid", "none"]


def test_training_is_deterministic():
    require(paths.split_path("train"))
    from safenet.ml.train import train_model

    a = train_model(seed=42, c_grid=(4.0,))
    b = train_model(seed=42, c_grid=(4.0,))
    assert a.val_metrics == b.val_metrics
    texts = pd.Series(["вложи 5к получи 10к", "Қатысу тегін", "казино бонус"])
    np.testing.assert_array_equal(a.model.predict_proba(texts), b.model.predict_proba(texts))


def test_report_lists_top_features_for_every_class():
    text = (paths.REPORTS_DIR / "model_top_features.md").read_text(encoding="utf-8")
    for label in LABELS:
        assert f"## {label}" in text


# ------------------------------------------------- numpy/scipy fallback backend
def test_softmax_gradient_is_correct():
    from safenet.ml.linear import softmax_objective

    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    Y = np.eye(3)[rng.integers(0, 3, 30)]
    sw = rng.uniform(0.5, 2.0, 30)
    w0 = rng.normal(size=5 * 3 + 3)
    err = check_grad(lambda w: softmax_objective(w, X, Y, sw, 2.0)[0],
                     lambda w: softmax_objective(w, X, Y, sw, 2.0)[1], w0)
    assert err < 1e-4


def test_fallback_classifier_learns_and_calibrates():
    from sklearn.model_selection import GroupKFold

    from safenet.ml.linear import GroupCalibratedSoftmax, fit_sigmoid

    rng = np.random.default_rng(1)
    centers = np.array([[3, 0], [0, 3], [-3, -3]])
    y = np.repeat(np.array(["a", "b", "c"]), 60)
    X = centers[np.repeat(np.arange(3), 60)] + rng.normal(size=(180, 2))
    groups = np.arange(180) // 6
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))
    clf = GroupCalibratedSoftmax(C=1.0, folds=folds).fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (clf.predict(X) == y).mean() > 0.95
    a, b = fit_sigmoid(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
    assert a < 0  # higher decision value -> higher calibrated probability


def test_backend_can_be_forced(monkeypatch):
    from safenet.ml import linear

    monkeypatch.setenv("SAFENET_ML_BACKEND", "numpy")
    assert linear.backend() == "numpy"
    assert isinstance(linear.make_calibrated_classifier(1.0, folds=[]), linear.GroupCalibratedSoftmax)


def test_embeddings_flag_skips_cleanly_without_package():
    from safenet.ml import embeddings

    ok, reason = embeddings.available()
    if ok:
        pytest.skip("sentence-transformers is installed; the comparison runs for real")
    result = embeddings.evaluate_embeddings()
    assert result["status"] == "skipped" and result["reason"]
