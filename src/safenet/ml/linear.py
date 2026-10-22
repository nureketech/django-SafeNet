"""Calibrated multinomial logistic regression with a numpy/scipy fallback.

The project uses scikit-learn's ``LogisticRegression`` + ``CalibratedClassifierCV``.
On machines where those modules cannot be imported (e.g. Windows Smart App
Control blocks sklearn's compiled ``_libsvm`` extension), the same algorithm is
provided here in pure numpy/scipy:

* ``SoftmaxRegression`` — multinomial logistic regression, L2 penalty with the
  sklearn convention ``0.5*||W||^2 + C * sum(w_i * logloss_i)``, unpenalised
  intercept, ``class_weight="balanced"``, L-BFGS;
* ``GroupCalibratedSoftmax`` — sklearn's ``CalibratedClassifierCV(method="sigmoid",
  ensemble=True)``: for every CV fold a classifier is fitted, a Platt sigmoid is
  fitted per class (one-vs-rest) on the held-out decision values, and the
  normalised calibrated probabilities are averaged over folds.

``make_calibrated_classifier`` picks the backend (env ``SAFENET_ML_BACKEND``:
``auto`` (default) | ``sklearn`` | ``numpy``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit, logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin


def _balanced_weights(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y_idx, minlength=n_classes).astype(float)
    weights = len(y_idx) / (n_classes * np.maximum(counts, 1.0))
    return weights[y_idx]


def softmax_objective(w: np.ndarray, X, Y: np.ndarray, sw: np.ndarray, C: float) -> tuple[float, np.ndarray]:
    """Loss ``0.5*||W||^2 - C * sum_i sw_i * log p(y_i)`` and its gradient (W: features x classes, then b)."""
    d, k = X.shape[1], Y.shape[1]
    W = w[: d * k].reshape(d, k)
    b = w[d * k:]
    Z = X @ W + b
    log_p = Z - logsumexp(Z, axis=1, keepdims=True)
    loss = -C * float(np.sum(sw * np.sum(Y * log_p, axis=1))) + 0.5 * float(np.sum(W * W))
    G = C * sw[:, None] * (np.exp(log_p) - Y)
    grad_w = np.asarray(X.T @ G) + W
    return loss, np.concatenate([grad_w.ravel(), G.sum(axis=0)])


class SoftmaxRegression(BaseEstimator, ClassifierMixin):
    """Multinomial logistic regression (sklearn-compatible attributes ``coef_``, ``intercept_``)."""

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced", max_iter: int = 2000,
                 tol: float = 1e-6):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        n, d = X.shape
        k = len(self.classes_)
        Y = np.zeros((n, k))
        Y[np.arange(n), y_idx] = 1.0
        sw = _balanced_weights(y_idx, k) if self.class_weight == "balanced" else np.ones(n)
        C = float(self.C)

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            return softmax_objective(w, X, Y, sw, C)

        res = minimize(objective, np.zeros(d * k + k), jac=True, method="L-BFGS-B",
                       options={"maxiter": self.max_iter, "gtol": self.tol})
        W = res.x[: d * k].reshape(d, k)
        self.coef_ = W.T.copy()
        self.intercept_ = res.x[d * k:].copy()
        self.n_iter_ = int(res.nit)
        return self

    def decision_function(self, X) -> np.ndarray:
        return np.asarray(X @ self.coef_.T) + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        Z = self.decision_function(X)
        return np.exp(Z - logsumexp(Z, axis=1, keepdims=True))

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


def fit_sigmoid(f: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling with Platt's target smoothing (as in sklearn's ``_sigmoid_calibration``)."""
    y = y.astype(bool)
    prior1 = float(y.sum())
    prior0 = float(len(y) - prior1)
    t = np.where(y, (prior1 + 1.0) / (prior1 + 2.0), 1.0 / (prior0 + 2.0))

    def objective(ab: np.ndarray) -> tuple[float, np.ndarray]:
        z = ab[0] * f + ab[1]
        # P = expit(-z): log P = log_expit(-z), log(1-P) = log_expit(z)
        loss = -float(np.sum(t * log_expit(-z) + (1 - t) * log_expit(z)))
        dz = t - expit(-z)  # d loss / d z
        return loss, np.array([float(np.sum(dz * f)), float(np.sum(dz))])

    x0 = np.array([0.0, np.log((prior0 + 1.0) / (prior1 + 1.0))])
    res = minimize(objective, x0, jac=True, method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


@dataclass
class _FoldModel:
    estimator: SoftmaxRegression
    sigmoids: list[tuple[float, float]]

    def predict_proba(self, X) -> np.ndarray:
        F = self.estimator.decision_function(X)
        P = np.column_stack([expit(-(a * F[:, i] + b)) for i, (a, b) in enumerate(self.sigmoids)])
        total = P.sum(axis=1, keepdims=True)
        uniform = np.full_like(P, 1.0 / P.shape[1])
        return np.where(total > 0, P / np.where(total > 0, total, 1.0), uniform)


class GroupCalibratedSoftmax(BaseEstimator, ClassifierMixin):
    """Equivalent of ``CalibratedClassifierCV(LogisticRegression(...), method="sigmoid", cv=folds)``."""

    def __init__(self, C: float = 1.0, folds=None, max_iter: int = 2000):
        self.C = C
        self.folds = folds
        self.max_iter = max_iter

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.calibrated_classifiers_: list[_FoldModel] = []
        for train_idx, test_idx in self.folds:
            est = SoftmaxRegression(C=self.C, max_iter=self.max_iter).fit(X[train_idx], y[train_idx])
            if list(est.classes_) != list(self.classes_):
                raise ValueError("every calibration fold must contain all classes")
            F = est.decision_function(X[test_idx])
            sigmoids = [fit_sigmoid(F[:, i], y[test_idx] == cls) for i, cls in enumerate(self.classes_)]
            self.calibrated_classifiers_.append(_FoldModel(est, sigmoids))
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.mean([cc.predict_proba(X) for cc in self.calibrated_classifiers_], axis=0)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


@lru_cache(maxsize=1)
def sklearn_linear_available() -> bool:
    try:
        import sklearn.calibration  # noqa: F401
        import sklearn.linear_model  # noqa: F401
    except ImportError:
        return False
    return True


def backend() -> str:
    """``sklearn`` or ``numpy`` (see module docstring)."""
    choice = os.environ.get("SAFENET_ML_BACKEND", "auto").lower()
    if choice == "numpy":
        return "numpy"
    if choice == "sklearn" or sklearn_linear_available():
        return "sklearn"
    return "numpy"


def make_calibrated_classifier(c: float, folds, seed: int = 42, max_iter: int = 5000):
    """Calibrated multinomial logistic regression for the selected backend."""
    if backend() == "sklearn":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.linear_model import LogisticRegression

        base = LogisticRegression(C=c, class_weight="balanced", max_iter=max_iter, random_state=seed)
        return CalibratedClassifierCV(base, method="sigmoid", cv=folds)
    return GroupCalibratedSoftmax(C=c, folds=folds, max_iter=max_iter)
