"""Load the trained classifier and predict track probabilities."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable

import joblib
import pandas as pd

from safenet import LABELS, paths


class ModelNotTrained(RuntimeError):
    """Raised when ``models/clf.joblib`` is missing."""


@lru_cache(maxsize=2)
def load_model(path: str = str(paths.MODEL_PATH)):
    try:
        return joblib.load(path)
    except FileNotFoundError as exc:
        raise ModelNotTrained(f"{path} not found — run `make train`") from exc


def predict_proba(texts: Iterable[str], model=None) -> pd.DataFrame:
    """Probabilities of ``lottery / casino / pyramid / none`` for each text (rows sum to 1)."""
    model = model or load_model()
    texts = pd.Series(list(texts), dtype=object)
    proba = model.predict_proba(texts)
    classes = list(model.classes_)
    df = pd.DataFrame(proba, columns=classes)
    return df[list(LABELS)]
