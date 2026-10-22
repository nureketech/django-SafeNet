"""Feature extraction for the classifier.

``TextAnalyzer`` runs the stage-2 normaliser and the stage-3 detector once per
text and returns a DataFrame with the folded normalised text (for TF-IDF) and
numeric columns (signal flags, computed return, entity counts).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from safenet.preprocess import normalize
from safenet.signals import get_detector

EXTRA_COLUMNS = [
    "monthly_return", "n_links", "n_amounts", "max_amount_log", "n_percent", "has_card", "has_phone",
    "has_promo", "has_kaspi", "has_ref", "has_shortener",
]


def numeric_features(text: str, signal_ids: list[str]) -> tuple[str, dict[str, float]]:
    """Folded normalised text and numeric features of one raw text."""
    nt = normalize(text)
    hits = get_detector().detect(nt)
    fired = {h.id: h for h in hits}
    row: dict[str, float] = {f"sig:{sid}": float(sid in fired) for sid in signal_ids}
    monthly = fired["P_UNREAL_RETURN"].details.get("monthly_pct", 0.0) if "P_UNREAL_RETURN" in fired else 0.0
    amounts = nt.amounts()
    kinds = [e.kind for e in nt.entities]
    row.update({
        "monthly_return": min(math.log1p(max(monthly, 0.0)) / math.log1p(1000.0), 1.0),
        "n_links": float(kinds.count("url") + kinds.count("domain")),
        "n_amounts": float(len(amounts)),
        "max_amount_log": math.log10(max(amounts)) / 7.0 if amounts else 0.0,
        "n_percent": float(kinds.count("percent")),
        "has_card": float("card" in kinds),
        "has_phone": float("phone" in kinds),
        "has_promo": float("promo_code" in kinds),
        "has_kaspi": float("kaspi" in kinds),
        "has_ref": float("ref_param" in kinds),
        "has_shortener": float("shortener" in kinds),
    })
    return nt.folded, row


class TextAnalyzer(BaseEstimator, TransformerMixin):
    """Raw texts -> DataFrame(``folded`` + numeric feature columns)."""

    def fit(self, X, y=None):
        self.signal_ids_ = [s.id for s in get_detector().signals]
        self.numeric_columns_ = [f"sig:{sid}" for sid in self.signal_ids_] + EXTRA_COLUMNS
        return self

    def transform(self, X) -> pd.DataFrame:
        texts = list(X)
        folded: list[str] = []
        rows: list[dict[str, float]] = []
        for text in texts:
            f, row = numeric_features(text or "", self.signal_ids_)
            folded.append(f)
            rows.append(row)
        df = pd.DataFrame(rows, columns=self.numeric_columns_, dtype=np.float32)
        df.insert(0, "folded", folded)
        return df


class FeatureStack(BaseEstimator, TransformerMixin):
    """TF-IDF over characters (2-5, word-bounded) and words (1-2) + scaled numeric features.

    A small replacement for ``ColumnTransformer`` (``sklearn.compose`` cannot be
    imported where sklearn's compiled linear models are blocked).
    """

    def __init__(self, numeric_columns: list[str] | None = None, max_char: int = 60000, max_word: int = 30000):
        self.numeric_columns = numeric_columns
        self.max_char = max_char
        self.max_word = max_word

    def fit(self, X: pd.DataFrame, y=None):
        self.char_ = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                     max_features=self.max_char, dtype=np.float32).fit(X["folded"])
        self.word_ = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                                     token_pattern=r"(?u)\b\w+\b", max_features=self.max_word,
                                     dtype=np.float32).fit(X["folded"])
        self.scaler_ = MaxAbsScaler().fit(X[list(self.numeric_columns)].to_numpy(dtype=np.float32))
        return self

    def transform(self, X: pd.DataFrame):
        numeric = self.scaler_.transform(X[list(self.numeric_columns)].to_numpy(dtype=np.float32))
        return sparse.hstack([
            self.char_.transform(X["folded"]),
            self.word_.transform(X["folded"]),
            sparse.csr_matrix(numeric.astype(np.float32)),
        ]).tocsr()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(
            [f"char__{n}" for n in self.char_.get_feature_names_out()]
            + [f"word__{n}" for n in self.word_.get_feature_names_out()]
            + [f"num__{c}" for c in self.numeric_columns],
            dtype=object,
        )


def make_vectorizer(numeric_columns: list[str], max_char: int = 60000, max_word: int = 30000) -> FeatureStack:
    return FeatureStack(list(numeric_columns), max_char, max_word)
