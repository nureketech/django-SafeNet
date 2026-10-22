"""Optional baseline: multilingual sentence embeddings + logistic regression (``--embeddings``).

Requires ``sentence-transformers`` (not in ``requirements.txt``: it pulls PyTorch
and downloads the model on first use). When it is missing, the comparison is
skipped and the reason is written to the report.
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np

from safenet.preprocess import normalize

DEFAULT_MODEL = "intfloat/multilingual-e5-small"


def available() -> tuple[bool, str]:
    if importlib.util.find_spec("sentence_transformers") is None:
        return False, "`sentence-transformers` пакеті орнатылмаған (`pip install sentence-transformers`)"
    return True, ""


def evaluate_embeddings(seed: int = 42, model_name: str = DEFAULT_MODEL) -> dict:
    """Train LogReg on embeddings of the train split and evaluate on val."""
    ok, reason = available()
    if not ok:
        return {"status": "skipped", "reason": reason}
    from sentence_transformers import SentenceTransformer
    from sklearn.model_selection import GroupKFold

    from safenet.ml.linear import make_calibrated_classifier
    from safenet.ml.train import evaluate, load_split

    encoder = SentenceTransformer(model_name)
    prefix = "query: " if "e5" in model_name else ""

    def encode(texts):
        return encoder.encode([prefix + normalize(t).text for t in texts], batch_size=32,
                              normalize_embeddings=True, show_progress_bar=False)

    train, val = load_split("train"), load_split("val")
    x_train, x_val = encode(train["text"]), encode(val["text"])
    folds = list(GroupKFold(n_splits=3).split(x_train, train["label"], train["template_id"]))
    clf = make_calibrated_classifier(4.0, folds, seed)
    clf.fit(x_train, np.asarray(train["label"]))
    metrics = evaluate(clf, x_val, val["label"], val["is_hard_negative"])
    start = time.perf_counter()
    for text in val["text"].tolist()[:30]:
        clf.predict_proba(encode([text]))
    latency = (time.perf_counter() - start) / 30 * 1000
    return {"status": "ok", "model": model_name, "macro_f1": metrics["macro_f1"],
            "hard_negative_non_none_rate": metrics["hard_negative_non_none_rate"], "latency_ms": latency}
