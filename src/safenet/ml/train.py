"""``python -m safenet.ml.train [--embeddings]`` — train the classifier.

* fit on **train** only; choose ``C`` on **val** (test and gold are never touched);
* LogisticRegression (lbfgs, i.e. multinomial softmax; ``class_weight="balanced"``)
  calibrated with ``CalibratedClassifierCV`` (sigmoid). The calibration folds are
  grouped by ``template_id`` so calibration is not fooled by template copies;
* save ``models/clf.joblib``, ``models/ml_metrics.json``,
  ``reports/model_report.md`` and ``reports/model_top_features.md``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from safenet import LABELS, paths
from safenet.ml.features import TextAnalyzer, make_vectorizer
from safenet.ml.linear import backend, make_calibrated_classifier

SEED = 42
C_GRID = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
CLASS_ORDER = list(LABELS)  # lottery, casino, pyramid, none


@dataclass
class TrainResult:
    model: Pipeline
    best_c: float
    val_metrics: dict
    grid: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def load_split(name: str) -> pd.DataFrame:
    return pd.read_parquet(paths.split_path(name))


def fit_calibrated(X, y, groups, c: float, seed: int = SEED):
    """Calibrated logistic regression; calibration folds are grouped by template."""
    folds = list(GroupKFold(n_splits=3).split(X, y, groups))
    clf = make_calibrated_classifier(c, folds, seed)
    clf.fit(X, np.asarray(y))
    return clf


def evaluate(clf, X, y: pd.Series, hard_negative: pd.Series) -> dict:
    """Macro-F1, per-class P/R/F1 and the non-``none`` rate on hard negatives."""
    pred = clf.predict(X)
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=CLASS_ORDER, zero_division=0)
    hn = hard_negative.to_numpy()
    return {
        "macro_f1": float(f1_score(y, pred, labels=CLASS_ORDER, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "hard_negative_non_none_rate": float((pred[hn] != "none").mean()) if hn.any() else 0.0,
        "per_class": {
            label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
            for i, label in enumerate(CLASS_ORDER)
        },
    }


def train_model(seed: int = SEED, c_grid: tuple[float, ...] = C_GRID) -> TrainResult:
    """Fit on train, pick ``C`` on val, return the full pipeline (fitted on train)."""
    np.random.seed(seed)
    train, val = load_split("train"), load_split("val")
    t0 = time.perf_counter()
    analyzer = TextAnalyzer().fit(train["text"])
    f_train, f_val = analyzer.transform(train["text"]), analyzer.transform(val["text"])
    vectorizer = make_vectorizer(analyzer.numeric_columns_).fit(f_train)
    x_train, x_val = vectorizer.transform(f_train), vectorizer.transform(f_val)
    t_features = time.perf_counter() - t0

    best = None
    grid = []
    for c in c_grid:
        clf = fit_calibrated(x_train, train["label"], train["template_id"], c, seed)
        metrics = evaluate(clf, x_val, val["label"], val["is_hard_negative"])
        grid.append({"C": c, **{k: v for k, v in metrics.items() if k != "per_class"}})
        key = (round(metrics["macro_f1"], 6), -metrics["hard_negative_non_none_rate"], -c)
        if best is None or key > best[0]:
            best = (key, c, clf, metrics)
    _, best_c, best_clf, val_metrics = best
    model = Pipeline([("analyze", analyzer), ("vectorize", vectorizer), ("clf", best_clf)])
    return TrainResult(model, best_c, val_metrics, grid,
                       {"features_s": round(t_features, 2), "total_s": round(time.perf_counter() - t0, 2)})


def inference_ms(model: Pipeline, texts: list[str], repeats: int = 1) -> float:
    """Average single-record latency of ``predict_proba`` (one text per call)."""
    model.predict_proba(pd.Series(texts[:3]))  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        for t in texts:
            model.predict_proba(pd.Series([t]))
    return (time.perf_counter() - start) / (len(texts) * repeats) * 1000


def top_features(model: Pipeline, n: int = 20) -> dict[str, list[tuple[str, float]]]:
    """Largest averaged logistic-regression coefficients per class."""
    clf = model.named_steps["clf"]
    names = model.named_steps["vectorize"].get_feature_names_out()
    coefs = np.mean([cc.estimator.coef_ for cc in clf.calibrated_classifiers_], axis=0)
    classes = list(clf.classes_)
    out = {}
    for label in CLASS_ORDER:
        row = coefs[classes.index(label)]
        idx = np.argsort(row)[::-1][:n]
        out[label] = [(str(names[i]), float(row[i])) for i in idx]
    return out


def _fmt_feature(name: str) -> str:
    kind, _, value = name.partition("__")
    label = {"char": "символдар", "word": "сөздер", "num": "белгі/сан"}.get(kind, kind)
    return f"`{value}` ({label})"


def write_reports(result: TrainResult, latency_ms: float, embeddings: dict | None) -> None:
    m = result.val_metrics
    lines = [
        "# ML-классификатор: есеп",
        "",
        "`python -m safenet.ml.train` арқылы жасалды. Модель **train** жинағында оқытылды, `C` гиперпараметрі "
        "**val** бойынша таңдалды. test пен gold қолданылған жоқ.",
        "",
        "- Белгілер: TF-IDF символдық n-граммдар (2–5, `char_wb`, нормаланған мәтін) + сөздік n-граммдар (1–2) "
        "+ 3-кезеңнің сандық белгілері (белгілер жалаушасы, есептелген табыс, нысандар саны).",
        "- Модель: `LogisticRegression` (lbfgs, softmax; `class_weight=\"balanced\"`) + "
        "`CalibratedClassifierCV(method=\"sigmoid\")`. Калибрлеу бөлінісі `template_id` бойынша (GroupKFold=3).",
        f"- Іске асыру (backend): **{backend()}** — `sklearn` немесе дәл сол алгоритмнің numpy/scipy нұсқасы "
        "(`safenet/ml/linear.py`); sklearn-нің сызықтық модельдері импортталмайтын машинада соңғысы қолданылады.",
        f"- Таңдалған `C` = **{result.best_c}**",
        f"- val macro-F1: **{m['macro_f1']:.3f}**; accuracy: {m['accuracy']:.3f}",
        f"- val қиын негативтерінде «none емес» болжам үлесі: **{m['hard_negative_non_none_rate'] * 100:.1f}%**",
        f"- Бір жазбаның инференсі (нормализация + белгілер + модель): **{latency_ms:.1f} мс**",
        f"- Уақыт: белгілерді есептеу {result.timings['features_s']} с, барлығы {result.timings['total_s']} с",
        "",
        "## val: кластар бойынша",
        "",
        "| класс | precision | recall | F1 | саны |",
        "|---|---:|---:|---:|---:|",
        *[f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |"
          for k, v in m["per_class"].items()],
        "",
        "## `C` бойынша іздеу (val)",
        "",
        "| C | macro-F1 | accuracy | қиын негативтер: none емес |",
        "|---:|---:|---:|---:|",
        *[f"| {g['C']} | {g['macro_f1']:.3f} | {g['accuracy']:.3f} | {g['hard_negative_non_none_rate'] * 100:.1f}% |"
          for g in result.grid],
        "",
        "## Эмбеддингтермен салыстыру (`--embeddings`)",
        "",
    ]
    if embeddings and embeddings.get("status") == "ok":
        lines += [
            f"- Модель: `{embeddings['model']}` + LogisticRegression",
            f"- val macro-F1: **{embeddings['macro_f1']:.3f}** (TF-IDF гибриді: {m['macro_f1']:.3f})",
            f"- Қиын негативтерде «none емес»: {embeddings['hard_negative_non_none_rate'] * 100:.1f}%",
            f"- Инференс: {embeddings['latency_ms']:.1f} мс/жазба",
        ]
    else:
        reason = (embeddings or {}).get("reason", "`--embeddings` жалаушасы берілмеді")
        lines += [f"Іске қосылмады: {reason}."]
    lines.append("")
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "model_report.md").write_text("\n".join(lines), encoding="utf-8")

    feats = top_features(result.model)
    lines = [
        "# Модельдің ең маңызды белгілері",
        "",
        "Логистикалық регрессия коэффициенттері (калибрлеу фолдтары бойынша орташа). Әр класс үшін бұл "
        "белгілер ықтималдықты ең көп арттырады. `символдар` — бүктелген мәтіндегі символдық n-грамм, "
        "`сөздер` — сөз немесе сөз жұбы, `белгі/сан` — 3-кезеңнің белгісі немесе сандық белгі.",
        "",
    ]
    for label, items in feats.items():
        lines += [f"## {label}", "", "| # | белгі | салмақ |", "|---:|---|---:|"]
        lines += [f"| {i} | {_fmt_feature(name)} | {w:.3f} |" for i, (name, w) in enumerate(items, 1)]
        lines.append("")
    (paths.REPORTS_DIR / "model_top_features.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", action="store_true", help="also evaluate a sentence-embedding baseline")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    result = train_model(args.seed)
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, paths.MODEL_PATH, compress=3)
    val = load_split("val")
    latency = inference_ms(result.model, val["text"].tolist()[:60])

    embeddings = None
    if args.embeddings:
        from safenet.ml.embeddings import evaluate_embeddings

        embeddings = evaluate_embeddings(args.seed)
    write_reports(result, latency, embeddings)
    metrics = {"backend": backend(), "best_c": result.best_c, "val": result.val_metrics, "grid": result.grid,
               "latency_ms": latency, "embeddings": embeddings, "timings": result.timings}
    (paths.MODELS_DIR / "ml_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")
    print(f"saved {paths.MODEL_PATH.name} [{backend()}]: C={result.best_c}, "
          f"val macro-F1={result.val_metrics['macro_f1']:.3f}, "
          f"hard-negative non-none={result.val_metrics['hard_negative_non_none_rate']:.3f}, "
          f"latency={latency:.1f} ms")


if __name__ == "__main__":
    main()
