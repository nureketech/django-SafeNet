"""``python -m safenet.preprocess`` — preprocessing statistics on the synthetic dataset.

Writes ``reports/preprocess_report.md``: speed, language-detection accuracy and
entity recall against the generator's ground truth.
"""

from __future__ import annotations

import time
from collections import Counter

import pandas as pd

from safenet import paths
from safenet.common import to_entities
from safenet.preprocess import normalize


def main() -> None:
    df = pd.read_parquet(paths.DATASET_PATH, columns=["text", "lang", "entities", "obfuscations"])
    start = time.perf_counter()
    results = [normalize(t) for t in df["text"]]
    avg_ms = (time.perf_counter() - start) / len(df) * 1000
    df["pred_lang"] = [r.lang for r in results]
    accuracy = (df["pred_lang"] == df["lang"]).mean()
    per_lang = (df["pred_lang"] == df["lang"]).groupby(df["lang"]).mean()
    confusion = pd.crosstab(df["lang"], df["pred_lang"])

    tp: Counter = Counter()
    total: Counter = Counter()
    for r, ents in zip(results, df["entities"].map(to_entities)):
        got = r.entity_dict()
        for key, values in ents.items():
            total[key] += len(values)
            tp[key] += len(set(values) & set(got[key]))

    lines = [
        "# Алдын ала өңдеу: статистика",
        "",
        "`python -m safenet.preprocess` арқылы синтетикалық датасетте есептелген.",
        "",
        f"- Жазбалар: {len(df)}",
        f"- Бір жазбаға орташа уақыт: **{avg_ms:.2f} мс**",
        f"- Тілді анықтау дәлдігі: **{accuracy * 100:.1f}%**",
        "",
        "| тіл | дәлдік, % |",
        "|---|---:|",
        *[f"| {lang} | {value * 100:.1f} |" for lang, value in per_lang.items()],
        "",
        "Қателер матрицасы (жол — эталон, баған — болжам):",
        "",
        "| эталон \\ болжам | " + " | ".join(confusion.columns) + " |",
        "|---|" + "---:|" * len(confusion.columns),
        *[f"| {idx} | " + " | ".join(str(v) for v in row) + " |" for idx, row in confusion.iterrows()],
        "",
        "Нысандарды табу (генератор салған нысандар бойынша recall):",
        "",
        "| нысан | эталон | табылды | recall, % |",
        "|---|---:|---:|---:|",
        *[f"| {k} | {total[k]} | {tp[k]} | {tp[k] / total[k] * 100:.1f} |" for k in total if total[k]],
        "",
    ]
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "preprocess_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
