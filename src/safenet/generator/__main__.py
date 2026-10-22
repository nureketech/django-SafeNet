"""``python -m safenet.generator`` — build dataset, splits, screenshots, gold set and report."""

from __future__ import annotations

import time

from safenet.generator.build import build_dataset, save_dataset
from safenet.generator.gold import build_gold, save_gold
from safenet.generator.report import build_report, save_report


def main() -> None:
    start = time.perf_counter()
    df, truth = build_dataset()
    save_dataset(df, truth)
    gold = build_gold()
    save_gold(gold)
    save_report(build_report(df, gold, truth))
    print(
        f"dataset: {len(df)} records, {df['template_id'].nunique()} templates, "
        f"{int(df['image_path'].notna().sum())} screenshots, {len(truth['networks'])} networks; "
        f"gold: {len(gold)}; {time.perf_counter() - start:.1f}s"
    )


if __name__ == "__main__":
    main()
