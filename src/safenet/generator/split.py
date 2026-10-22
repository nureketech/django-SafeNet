"""Leakage-free train/val/test split (60/20/20) grouped by ``template_id``.

Every template belongs to one phrase family (``template_fold``); a whole family
goes to one split, so a template — and even a phrase variant — never appears in
two splits. Families are balanced by label and language at generation time,
which gives the stratification.
"""

from __future__ import annotations

import pandas as pd

from safenet.generator.templates import FOLD_SPLIT

SPLITS = ("train", "val", "test")


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``split`` column; raises if a template would fall into two splits."""
    folds_per_template = df.groupby("template_id")["template_fold"].nunique()
    if (folds_per_template > 1).any():
        bad = folds_per_template[folds_per_template > 1].index[:5].tolist()
        raise ValueError(f"templates with several phrase families: {bad}")
    out = df.copy()
    out["split"] = out["template_fold"].map(FOLD_SPLIT)
    return out


def leakage(df: pd.DataFrame) -> dict[str, int]:
    """Number of shared template ids between every pair of splits (must be all zeros)."""
    ids = {s: set(df.loc[df["split"] == s, "template_id"]) for s in SPLITS}
    return {
        f"{a}&{b}": len(ids[a] & ids[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1:]
    }
