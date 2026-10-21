"""Filesystem locations used across the project."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("SAFENET_ROOT", Path(__file__).resolve().parents[2]))

CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
GOLD_DIR = DATA_DIR / "gold"
IMAGES_DIR = DATA_DIR / "images"
FEEDBACK_DIR = DATA_DIR / "feedback"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

DATASET_PATH = SYNTHETIC_DIR / "dataset.parquet"
NETWORKS_PATH = SYNTHETIC_DIR / "networks.json"
GOLD_PATH = GOLD_DIR / "gold.parquet"
SIGNALS_CONFIG = CONFIG_DIR / "signals.yaml"
SCORING_CONFIG = CONFIG_DIR / "scoring.yaml"
MODEL_PATH = MODELS_DIR / "clf.joblib"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
GRAPH_PATH = MODELS_DIR / "graph.joblib"
FEEDBACK_PATH = FEEDBACK_DIR / "feedback.csv"


def split_path(split: str) -> Path:
    """Return the parquet path for a ``train`` / ``val`` / ``test`` split."""
    return SYNTHETIC_DIR / f"{split}.parquet"
