"""Scoring configuration: ``config/scoring.yaml`` + thresholds tuned on val."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from safenet import paths


@dataclass
class ScoringConfig:
    rule_cap: float
    ml_weight: float
    network_max: float
    no_core_cap: float
    require_evidence_for_medium: bool
    combos: list[dict]
    medium: float
    high: float
    tuning: dict
    thresholds_source: str = "default"
    thresholds_meta: dict = field(default_factory=dict)


def _read(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_config(path: str = str(paths.SCORING_CONFIG), thresholds_path: str = str(paths.THRESHOLDS_PATH),
                use_tuned: bool = True) -> ScoringConfig:
    raw = _read(Path(path))
    cfg = ScoringConfig(
        rule_cap=float(raw["rule_cap"]),
        ml_weight=float(raw["ml_weight"]),
        network_max=float(raw["network_max"]),
        no_core_cap=float(raw["no_core_cap"]),
        require_evidence_for_medium=bool(raw.get("require_evidence_for_medium", True)),
        combos=list(raw.get("combos", [])),
        medium=float(raw["thresholds"]["medium"]),
        high=float(raw["thresholds"]["high"]),
        tuning=dict(raw.get("tuning", {})),
    )
    tuned = Path(thresholds_path)
    if use_tuned and tuned.exists():
        data = json.loads(tuned.read_text(encoding="utf-8"))
        cfg.medium, cfg.high = float(data["medium"]), float(data["high"])
        cfg.thresholds_source = data.get("source", "tuned")
        cfg.thresholds_meta = data
    return cfg


def reload_config() -> ScoringConfig:
    load_config.cache_clear()
    return load_config()
