"""Score the dataset, build and save the graph, measure recovery of the planted networks."""

from __future__ import annotations

import json

import pandas as pd

from safenet import paths
from safenet.graph.build import GraphModel, build_graph
from safenet.scoring.config import load_config

SCORES_PATH = paths.MODELS_DIR / "dataset_scores.parquet"
MIN_COVERAGE = 0.7


def score_records(df: pd.DataFrame) -> pd.DataFrame:
    """Risk scores of every record *before* the network bonus (cached in ``models/dataset_scores.parquet``)."""
    from safenet.pipeline import analyze_many

    results = analyze_many(
        ({"id": r.id, "text": r.text, "source_type": r.source_type, "account_id": r.account_id}
         for r in df.itertuples()),
        use_ocr=False,
        network_lookup=None,
    )
    scores = pd.DataFrame({
        "id": [r.id for r in results],
        "risk_score": [r.risk_score for r in results],
        "risk_level": [r.risk_level for r in results],
        "primary_track": [r.primary_track for r in results],
    })
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_PATH, index=False)
    return scores


def records_with_scores(df: pd.DataFrame | None = None, rescore: bool = False) -> pd.DataFrame:
    df = df if df is not None else pd.read_parquet(paths.DATASET_PATH)
    scores = pd.read_parquet(SCORES_PATH) if SCORES_PATH.exists() and not rescore else score_records(df)
    return df.merge(scores, on="id", how="inner")


def planted_recovery(model: GraphModel, truth: dict) -> pd.DataFrame:
    """For every planted network: the cluster holding most of its accounts and the covered share."""
    rows = []
    for net in truth["networks"]:
        members = net["accounts"]
        counts: dict[str, int] = {}
        for acc in members:
            cid = model.account_cluster.get(acc)
            if cid:
                counts[cid] = counts.get(cid, 0) + 1
        best = max(counts, key=counts.get) if counts else None
        coverage = counts[best] / len(members) if best else 0.0
        cluster = model.clusters.get(best) if best else None
        rows.append({
            "network_id": net["network_id"],
            "track": net["track"],
            "shared_entity": net["shared_entity_type"],
            "accounts": len(members),
            "cluster_id": best,
            "cluster_accounts": len(cluster.accounts) if cluster else 0,
            "coverage": round(coverage, 3),
            "found": coverage >= MIN_COVERAGE,
            "cluster_risk": cluster.risk_share if cluster else None,
        })
    return pd.DataFrame(rows)


def trap_bonuses(model: GraphModel, records: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """Network bonus of the legit shops that share a website (must be 0 while their own score is low)."""
    cfg = load_config()
    rows = []
    for trap in truth.get("legit_shared_domains", []):
        part = records[records["id"].isin(trap["record_ids"])]
        for row in part.itertuples():
            cluster = model.cluster_of(row.account_id, [])
            rows.append({
                "trap_id": trap["trap_id"], "id": row.id, "risk_score": row.risk_score,
                "cluster_id": cluster.cluster_id if cluster else None,
                "bonus": model.bonus(cluster, row.risk_score, cfg.medium, cfg.network_max),
            })
    return pd.DataFrame(rows)


def all_bonuses(model: GraphModel, records: pd.DataFrame) -> pd.Series:
    cfg = load_config()
    return pd.Series(
        [model.bonus(model.cluster_of(r.account_id, []), r.risk_score, cfg.medium, cfg.network_max)
         for r in records.itertuples()],
        index=records.index,
    )


def load_truth() -> dict:
    return json.loads(paths.NETWORKS_PATH.read_text(encoding="utf-8"))


def build_from_dataset(rescore: bool = False) -> tuple[GraphModel, pd.DataFrame]:
    records = records_with_scores(rescore=rescore)
    return build_graph(records), records
