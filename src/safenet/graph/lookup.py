"""Saved graph and the ``network_lookup`` hook used by ``pipeline.analyze``."""

from __future__ import annotations

from functools import lru_cache

import joblib

from safenet import paths
from safenet.graph.build import GraphModel
from safenet.graph.entities import graph_entities
from safenet.scoring.config import load_config
from safenet.scoring.models import NetworkOut


def save_graph(model: GraphModel, path=paths.GRAPH_PATH) -> None:
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


@lru_cache(maxsize=2)
def load_graph(path: str = str(paths.GRAPH_PATH)) -> GraphModel | None:
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None


def make_lookup(model: GraphModel):
    """``network_lookup(item, nt, hits, pre_score) -> NetworkOut`` backed by ``model``."""

    def lookup(item, nt, hits, pre_score: float) -> NetworkOut:
        cfg = load_config()
        entity_ids = [e.node_id for e in graph_entities(nt)]
        cluster = model.cluster_of(item.account_id, entity_ids)
        if cluster is None:
            return NetworkOut()
        bonus = model.bonus(cluster, pre_score, cfg.medium, cfg.network_max)
        return NetworkOut(
            cluster_id=cluster.cluster_id,
            cluster_size=len(cluster.accounts),
            cluster_risk=cluster.risk_share,
            shared_entities=[f"{e.kind}: {e.label}" for e in cluster.shared if e.kind != "group"][:8],
            contribution=bonus,
        )

    return lookup


def default_lookup():
    """Lookup over ``models/graph.joblib`` or ``None`` when the graph has not been built."""
    model = load_graph()
    return make_lookup(model) if model is not None else None
