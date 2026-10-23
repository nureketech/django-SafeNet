"""Account graph: who shares cards, phones, promo codes, links, domains or groups.

Nodes: ``account``, ``group``, ``card`` and ``phone`` (SHA-256 hashes, masked
labels), ``domain``, ``promo``, ``link``. Edges: account -> entity ("mentions"),
account -> group ("publishes").

Clusters are connected components of the graph after removing noisy nodes
(entities or groups used by more than ``MAX_ENTITY_ACCOUNTS`` accounts — big
public chats, popular sites); components larger than ``LOUVAIN_MIN_ACCOUNTS``
are split with Louvain community detection.

``network_bonus`` (up to +10) is given to a record that is itself at least
``medium`` and belongs to a high-risk cluster (share of high-risk records >=
``MIN_CLUSTER_RISK``) whose accounts share a payment or promotion entity.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities

from safenet.graph.entities import BONUS_ENTITY_TYPES, EntityNode, graph_entities
from safenet.preprocess import normalize

MAX_ENTITY_ACCOUNTS = 12
LOUVAIN_MIN_ACCOUNTS = 15
MIN_CLUSTER_RISK = 0.5
SEED = 42
LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass
class SharedEntity:
    node_id: str
    kind: str
    label: str
    n_accounts: int


@dataclass
class Cluster:
    cluster_id: str
    accounts: list[str]
    record_ids: list[str]
    shared: list[SharedEntity]
    risk_share: float      # share of the cluster's records with level "high"
    mean_score: float
    top_track: str

    @property
    def has_bonus_entity(self) -> bool:
        return any(e.kind in BONUS_ENTITY_TYPES for e in self.shared)


@dataclass
class GraphModel:
    graph: nx.Graph
    clusters: dict[str, Cluster]
    account_cluster: dict[str, str]
    entity_cluster: dict[str, str]
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def bonus(self, cluster: Cluster | None, pre_score: float, medium: float, network_max: float = 10.0) -> float:
        """Network bonus for a record of ``cluster`` whose own score is ``pre_score``."""
        if cluster is None or len(cluster.accounts) < 2 or not cluster.has_bonus_entity:
            return 0.0
        if round(pre_score) < medium or cluster.risk_share < self.params.get("min_cluster_risk", MIN_CLUSTER_RISK):
            return 0.0
        return float(min(network_max, round(network_max * cluster.risk_share, 1)))

    def cluster_of(self, account_id: str | None, entity_ids: list[str]) -> Cluster | None:
        if account_id and f"acc:{account_id}" in self.graph and account_id in self.account_cluster:
            return self.clusters[self.account_cluster[account_id]]
        for node_id in entity_ids:
            cid = self.entity_cluster.get(node_id)
            if cid:
                return self.clusters[cid]
        return None


def _acc(account_id: str) -> str:
    return f"acc:{account_id}"


def build_graph(records: pd.DataFrame, max_entity_accounts: int = MAX_ENTITY_ACCOUNTS,
                louvain_min: int = LOUVAIN_MIN_ACCOUNTS) -> GraphModel:
    """Build the graph from records with ``id, text, account_id, group_id, risk_score, risk_level``."""
    start = time.perf_counter()
    G = nx.Graph()
    account_records: dict[str, list[str]] = defaultdict(list)
    record_score: dict[str, float] = {}
    record_level: dict[str, str] = {}
    record_track: dict[str, str] = {}
    for row in records.itertuples(index=False):
        acc = _acc(row.account_id)
        if acc not in G:
            G.add_node(acc, kind="account", label=f"acc …{str(row.account_id)[-4:]}")
        account_records[row.account_id].append(row.id)
        record_score[row.id] = float(row.risk_score)
        record_level[row.id] = row.risk_level
        record_track[row.id] = getattr(row, "primary_track", "none")
        nodes: list[EntityNode] = graph_entities(normalize(row.text))
        if getattr(row, "group_id", None):
            nodes.append(EntityNode(f"group:{row.group_id}", "group", str(row.group_id)))
        for node in nodes:
            if node.node_id not in G:
                G.add_node(node.node_id, kind=node.kind, label=node.label)
            relation = "publishes" if node.kind == "group" else "mentions"
            if G.has_edge(acc, node.node_id):
                G[acc][node.node_id]["weight"] += 1
            else:
                G.add_edge(acc, node.node_id, relation=relation, weight=1)

    # account-level attributes
    for account_id, rec_ids in account_records.items():
        levels = [record_level[r] for r in rec_ids]
        G.nodes[_acc(account_id)].update(
            n_records=len(rec_ids),
            max_score=max(record_score[r] for r in rec_ids),
            level=max(levels, key=LEVEL_RANK.get),
        )

    # noise removal: entities shared by too many accounts carry no "same operator" information
    noisy = [n for n, d in G.nodes(data=True) if d["kind"] != "account" and G.degree(n) > max_entity_accounts]
    noisy_by_kind = Counter(G.nodes[n]["kind"] for n in noisy)
    G.remove_nodes_from(noisy)

    clusters: dict[str, Cluster] = {}
    groups: list[set[str]] = []
    for component in nx.connected_components(G):
        accounts = {n for n in component if G.nodes[n]["kind"] == "account"}
        if len(accounts) < 2:
            continue
        if len(accounts) > louvain_min:
            sub = G.subgraph(component)
            for community in louvain_communities(sub, weight="weight", seed=SEED):
                acc_in = {n for n in community if G.nodes[n]["kind"] == "account"}
                if len(acc_in) >= 2:
                    groups.append(acc_in)
        else:
            groups.append(accounts)
    groups.sort(key=lambda accs: (-len(accs), min(accs)))

    account_cluster: dict[str, str] = {}
    entity_cluster: dict[str, str] = {}
    for i, accs in enumerate(groups, 1):
        cid = f"C{i:03d}"
        acc_ids = sorted(a.removeprefix("acc:") for a in accs)
        rec_ids = sorted(r for a in acc_ids for r in account_records[a])
        counts: Counter = Counter()
        for acc in accs:
            for nb in G.neighbors(acc):
                counts[nb] += 1
        shared = [
            SharedEntity(n, G.nodes[n]["kind"], G.nodes[n]["label"], c)
            for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if c >= 2
        ]
        tracks = Counter(record_track[r] for r in rec_ids if record_level[r] != "low" and record_track[r] != "none")
        clusters[cid] = Cluster(
            cluster_id=cid,
            accounts=acc_ids,
            record_ids=rec_ids,
            shared=shared,
            risk_share=round(sum(record_level[r] == "high" for r in rec_ids) / len(rec_ids), 3),
            mean_score=round(sum(record_score[r] for r in rec_ids) / len(rec_ids), 1),
            top_track=tracks.most_common(1)[0][0] if tracks else "none",
        )
        for a in acc_ids:
            account_cluster[a] = cid
            G.nodes[_acc(a)]["cluster"] = cid
        for e in shared:
            entity_cluster.setdefault(e.node_id, cid)

    kinds = Counter(d["kind"] for _, d in G.nodes(data=True))
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "nodes_by_kind": dict(kinds),
        "noisy_removed": dict(noisy_by_kind),
        "clusters": len(clusters),
        "build_seconds": round(time.perf_counter() - start, 2),
    }
    params = {"max_entity_accounts": max_entity_accounts, "louvain_min_accounts": louvain_min,
              "min_cluster_risk": MIN_CLUSTER_RISK}
    return GraphModel(G, clusters, account_cluster, entity_cluster, params, stats)
