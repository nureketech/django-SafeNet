"""Live account graph of the platform, built only from real saved cases (checks and API calls).

The research graph ``models/graph.joblib`` (stage 7) comes from the synthetic dataset and serves the
evaluation only. The dashboard and the API link cases through this graph instead. It is rebuilt when the
saved cases change, which takes milliseconds for thousands of cases.

A case whose account is unknown is its own node (``post:<case id>``), so two posts that share a card,
phone, promo code, link, domain or chat are linked even without an account.
"""

from __future__ import annotations

import dataclasses
import hashlib
import threading
from dataclasses import dataclass, field

import pandas as pd
from django.db.models import Count, Max

from app.dashboard.models import Case
from safenet.common import digits_only, mask_phone
from safenet.graph.build import Cluster, GraphModel, SharedEntity, build_graph
from safenet.graph.entities import EntityNode, graph_entities
from safenet.graph.lookup import make_lookup
from safenet.preprocess import normalize

REAL_ORIGINS = ("check", "api")
POST_PREFIX = "post:"
RELATED_LIMIT = 10

_lock = threading.Lock()
_cache: dict = {"version": None, "model": None}


def account_key(account_id: str, case_id: str) -> str:
    """Graph account of a case: its account, or the post itself when the account is unknown."""
    return account_id or f"{POST_PREFIX}{case_id}"


def account_label(key: str) -> str:
    """What the interface shows for an account node (phone numbers only masked)."""
    if key.startswith(POST_PREFIX):
        return f"пост …{key[-6:]}"
    if len(digits_only(key)) >= 10 and not any(ch.isalpha() for ch in key):
        return mask_phone(key)
    return key


def case_text(case: Case) -> str:
    return "\n".join(p for p in (case.text, case.result.get("ocr_text") or "") if p)


def _real_cases():
    return Case.objects.filter(origin__in=REAL_ORIGINS)


def _records() -> pd.DataFrame:
    rows = [
        {"id": c.id, "text": case_text(c), "account_id": account_key(c.account_id, c.id), "group_id": c.group_id,
         "risk_score": c.risk_score, "risk_level": c.risk_level, "primary_track": c.primary_track}
        for c in _real_cases()
    ]
    return pd.DataFrame(rows)


def _stable_ids(model: GraphModel) -> GraphModel:
    """Cluster ids that survive rebuilds (derived from the cluster's first account, not its rank)."""
    rename: dict[str, str] = {}
    for cid, cluster in model.clusters.items():
        new = "N" + hashlib.sha1(cluster.accounts[0].encode("utf-8")).hexdigest()[:6].upper()
        while new in rename.values():
            new += "X"
        rename[cid] = new
    model.clusters = {rename[c]: dataclasses.replace(cl, cluster_id=rename[c]) for c, cl in model.clusters.items()}
    model.account_cluster = {a: rename[c] for a, c in model.account_cluster.items()}
    model.entity_cluster = {e: rename[c] for e, c in model.entity_cluster.items()}
    for node, data in model.graph.nodes(data=True):
        if data.get("kind") == "account":
            data["label"] = account_label(node.removeprefix("acc:"))
            if "cluster" in data:
                data["cluster"] = rename[data["cluster"]]
    return model


def live_graph() -> GraphModel | None:
    """Graph of the saved real cases; ``None`` while there are none."""
    version = tuple(_real_cases().aggregate(n=Count("id"), last=Max("analysed_at")).values())
    if not version[0]:
        return None
    with _lock:
        if _cache["version"] != version:
            _cache.update(version=version, model=_stable_ids(build_graph(_records())))
        return _cache["model"]


def cluster_of_case(model: GraphModel | None, case: Case) -> Cluster | None:
    """Network a saved case belongs to (queue badge); synthetic cases never belong to one."""
    if model is None or case.origin not in REAL_ORIGINS:
        return None
    cid = model.account_cluster.get(account_key(case.account_id, case.id))
    return model.clusters.get(cid) if cid else None


def network_name(cluster: Cluster, limit: int = 3) -> str:
    """A network is named after its accounts: «@fugu_kz, @gizbocasino_kz»."""
    names = [account_label(a) for a in cluster.accounts]
    return ", ".join(names[:limit]) + (f" +{len(names) - limit}" if len(names) > limit else "")


def live_lookup():
    """``network_lookup`` for ``pipeline.analyze`` over the live graph (``None`` = no network bonus)."""
    model = live_graph()
    return make_lookup(model) if model is not None else None


@dataclass
class Links:
    """What the case card shows under «Байланыстар»."""

    key: str | None = None  # the graph account of the content (None: checked, not saved, no account)
    account_id: str = ""
    own: list[EntityNode] = field(default_factory=list)  # cards, phones, promo codes, sites found in the content
    cluster: Cluster | None = None
    entities: list[SharedEntity] = field(default_factory=list)
    groups: list[SharedEntity] = field(default_factory=list)
    cluster_cases: list[Case] = field(default_factory=list)
    account_cases: list[Case] = field(default_factory=list)

    @property
    def other_accounts(self) -> list[str]:
        return [a for a in self.cluster.accounts if a != self.key] if self.cluster else []


def links_for(*, account_id: str = "", text: str = "", case_id: str | None = None, group_id: str = "") -> Links:
    """Live links of a saved case (``case_id``) or of content that was only checked."""
    key = account_key(account_id, case_id) if (account_id or case_id) else None
    own = graph_entities(normalize(text))
    if group_id:
        own.append(EntityNode(f"group:{group_id}", "group", group_id))
    links = Links(key=key, account_id=account_id, own=own)
    others = _real_cases().exclude(pk=case_id) if case_id else _real_cases()
    if account_id:
        links.account_cases = list(others.filter(account_id=account_id).order_by("-analysed_at")[:RELATED_LIMIT])
    model = live_graph()
    if model is None:
        return links
    cluster = model.cluster_of(key, [e.node_id for e in own])
    if cluster is not None:
        links.cluster = cluster
        links.entities = [e for e in cluster.shared if e.kind != "group"]
        links.groups = [e for e in cluster.shared if e.kind == "group"]
        in_cluster = others.filter(pk__in=cluster.record_ids)
        if account_id:  # the account's own posts are listed separately
            in_cluster = in_cluster.exclude(account_id=account_id)
        links.cluster_cases = list(in_cluster.order_by("-risk_score")[:RELATED_LIMIT])
    return links
