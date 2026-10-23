"""Stage 7: account graph, clusters, network bonus."""

from __future__ import annotations

import re
import time

import pytest

from safenet import paths
from safenet.graph.build import build_graph
from safenet.graph.entities import graph_entities
from safenet.graph.report import (
    SCORES_PATH,
    all_bonuses,
    load_truth,
    planted_recovery,
    records_with_scores,
    trap_bonuses,
)
from safenet.preprocess import normalize
from safenet.scoring import load_config
from tests.conftest import require

CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,19}(?!\d)")
PHONE_RE = re.compile(r"(?:\+7|(?<!\d)8)[\s(-]*7\d{2}[\s)-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}")


@pytest.fixture(scope="module")
def records():
    require(paths.DATASET_PATH)
    require(paths.MODEL_PATH, "make train")
    return records_with_scores(rescore=not SCORES_PATH.exists())


@pytest.fixture(scope="module")
def model(records):
    return build_graph(records)


@pytest.fixture(scope="module")
def truth():
    return load_truth()


def test_graph_builds_fast(records):
    start = time.perf_counter()
    model = build_graph(records)
    elapsed = time.perf_counter() - start
    assert elapsed < 10, elapsed
    kinds = set(model.stats["nodes_by_kind"])
    assert {"account", "group", "card", "phone", "domain", "promo", "link"} <= kinds


def test_no_plain_card_or_phone_numbers_in_graph(model):
    for node, data in model.graph.nodes(data=True):
        hashed = data["kind"] in ("card", "phone")
        # hashed ids are hex digests (they may contain digit runs by chance) — checked by format instead
        blob = ("" if hashed else node) + " " + " ".join(str(v) for v in data.values())
        assert not CARD_RE.search(blob), blob
        assert not PHONE_RE.search(blob), blob
        if hashed:
            assert "*" in data["label"]
            assert re.fullmatch(rf"{data['kind']}:[0-9a-f]{{24}}", node)


def test_card_nodes_are_hashes_of_digits():
    nodes = graph_entities(normalize("Карта 4400 4302 1234 5678, тел +7 701 234 56 78"))
    by_kind = {n.kind: n for n in nodes}
    assert by_kind["card"].label == "4400 43** **** 5678"
    assert by_kind["phone"].label == "+7 701 *** ** 78"
    assert "4400430212345678" not in by_kind["card"].node_id


def test_unspecific_masks_and_legit_domains_are_not_links():
    nodes = graph_entities(normalize("**** **** **** 1234, +7 701 *** ** 78, итоги через random.org, kaspi.kz"))
    assert not nodes


def test_noisy_entities_are_removed(model):
    assert model.stats["noisy_removed"].get("group", 0) > 0
    for node, data in model.graph.nodes(data=True):
        if data["kind"] != "account":
            assert model.graph.degree(node) <= model.params["max_entity_accounts"]


def test_planted_networks_are_recovered(model, truth):
    recovery = planted_recovery(model, truth)
    assert len(recovery) >= 8
    found = int(recovery["found"].sum())
    assert found >= 6, recovery
    assert found / len(recovery) >= 0.75


def test_legit_shops_with_shared_domain_get_no_bonus(model, records, truth):
    traps = trap_bonuses(model, records, truth)
    assert len(traps) > 0
    assert (traps["risk_score"] < load_config().medium).all()  # their own scores are low
    assert (traps["bonus"] == 0).all(), traps
    assert traps["cluster_id"].notna().any(), "the shops are linked by their shared domain"


def test_network_bonus_range(model, records):
    bonuses = all_bonuses(model, records)
    assert bonuses.between(0, 10).all()
    assert (bonuses > 0).any()


def test_bonus_requires_own_risk(model):
    cluster = next(c for c in model.clusters.values() if c.risk_share >= 0.9 and c.has_bonus_entity)
    assert model.bonus(cluster, pre_score=10, medium=26) == 0
    assert 0 < model.bonus(cluster, pre_score=70, medium=26) <= 10
    assert model.bonus(None, pre_score=90, medium=26) == 0


def test_pipeline_uses_graph(records, truth, monkeypatch):
    from safenet.graph import lookup as graph_lookup
    from safenet.pipeline import analyze

    require(paths.GRAPH_PATH, "make train")
    graph_lookup.load_graph.cache_clear()
    net = next(n for n in truth["networks"] if n["shared_entity_type"] == "phone")
    row = records[records["id"].isin(net["record_ids"]) & (records["risk_level"] == "high")].iloc[0]
    by_account = analyze({"id": row.id, "text": row.text, "account_id": row.account_id})
    assert by_account.network.cluster_id is not None
    assert by_account.network.cluster_size >= 3
    assert 0 < by_account.network.contribution <= 10
    assert abs(by_account.contributions_total() - by_account.risk_score) <= 1
    # an unknown account that reuses the network's phone number is linked through the entity
    phone = next(e.text for e in normalize(row.text).entities if e.kind == "phone")
    new = analyze(f"Вложи 10 000 — через неделю получишь 30 000, гарантированно. Переводите на {phone}")
    assert new.network.cluster_id == by_account.network.cluster_id
    assert not any(phone in s for s in new.network.shared_entities)


def test_cluster_visualisation(model):
    from safenet.graph.viz import cluster_html

    cid = next(iter(model.clusters))
    html = cluster_html(model, cid)
    assert "<html" in html.lower() and "vis.DataSet" in html
    # only the graph data, not the inlined vis.js library (which contains large numeric constants)
    data = " ".join(re.findall(r"new vis\.DataSet\((\[.*?\])\)", html, flags=re.S))
    assert data
    labels = re.findall(r'"label": "([^"]*)"', data)
    assert labels
    for label in labels:
        assert not CARD_RE.search(label) and not PHONE_RE.search(label), label
