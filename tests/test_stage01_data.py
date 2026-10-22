"""Stage 1: synthetic dataset, leakage-free splits, screenshots, networks and gold set."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from safenet import paths
from safenet.common import dedup_key, find_forbidden, luhn_valid, to_entities, to_list
from tests.conftest import require

SOURCE_TYPES = {"post", "chat_message", "stream_transcript", "screenshot", "account_bio"}
PLATFORMS = {"telegram", "whatsapp", "instagram", "tiktok", "youtube", "vk"}
LANGS = {"ru", "kk", "mixed", "translit"}
LABELS = {"lottery", "casino", "pyramid", "none"}
RISKS = {"low", "medium", "high"}
REQUIRED_FIELDS = [
    "id", "text", "source_type", "platform", "lang", "label", "is_hard_negative", "risk_gold",
    "gold_signals", "obfuscations", "template_id", "account_id", "group_id", "entities",
    "image_path", "created_at", "synthetic",
]
TARGET_SHARE = {"none": 0.40, "lottery": 0.20, "casino": 0.20, "pyramid": 0.20}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    require(paths.DATASET_PATH)
    return pd.read_parquet(paths.DATASET_PATH)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    for split in ("train", "val", "test"):
        require(paths.split_path(split))
    return {s: pd.read_parquet(paths.split_path(s)) for s in ("train", "val", "test")}


@pytest.fixture(scope="module")
def gold() -> pd.DataFrame:
    require(paths.GOLD_PATH)
    return pd.read_parquet(paths.GOLD_PATH)


@pytest.fixture(scope="module")
def truth() -> dict:
    require(paths.NETWORKS_PATH)
    return json.loads(paths.NETWORKS_PATH.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- schema
def test_schema_fields(df):
    missing = [f for f in REQUIRED_FIELDS if f not in df.columns]
    assert not missing, missing


def test_schema_types_and_enums(df):
    assert df["id"].is_unique
    assert df["text"].map(lambda t: isinstance(t, str) and t.strip() != "").all()
    assert set(df["source_type"]) <= SOURCE_TYPES
    assert set(df["platform"]) <= PLATFORMS
    assert set(df["lang"]) <= LANGS
    assert set(df["label"]) == LABELS
    assert set(df["risk_gold"]) <= RISKS
    assert df["is_hard_negative"].dtype == bool
    assert pd.api.types.is_datetime64_any_dtype(df["created_at"])
    assert df["synthetic"].astype(bool).all()
    assert df["gold_signals"].map(lambda v: all(isinstance(x, str) for x in to_list(v))).all()
    assert df["obfuscations"].map(lambda v: all(isinstance(x, str) for x in to_list(v))).all()
    ents = df["entities"].map(to_entities)
    assert ents.map(lambda e: set(e) == {"cards", "phones", "domains", "promo_codes", "links"}).all()


def test_risk_gold_consistent_with_label(df):
    assert (df.loc[df["label"] == "none", "risk_gold"] == "low").all()
    schemes = df[df["label"] != "none"]
    assert set(schemes["risk_gold"]) == {"medium", "high"}
    assert 0.15 <= (schemes["risk_gold"] == "medium").mean() <= 0.5
    assert schemes["gold_signals"].map(lambda v: len(to_list(v)) > 0).all()


# ------------------------------------------------------------ volume/balance
def test_size(df):
    assert len(df) >= 2000


@pytest.mark.parametrize("label,target", TARGET_SHARE.items())
def test_class_balance(df, label, target):
    share = (df["label"] == label).mean()
    assert abs(share - target) <= 0.05, f"{label}: {share:.3f}"


def test_languages(df):
    share = df["lang"].value_counts(normalize=True)
    assert share["kk"] >= 0.25
    assert share["translit"] >= 0.10
    assert 0.35 <= share["ru"] <= 0.55
    assert 0.08 <= share["mixed"] <= 0.22


def test_source_types(df):
    counts = df["source_type"].value_counts()
    assert set(counts.index) == SOURCE_TYPES
    assert (counts / len(df) >= 0.05).all(), counts.to_dict()
    assert counts["stream_transcript"] >= 150
    assert counts["screenshot"] >= 80


def test_hard_negatives(df):
    none = df[df["label"] == "none"]
    assert none["is_hard_negative"].mean() >= 0.5
    assert not df.loc[df["label"] != "none", "is_hard_negative"].any()


def test_obfuscation_shares(df):
    obf = df["obfuscations"].map(lambda v: len(to_list(v)) > 0)
    schemes = obf[df["label"] != "none"].mean()
    negatives = obf[df["label"] == "none"].mean()
    assert 0.30 <= schemes <= 0.40, schemes
    assert 0.05 <= negatives <= 0.15, negatives


# --------------------------------------------------------------- duplicates
def test_no_exact_duplicates(df):
    assert not df["text"].duplicated().any()


def test_near_duplicates_below_5_percent(df):
    assert df["text"].map(dedup_key).duplicated().mean() < 0.05


def test_no_forbidden_words_in_texts(df):
    hits = df["text"].map(find_forbidden)
    assert not hits.map(bool).any(), df.loc[hits.map(bool), "text"].head().tolist()


# ------------------------------------------------------------------- splits
def test_split_sizes(df, splits):
    n = sum(len(s) for s in splits.values())
    assert n == len(df)
    for name, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(len(splits[name]) / n - target) <= 0.03, name


def test_no_template_leakage(splits):
    ids = {name: set(s["template_id"]) for name, s in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_splits_stratified(splits):
    for name, part in splits.items():
        assert set(part["label"]) == LABELS, name
        assert set(part["lang"]) == LANGS, name
        for label, target in TARGET_SHARE.items():
            assert abs((part["label"] == label).mean() - target) <= 0.05, (name, label)


@pytest.mark.parametrize("label", sorted(LABELS))
@pytest.mark.parametrize("lang", ["ru", "kk"])
def test_templates_per_class_and_language(df, label, lang):
    n = df[(df["label"] == label) & (df["lang"] == lang)]["template_id"].nunique()
    assert n >= 30, f"{label}/{lang}: {n} templates"


# --------------------------------------------------------------- screenshots
def test_screenshots_exist_and_open(df):
    shots = df[df["source_type"] == "screenshot"]
    assert shots["image_path"].notna().all()
    assert df.loc[df["source_type"] != "screenshot", "image_path"].isna().all()
    for rel in shots["image_path"]:
        path = paths.ROOT / rel
        assert path.exists(), rel
        with Image.open(path) as img:
            img.verify()


# ------------------------------------------------------------------ entities
def test_cards_fail_luhn(df):
    cards = [c for e in df["entities"].map(to_entities) for c in e["cards"]]
    full = [c for c in cards if "*" not in c]
    assert full, "dataset must contain some unmasked fictional cards"
    assert any("*" in c for c in cards), "dataset must contain masked cards"
    assert not [c for c in full if luhn_valid(c)]


def test_cards_and_phones_appear_in_text(df):
    for text, ents in zip(df["text"], df["entities"].map(to_entities)):
        for value in ents["cards"] + ents["phones"] + ents["promo_codes"] + ents["links"]:
            assert value in text


def test_domains_are_fictional(df):
    domains = {d for e in df["entities"].map(to_entities) for d in e["domains"]}
    assert domains
    bad = [d for d in domains if not d.endswith((".test", ".example"))]
    assert not bad, bad[:5]


# ------------------------------------------------------------------ networks
def test_networks(df, truth):
    networks = truth["networks"]
    assert len(networks) >= 8
    accounts = set(df["account_id"])
    ids = set(df["id"])
    for net in networks:
        assert 3 <= len(net["accounts"]) <= 10, net["network_id"]
        assert set(net["accounts"]) <= accounts, net["network_id"]
        assert set(net["record_ids"]) <= ids
        members = df[df["id"].isin(net["record_ids"])]
        assert set(members["account_id"]) == set(net["accounts"])
        assert (members["label"] == net["track"]).all()


def test_networks_share_entity_in_text(df, truth):
    from safenet.common import sha256

    for net in truth["networks"]:
        members = df[df["id"].isin(net["record_ids"])]
        key = {"card": "cards", "phone": "phones", "promo": "promo_codes", "domain": "domains",
               "short_link": "links"}[net["shared_entity_type"]]
        for ents in members["entities"].map(to_entities):
            assert any(sha256(v) == net["shared_entity_sha256"] for v in ents[key]), net["network_id"]


def test_legit_shops_trap(df, truth):
    traps = truth["legit_shared_domains"]
    assert traps
    for trap in traps:
        members = df[df["id"].isin(trap["record_ids"])]
        assert (members["label"] == "none").all()
        assert len(set(members["account_id"])) >= 3
        assert members["entities"].map(lambda e: trap["shared_domain"] in to_entities(e)["domains"]).all()


# ---------------------------------------------------------------------- gold
def test_gold_size_and_classes(gold):
    assert 100 <= len(gold) <= 150
    assert set(gold["label"]) == LABELS
    assert set(gold["lang"]) <= LANGS
    assert {"ru", "kk", "mixed"} <= set(gold["lang"])
    assert gold["handwritten"].all()


def test_gold_does_not_overlap_synthetic(df, gold):
    assert not set(gold["text"]) & set(df["text"])
    assert not set(gold["text"].map(dedup_key)) & set(df["text"].map(dedup_key))


def test_gold_has_no_forbidden_words(gold):
    assert not gold["text"].map(find_forbidden).map(bool).any()


# -------------------------------------------------------------------- report
def test_dataset_report():
    report = paths.REPORTS_DIR / "dataset_report.md"
    require(report)
    text = report.read_text(encoding="utf-8")
    for section in ("## Өлшемдер", "## Бөлінулер", "## Мысалдар", "## Желілер", "## Gold-жинақ"):
        assert section in text
    for label in LABELS:
        assert f"### {label}" in text


def test_generator_is_deterministic():
    from safenet.generator.build import build_dataset

    a, _ = build_dataset(seed=7, render_images=False)
    b, _ = build_dataset(seed=7, render_images=False)
    assert a["text"].tolist() == b["text"].tolist()
    assert Path(paths.DATASET_PATH).exists()
