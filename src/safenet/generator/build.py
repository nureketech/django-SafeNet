"""Build the synthetic dataset, screenshots and the ground-truth networks."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from safenet import paths
from safenet.common import dedup_key, mask_card, mask_phone, sha256
from safenet.generator import fillers as F
from safenet.generator.images import render_screenshot
from safenet.generator.render import render_record
from safenet.generator.split import assign_splits
from safenet.generator.templates import Template, build_catalog, template_slots

SEED = 42

# (label, is_hard_negative, source_type, records)
QUOTAS: list[tuple[str, bool, str, int]] = [
    ("lottery", False, "post", 260), ("lottery", False, "chat_message", 115),
    ("lottery", False, "stream_transcript", 40), ("lottery", False, "account_bio", 35),
    ("lottery", False, "screenshot", 30),
    ("casino", False, "post", 205), ("casino", False, "chat_message", 90),
    ("casino", False, "stream_transcript", 100), ("casino", False, "account_bio", 55),
    ("casino", False, "screenshot", 30),
    ("pyramid", False, "post", 215), ("pyramid", False, "chat_message", 160),
    ("pyramid", False, "stream_transcript", 25), ("pyramid", False, "account_bio", 50),
    ("pyramid", False, "screenshot", 30),
    ("none", True, "post", 330), ("none", True, "chat_message", 115),
    ("none", True, "stream_transcript", 25), ("none", True, "account_bio", 30),
    ("none", True, "screenshot", 40),
    ("none", False, "post", 40), ("none", False, "chat_message", 315),
    ("none", False, "stream_transcript", 20), ("none", False, "account_bio", 45),
]
RU_BASE_SHARE = 0.51
# Language mix per base language. Schemes are transliterated more often than normal
# content, so that ~35% of schemes and ~10% of negatives are obfuscated (TZ 1.3)
# while translit stays >= 10% overall.
LANG_SHARES = {
    "scheme": {"ru": {"ru": 43, "translit": 8}, "kk": {"kk": 27, "mixed": 14, "translit": 8}},
    "none": {"ru": {"ru": 47, "translit": 4}, "kk": {"kk": 29, "mixed": 16, "translit": 4}},
}
OBFUSCATION_PROB = {"scheme": 0.25, "none": 0.03}

# (track, shared entity slot, number of member accounts)
NETWORK_SPECS: list[tuple[str, str, int]] = [
    ("lottery", "card", 5), ("lottery", "phone", 4), ("lottery", "card", 7),
    ("casino", "promo", 8), ("casino", "domain", 6), ("casino", "short_link", 5),
    ("casino", "ref_link", 10),
    ("pyramid", "card", 6), ("pyramid", "phone", 9), ("pyramid", "card", 3),
]
SHOP_TRAPS = [4, 3]  # legit shops sharing one website domain (false-positive trap for the graph)
CHAT_GROUPS = 40
NORMAL_CHAT_USERS = 250

PLATFORMS = {
    "post": ["telegram", "instagram", "vk", "tiktok", "telegram", "instagram"],
    "chat_message": ["telegram", "whatsapp", "vk", "whatsapp"],
    "stream_transcript": ["youtube", "tiktok", "instagram", "youtube"],
    "account_bio": ["instagram", "tiktok", "telegram", "instagram"],
    "screenshot": ["telegram", "whatsapp", "instagram"],
}
PYRAMID_PLATFORMS = ["whatsapp", "telegram", "telegram", "instagram"]


@dataclass
class Spec:
    """Generation plan of one record."""

    order: int
    template: Template
    lang: str
    account_id: str | None = None
    group_id: str | None = None
    forced: dict[str, str] = field(default_factory=dict)
    network_id: str | None = None


def _apportion(n: int, weights: dict[str, int]) -> list[str]:
    """Split ``n`` items across keys proportionally (largest remainder)."""
    total = sum(weights.values())
    raw = {k: n * w / total for k, w in weights.items()}
    counts = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: n - sum(counts.values())]:
        counts[k] += 1
    return [k for k, c in counts.items() for _ in range(c)]


def _account(rng: random.Random) -> str:
    return f"acc_{rng.getrandbits(40):010x}"


def plan_records(catalog: list[Template], rng: random.Random) -> list[Spec]:
    specs: list[Spec] = []
    for label, hard_neg, source, count in QUOTAS:
        pool = [t for t in catalog if t.label == label and t.is_hard_negative == hard_neg and t.source_type == source]
        n_ru = round(count * RU_BASE_SHARE)
        for base, n in (("ru", n_ru), ("kk", count - n_ru)):
            templates = [t for t in pool if t.base_lang == base]
            rng.shuffle(templates)
            shares = LANG_SHARES["none" if label == "none" else "scheme"][base]
            langs = [base] * n if source == "screenshot" else _apportion(n, shares)
            rng.shuffle(langs)
            for i, lang in enumerate(langs):
                specs.append(Spec(len(specs), templates[i % len(templates)], lang))
    return specs


def _candidates(specs: list[Spec], used: set[int], predicate, slot: str) -> list[Spec]:
    out = []
    for s in specs:
        t = s.template
        if s.order in used or not predicate(t) or s.lang not in (t.base_lang, "translit"):
            continue
        if slot in template_slots(t, t.base_lang):
            out.append(s)
    return out


def assign_networks(specs: list[Spec], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Attach ground-truth scheme networks and legit shop traps to planned records."""
    used: set[int] = set()
    networks: list[dict] = []
    for ni, (track, slot, n_members) in enumerate(NETWORK_SPECS, 1):
        lookup_slot = slot
        cands = _candidates(specs, used, lambda t, tr=track: t.label == tr, lookup_slot)
        rng.shuffle(cands)
        if slot == "card":
            value, forced = F.card(rng), None
        elif slot == "phone":
            value, forced = F.phone(rng), None
        elif slot == "promo":
            value, forced = F.promo(rng), None
        elif slot == "domain":
            value, forced = F.casino_domain(rng, rng.choice(F.CASINOS)), None
        elif slot == "short_link":
            value, forced = F.short_link(rng), None
        elif slot == "ref_link":
            value = F.casino_domain(rng, rng.choice(F.CASINOS))
            forced = {"ref_domain": value}
        else:
            raise ValueError(slot)
        forced = forced or {slot: value}
        net_id = f"net{ni:02d}"
        members: dict[str, list[int]] = {}
        it = iter(cands)
        accounts = [_account(rng) for _ in range(n_members)]
        # Every member first gets one record, then up to two more while candidates last.
        plan = accounts + [acc for acc in accounts for _ in range(rng.randint(0, 2))]
        for acc in plan:
            spec = next(it, None)
            if spec is None:
                break
            spec.account_id, spec.forced, spec.network_id = acc, dict(forced), net_id
            if spec.template.source_type in ("chat_message", "screenshot"):
                spec.group_id = f"chat_{net_id}"
            used.add(spec.order)
            members.setdefault(acc, []).append(spec.order)
        if len(members) < 3:
            raise RuntimeError(f"network {net_id}: not enough candidate records ({len(members)} members)")
        shown = mask_card(value) if slot == "card" else mask_phone(value) if slot == "phone" else value
        networks.append({
            "network_id": net_id,
            "track": track,
            "shared_entity_type": "domain" if slot == "ref_link" else slot,
            "shared_entity_masked": shown,
            "shared_entity_sha256": sha256(value),
            "accounts": list(members),
            "_orders": [o for orders in members.values() for o in orders],
        })

    traps: list[dict] = []
    for ti, n_shops in enumerate(SHOP_TRAPS, 1):
        cands = _candidates(specs, used, lambda t: t.subtype in ("commerce", "hn_bio"), "shop_domain")
        rng.shuffle(cands)
        dom = F.shop_domain(rng)
        members = {}
        it = iter(cands)
        shops = [_account(rng) for _ in range(n_shops)]
        for acc in shops + [acc for acc in shops if rng.random() < 0.5]:
            spec = next(it, None)
            if spec is None:
                break
            spec.account_id, spec.forced = acc, {"shop_domain": dom}
            used.add(spec.order)
            members.setdefault(acc, []).append(spec.order)
        traps.append({"trap_id": f"shop{ti}", "shared_domain": dom, "accounts": list(members),
                      "_orders": [o for orders in members.values() for o in orders]})
    return networks, traps


def _platform(template: Template, rng: random.Random) -> str:
    if template.label == "pyramid" and template.source_type in ("post", "chat_message"):
        return rng.choice(PYRAMID_PLATFORMS)
    return rng.choice(PLATFORMS[template.source_type])


def _group(spec: Spec, rng: random.Random) -> str:
    acc = spec.account_id[4:]
    source = spec.template.source_type
    if source == "post":
        return f"ch_{acc}"
    if source == "account_bio":
        return f"profile_{acc}"
    if source == "stream_transcript":
        return f"stream_{acc}"
    return f"chat_{rng.randrange(CHAT_GROUPS):03d}"


def build_dataset(seed: int = SEED, render_images: bool = True) -> tuple[pd.DataFrame, dict]:
    """Generate all records; returns the dataframe and the networks ground truth."""
    rng = random.Random(seed)
    catalog = build_catalog(seed)
    specs = plan_records(catalog, rng)
    networks, traps = assign_networks(specs, rng)

    normal_users = [_account(rng) for _ in range(NORMAL_CHAT_USERS)]
    for spec in specs:
        if spec.account_id is None:
            spec.account_id = rng.choice(normal_users) if spec.template.subtype == "normal_chat" else _account(rng)
        if spec.group_id is None:
            spec.group_id = _group(spec, rng)

    order = list(range(len(specs)))
    rng.shuffle(order)
    record_id = {spec_order: f"syn-{i + 1:05d}" for i, spec_order in enumerate(order)}

    seen: set[str] = set()
    rows: list[dict] = []
    start = datetime(2025, 9, 1)
    for spec in specs:
        t = spec.template
        prob = OBFUSCATION_PROB["none" if t.label == "none" else "scheme"]
        rec_rng = random.Random(seed * 1_000_003 + spec.order)
        for attempt in range(40):
            rendered = render_record(
                t, spec.lang, rec_rng, spec.forced, prob,
                allow_link_hiding=spec.network_id is None, variation=0 if attempt < 6 else 1 + attempt // 20,
            )
            key = dedup_key(rendered.text)
            if key not in seen:
                break
        seen.add(key)
        rid = record_id[spec.order]
        rows.append({
            "id": rid,
            "text": rendered.text,
            "source_type": t.source_type,
            "platform": _platform(t, rec_rng),
            "lang": spec.lang,
            "label": t.label,
            "is_hard_negative": t.is_hard_negative,
            "risk_gold": t.risk_gold,
            "gold_signals": rendered.gold_signals,
            "obfuscations": rendered.obfuscations,
            "template_id": t.template_id,
            "template_fold": t.fold,
            "subtype": t.subtype,
            "account_id": spec.account_id,
            "group_id": spec.group_id,
            "entities": rendered.entities,
            "image_path": f"data/images/{rid}.png" if t.source_type == "screenshot" else None,
            "created_at": start + timedelta(seconds=rec_rng.randrange(400 * 24 * 3600)),
            "synthetic": True,
            "network_id": spec.network_id,
            "_order": spec.order,
        })

    image_meta: Counter = Counter()
    if render_images:
        paths.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        for old in paths.IMAGES_DIR.glob("syn-*.png"):
            old.unlink()
        for row in rows:
            if row["image_path"]:
                meta = render_screenshot(row["text"], paths.ROOT / row["image_path"], random.Random(row["id"]))
                image_meta[meta["theme"]] += 1
                image_meta["noisy"] += int(meta["noise"] or bool(meta["angle"]) or meta["blur"])

    by_order = {row["_order"]: row for row in rows}
    for net in networks:
        net["record_ids"] = sorted(by_order[o]["id"] for o in net.pop("_orders"))
    for trap in traps:
        trap["record_ids"] = sorted(by_order[o]["id"] for o in trap.pop("_orders"))

    df = pd.DataFrame(rows).drop(columns=["_order"]).sort_values("id").reset_index(drop=True)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = assign_splits(df)
    truth = {"seed": seed, "networks": networks, "legit_shared_domains": traps, "images": dict(image_meta)}
    return df, truth


def save_dataset(df: pd.DataFrame, truth: dict) -> None:
    paths.SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(paths.DATASET_PATH, index=False)
    for split in ("train", "val", "test"):
        df[df["split"] == split].to_parquet(paths.split_path(split), index=False)
    paths.NETWORKS_PATH.write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
