"""Entities of a text as graph nodes.

Card and phone numbers never enter the graph in plain form: the node id is a
SHA-256 hash of the canonical digits and the label is a mask
(``4400 43** **** 1234``, ``+7 701 *** ** 78``). Popular legitimate domains are
not nodes at all (they would connect unrelated accounts).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from safenet.common import mask_card, mask_phone, sha256
from safenet.preprocess import NormalizedText
from safenet.preprocess.tables import SHORTENERS

# Domains that many unrelated people mention: never a sign of a shared operator (neither the domain nor
# links on it become nodes).
LEGIT_DOMAINS = {
    "kaspi.kz", "random.org", "instagram.com", "t.me", "telegram.org", "telegram.me", "youtube.com", "youtu.be",
    "tiktok.com", "vk.com", "vk.ru", "m.vk.com", "ok.ru", "facebook.com", "x.com", "twitter.com", "wa.me",
    "whatsapp.com", "google.com", "play.google.com", "apps.apple.com", "gov.kz", "egov.kz", "afm.gov.kz", "olx.kz",
    "halykbank.kz",
}
# Link-in-bio services: the site itself links nothing, but one page on it (one profile) does.
PROFILE_HOSTS = {"linktr.ee", "taplink.cc", "taplink.ws"}
ENTITY_TYPES = ("card", "phone", "domain", "promo", "link")
# Shared entities that tie accounts to the same payment or promotion channel.
BONUS_ENTITY_TYPES = {"card", "phone", "promo", "link", "domain"}


@dataclass(frozen=True)
class EntityNode:
    node_id: str
    kind: str
    label: str


# Masked numbers with fewer known digits (e.g. «**** **** **** 1234», «+7 701 *** ** 78») are too
# unspecific to link accounts.
MIN_KNOWN_DIGITS = 6


def _known_digits(key: str) -> int:
    return sum(ch.isdigit() for ch in key)


def _card_key(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit() or ch == "*")


def _phone_key(value: str) -> str:
    raw = "".join(ch for ch in value if ch.isdigit() or ch == "*")
    return raw[-10:]


def _link_key(value: str) -> str:
    value = re.sub(r"^https?://", "", value.strip().lower())
    return value.rstrip("/.,")


def graph_entities(nt: NormalizedText) -> list[EntityNode]:
    """Canonical entity nodes found in a normalised text (deduplicated)."""
    out: dict[str, EntityNode] = {}

    def add(kind: str, key: str, label: str, hashed: bool = False) -> None:
        if not key:
            return
        node_id = f"{kind}:{sha256(key)[:24]}" if hashed else f"{kind}:{key}"
        out.setdefault(node_id, EntityNode(node_id, kind, label))

    for ent in nt.entities:
        if ent.kind == "card":
            key = _card_key(ent.text)
            if _known_digits(key) >= MIN_KNOWN_DIGITS:
                add("card", key, ent.text if "*" in ent.text else mask_card(key), hashed=True)
        elif ent.kind == "phone":
            key = _phone_key(ent.text)
            if _known_digits(key) >= MIN_KNOWN_DIGITS:
                add("phone", key, ent.text if "*" in ent.text else mask_phone(key), hashed=True)
        elif ent.kind == "domain":
            host = ent.meta.get("host", ent.text).lower().removeprefix("www.")
            if host not in LEGIT_DOMAINS and host not in SHORTENERS and host not in PROFILE_HOSTS:
                add("domain", host, host)
        elif ent.kind == "promo_code":
            add("promo", ent.text.upper(), ent.text.upper())
        elif ent.kind in ("url", "shortener"):
            key = _link_key(ent.text)
            host = key.split("/")[0]
            if host not in LEGIT_DOMAINS and "/" in key:
                add("link", key, key)
    return list(out.values())
