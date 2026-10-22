"""Template catalog of the synthetic generator.

A *template* is a fixed structure: an ordered list of blocks, one concrete phrase
variant per block, a layout (line list, paragraph, chat, stream, bio, ...) and
decoration choices. Records generated from one template differ only in slot
fillers (amounts, prizes, entities, nicknames), language transform and
obfuscation.

Leakage control: every phrase variant belongs to a *phrase family* (its index
modulo ``N_FOLDS``). A template draws all of its variants from one family and the
family decides the split (``FOLD_SPLIT``), so train/val/test share neither
templates nor phrasing.
"""

from __future__ import annotations

import dataclasses
import functools
import random
import re
from dataclasses import dataclass, field

from safenet.generator import pools_kk, pools_ru

N_FOLDS = 5
FOLD_SPLIT = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}

POOLS = {"ru": pools_ru.POOLS, "kk": pools_kk.POOLS}

# Default gold signals of each block; variant tuples can add more.
BLOCK_SIGNALS: dict[str, tuple[str, ...]] = {
    "transfer": ("C_PAY_TRANSFER",),
    "card": ("C_PAY_TRANSFER", "C_PERSONAL_CARD"),
    "urgency": ("C_URGENCY",),
    "dm": ("C_DM_REDIRECT",),
    "referral": ("C_REFERRAL",),
    "short": ("C_SHORTLINK",),
    "l_paid": ("L_PAID_ENTRY",),
    "l_draw": ("L_RANDOM_DRAW",),
    "l_profit": ("L_ORGANIZER_PROFIT",),
    "l_series": ("L_SERIES",),
    "k_win": ("K_WIN_BRAG",),
    "k_gamble": ("K_GAMBLING_TERMS",),
    "k_promo": ("K_PROMO_BONUS",),
    "k_ref": ("K_REF_TRACKING",),
    "k_mirror": ("K_MIRROR",),
    "k_cta_stream": ("K_STREAM_CTA",),
    "k_proxy": ("C_SHORTLINK",),
    "k_referral": ("C_REFERRAL",),
    "p_return": ("P_UNREAL_RETURN",),
    "p_guarantee": ("P_GUARANTEE",),
    "p_tiers": ("P_REFERRAL_TIERS", "C_REFERRAL"),
    "p_proof": ("P_PAYOUT_PROOF",),
    "p_org": ("P_ORG_STRUCTURE",),
    "p_urgency": ("C_URGENCY",),
    "n_free": ("N_FREE_ENTRY",),
    "g_draw": ("L_RANDOM_DRAW",),
    "n_pay": ("C_PAY_TRANSFER", "N_COMMERCE"),
    "n_delivery": ("N_COMMERCE",),
    "n_bank": ("N_BANK_RATE",),
    "w_open": ("N_WARNING_CONTEXT",),
    "w_advice": ("N_WARNING_CONTEXT",),
    "e_open": ("N_SKILL_EVENT",),
    "e_rules": ("N_SKILL_EVENT",),
    "f_open": ("N_WARNING_CONTEXT",),
    "s_official": ("N_OFFICIAL_LOTTERY",),
    "s_draw": ("L_RANDOM_DRAW",),
    "sp_purchase": ("N_COMMERCE",),
    "sp_free": ("N_FREE_ENTRY",),
}

# Pool lookup chain for each subtype (first pool that has the block wins).
POOL_CHAIN: dict[str, tuple[str, ...]] = {
    "lottery": ("lottery", "common"),
    "casino": ("casino", "common"),
    "pyramid": ("pyramid", "common"),
    "free_giveaway": ("free_giveaway", "common"),
    "commerce": ("commerce", "common"),
    "bank": ("bank",),
    "warning": ("warning", "common"),
    "esports": ("esports", "common"),
    "literacy": ("literacy",),
    "state_promo": ("state_promo", "free_giveaway"),
    "hn_bio": ("hn_bio", "commerce", "common"),
    "normal_chat": ("normal_chat", "common"),
    "normal_post": ("normal_post",),
    "normal_stream": ("normal_stream", "common"),
    "normal_bio": ("normal_bio",),
}

HARD_NEGATIVE_SUBTYPES = {
    "free_giveaway", "commerce", "bank", "warning", "esports", "literacy", "state_promo", "hn_bio",
    "hn_screen",
}
SCHEME_LABELS = ("lottery", "casino", "pyramid")

# STRUCTURES[subtype][source_type][strength] -> list of block sequences.
STRUCTURES: dict[str, dict[str, dict[str, list[list[str]]]]] = {
    "lottery": {
        "post": {
            "high": [
                ["l_opener", "l_prize", "l_paid", "transfer", "l_draw", "urgency"],
                ["l_series", "l_prize", "l_paid", "l_profit", "l_draw", "dm"],
                ["l_opener", "l_paid", "card", "l_draw", "l_profit", "l_cta"],
                ["l_prize", "l_paid", "l_draw", "l_series", "transfer", "l_cta"],
                ["l_opener", "l_prize", "l_profit", "l_paid", "transfer", "l_draw", "referral"],
                ["l_series", "l_opener", "l_prize", "l_paid", "card", "urgency", "l_draw"],
            ],
            "medium": [
                ["l_opener", "l_prize", "l_vague", "dm"],
                ["l_opener", "l_prize", "l_draw", "l_vague"],
                ["l_paid", "urgency", "dm"],
                ["l_series", "l_prize", "l_paid", "transfer"],
            ],
        },
        "chat_message": {
            "high": [
                ["l_paid", "transfer", "l_draw"],
                ["l_prize", "l_paid", "l_draw", "dm"],
                ["l_opener", "l_paid", "card", "l_draw", "urgency"],
            ],
            "medium": [["l_prize", "l_vague", "dm"], ["l_paid", "l_profit", "dm"]],
        },
        "stream_transcript": {
            "high": [
                ["s_open", "l_stream", "l_paid", "l_draw", "s_fill", "transfer", "l_profit"],
                ["s_open", "l_prize", "s_fill", "l_stream", "l_draw", "l_paid"],
            ],
            "medium": [["s_open", "l_stream", "s_fill", "l_draw", "dm"]],
        },
        "account_bio": {
            "high": [["l_bio", "l_paid", "transfer"], ["l_bio", "l_series", "l_paid", "dm"]],
            "medium": [["l_bio", "dm"], ["l_bio", "l_series", "dm"]],
        },
    },
    "casino": {
        "post": {
            "high": [
                ["k_opener", "k_win", "k_gamble", "k_promo", "k_ref"],
                ["k_gamble", "k_promo", "short", "k_cta"],
                ["k_win", "k_gamble", "k_mirror", "k_promo"],
                ["k_opener", "k_gamble", "k_ref", "k_referral", "urgency"],
                ["k_win", "k_promo", "k_proxy"],
                ["k_opener", "k_win", "k_gamble", "k_cta_stream"],
            ],
            "medium": [
                ["k_opener", "k_win", "dm"],
                ["k_mirror", "short"],
                ["k_win", "short", "k_cta"],
                ["k_gamble", "k_cta"],
            ],
        },
        "chat_message": {
            "high": [
                ["k_win", "k_gamble", "k_ref"],
                ["k_gamble", "k_promo", "dm"],
                ["k_mirror", "k_gamble", "short"],
            ],
            "medium": [["k_win", "dm"], ["k_mirror", "dm"]],
        },
        "stream_transcript": {
            "high": [
                ["s_open", "k_stream", "k_win", "k_cta_stream", "k_promo", "s_fill"],
                ["s_open", "k_stream", "k_gamble", "s_fill", "k_cta_stream"],
                ["s_open", "k_win", "k_stream", "k_promo", "k_cta_stream"],
            ],
            "medium": [["s_open", "k_stream", "k_win", "s_fill", "dm"]],
        },
        "account_bio": {
            "high": [["k_bio", "k_promo", "short"], ["k_bio", "k_gamble", "k_ref"], ["k_bio", "k_mirror", "short"]],
            "medium": [["k_bio", "short"], ["k_bio", "k_win", "dm"]],
        },
    },
    "pyramid": {
        "post": {
            "high": [
                ["p_opener", "p_return", "p_guarantee", "p_urgency", "dm"],
                ["p_opener", "p_return", "p_tiers", "p_proof", "p_org"],
                ["p_return", "p_guarantee", "transfer", "p_urgency"],
                ["p_opener", "p_vague", "p_return", "card", "p_tiers"],
                ["p_proof", "p_return", "p_org", "p_cta"],
                ["p_opener", "p_return", "p_guarantee", "p_tiers", "p_urgency", "dm"],
            ],
            "medium": [
                ["p_opener", "p_vague", "p_org", "dm"],
                ["p_opener", "p_guarantee", "p_vague", "p_cta"],
                ["p_tiers", "p_org", "dm"],
                ["p_warmup", "p_vague", "dm"],
            ],
        },
        "chat_message": {
            "high": [
                ["p_return", "p_guarantee", "dm"],
                ["p_opener", "p_return", "p_urgency", "transfer"],
                ["p_warmup", "p_return", "p_tiers"],
                ["p_return", "card", "p_urgency"],
            ],
            "medium": [["p_vague", "dm"], ["p_opener", "p_org", "dm"], ["p_warmup", "p_org"]],
        },
        "stream_transcript": {
            "high": [["s_open", "p_stream", "p_return", "s_fill", "p_tiers", "p_org"]],
            "medium": [["s_open", "p_stream", "p_vague", "s_fill", "p_org"]],
        },
        "account_bio": {
            "high": [["p_bio", "p_return", "dm"], ["p_bio", "p_return", "short"]],
            "medium": [["p_bio", "p_vague", "dm"], ["p_bio", "p_org", "short"]],
        },
    },
    "free_giveaway": {"post": {"low": [
        ["g_open", "g_prize", "n_free", "g_draw", "g_close"],
        ["g_prize", "n_free", "g_deadline"],
        ["g_open", "n_free", "g_draw"],
        ["g_open", "g_prize", "n_free", "g_deadline", "g_close"],
        ["g_open", "g_prize", "g_draw", "n_free"],
    ]}},
    "commerce": {
        "post": {"low": [
            ["c_item", "c_price", "n_pay", "n_delivery", "dm"],
            ["c_item", "c_price", "n_delivery", "n_pay"],
            ["c_open", "c_item", "n_pay", "c_shop", "dm"],
            ["c_open", "c_item", "c_price", "n_pay"],
        ]},
        "chat_message": {"low": [
            ["c_item", "c_price", "n_pay"],
            ["c_chat_q", "n_pay", "n_delivery"],
            ["c_chat_q", "c_price", "n_pay"],
        ]},
    },
    "bank": {"post": {"low": [
        ["b_open", "n_bank", "b_ins", "b_app"],
        ["n_bank", "b_min", "b_ins"],
        ["b_open", "n_bank", "b_cap", "b_close"],
        ["b_open", "n_bank", "b_min", "b_app", "b_close"],
    ]}},
    "warning": {
        "post": {"low": [
            ["w_open", "w_quote", "w_advice"],
            ["w_open", "w_quote", "w_advice", "w_src"],
            ["w_quote", "w_advice", "w_src"],
            ["w_open", "w_quote", "w_quote", "w_advice"],
        ]},
        "chat_message": {"low": [
            ["w_fwd", "w_open", "w_quote", "w_advice"],
            ["w_fwd", "w_quote", "w_advice"],
            ["w_open", "w_advice"],
        ]},
    },
    "esports": {
        "post": {"low": [
            ["e_open", "e_pool", "e_fee", "e_reg", "e_rules"],
            ["e_open", "e_pool", "e_reg"],
            ["e_open", "e_rules", "e_pool", "e_stream"],
            ["e_open", "e_fee", "e_rules", "e_reg"],
        ]},
        "stream_transcript": {"low": [
            ["s_open", "e_lines", "s_fill", "e_lines", "e_rules"],
            ["s_open", "e_lines", "e_lines", "s_fill"],
        ]},
    },
    "literacy": {"post": {"low": [
        ["f_open", "f_points", "f_advice"],
        ["f_open", "f_points", "f_points", "f_hotline"],
        ["f_open", "f_advice", "f_hotline"],
    ]}},
    "state_promo": {"post": {"low": [
        ["s_official", "s_draw", "s_buy"],
        ["sp_open", "sp_purchase", "sp_free", "s_draw"],
        ["sp_purchase", "sp_free", "g_deadline"],
        ["s_official", "s_buy", "s_draw"],
    ]}},
    "hn_bio": {"account_bio": {"low": [
        ["hb_head", "n_pay", "n_delivery"],
        ["hb_head", "c_shop", "dm"],
        ["hb_head", "n_pay", "dm"],
    ]}},
    "normal_chat": {"chat_message": {"low": [
        ["nc_greet", "nc_body"],
        ["nc_body", "nc_close"],
        ["nc_body"],
        ["nc_greet", "nc_body", "nc_close"],
    ]}},
    "normal_post": {"post": {"low": [
        ["np_title", "np_body", "np_contact"],
        ["np_body", "np_contact"],
        ["np_title", "np_body"],
    ]}},
    "normal_stream": {"stream_transcript": {"low": [
        ["s_open", "ns_body", "s_fill", "ns_body"],
        ["s_open", "ns_body", "s_fill"],
    ]}},
    "normal_bio": {"account_bio": {"low": [
        ["nb_head", "nb_item", "nb_contact"],
        ["nb_head", "nb_item"],
    ]}},
}

LAYOUTS: dict[str, list[str]] = {
    "post": ["lines", "paragraph", "emoji_lines", "numbered", "caps_lines"],
    "chat_message": ["chat", "chat_lower"],
    "stream_transcript": ["stream"],
    "account_bio": ["bio_pipe", "bio_lines", "bio_dot"],
    "screenshot": ["screen"],
}

# Templates per fold for every (label, subtype, source_type); identical for ru and kk.
TEMPLATE_PLAN: list[tuple[str, str, str, int]] = [
    ("lottery", "lottery", "post", 3),
    ("lottery", "lottery", "chat_message", 2),
    ("lottery", "lottery", "stream_transcript", 1),
    ("lottery", "lottery", "account_bio", 1),
    ("lottery", "lottery", "screenshot", 1),
    ("casino", "casino", "post", 2),
    ("casino", "casino", "chat_message", 2),
    ("casino", "casino", "stream_transcript", 2),
    ("casino", "casino", "account_bio", 1),
    ("casino", "casino", "screenshot", 1),
    ("pyramid", "pyramid", "post", 3),
    ("pyramid", "pyramid", "chat_message", 2),
    ("pyramid", "pyramid", "stream_transcript", 1),
    ("pyramid", "pyramid", "account_bio", 1),
    ("pyramid", "pyramid", "screenshot", 1),
    ("none", "free_giveaway", "post", 1),
    ("none", "commerce", "post", 1),
    ("none", "commerce", "chat_message", 1),
    ("none", "bank", "post", 1),
    ("none", "warning", "post", 1),
    ("none", "warning", "chat_message", 1),
    ("none", "esports", "post", 1),
    ("none", "esports", "stream_transcript", 1),
    ("none", "literacy", "post", 1),
    ("none", "state_promo", "post", 1),
    ("none", "hn_bio", "account_bio", 1),
    ("none", "hn_screen", "screenshot", 2),
    ("none", "normal_chat", "chat_message", 3),
    ("none", "normal_post", "post", 1),
    ("none", "normal_stream", "stream_transcript", 1),
    ("none", "normal_bio", "account_bio", 1),
]

BULLETS = ["", "• ", "— ", "▪️ ", "✅ ", "👉 ", "🔹 ", "➡️ "]
EMOJI_SETS = [
    ["🔥", "💰", "✅", "👉"], ["🎁", "🎉", "⭐", "📌"], ["💸", "💎", "🚀", "✨"], ["📍", "☑️", "🔸", "💬"],
]
SLOT_RE = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class Template:
    """A fixed generation structure; see the module docstring."""

    template_id: str
    label: str
    subtype: str
    base_lang: str
    source_type: str
    layout: str
    fold: int
    strength: str
    is_hard_negative: bool
    blocks: tuple[tuple[str, int], ...] = ()
    screen_index: int | None = None
    deco: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def split(self) -> str:
        return FOLD_SPLIT[self.fold]

    @property
    def risk_gold(self) -> str:
        return "low" if self.label == "none" else self.strength


def find_pool(subtype: str, block: str, lang: str) -> list:
    """Return the variant list for ``block`` in the pool chain of ``subtype``."""
    for pool_name in POOL_CHAIN[subtype]:
        pool = POOLS[lang].get(pool_name, {})
        if block in pool:
            return pool[block]
    raise KeyError(f"block {block!r} not found for subtype {subtype!r} ({lang})")


def fold_variant(pool: list, fold: int, rank: int):
    """Pick the ``rank``-th variant of phrase family ``fold`` (wrapping around)."""
    indices = [i for i in range(len(pool)) if i % N_FOLDS == fold] or [fold % len(pool)]
    return pool[indices[rank % len(indices)]]


def variant_text(variant) -> tuple[str, tuple[str, ...]]:
    if isinstance(variant, str):
        return variant, ()
    text, extra = variant
    return text, tuple(extra)


def resolve_block(template: Template, block: str, rank: int, lang: str) -> tuple[str, tuple[str, ...]]:
    """Text and gold signals of one block of ``template`` rendered in ``lang``."""
    text, extra = variant_text(fold_variant(find_pool(template.subtype, block, lang), template.fold, rank))
    return text, BLOCK_SIGNALS.get(block, ()) + extra


@functools.lru_cache(maxsize=None)
def template_slots(template: Template, lang: str) -> frozenset[str]:
    """Slot names used by the template in ``lang`` (screens are handled in ``screens``)."""
    if template.source_type == "screenshot":
        from safenet.generator.screens import screen_spec

        return frozenset(SLOT_RE.findall(screen_spec(template, lang)["text"]))
    slots: set[str] = set()
    for block, rank in template.blocks:
        slots |= set(SLOT_RE.findall(resolve_block(template, block, rank, lang)[0]))
    return frozenset(slots)


def build_catalog(seed: int = 42) -> list[Template]:
    """Create the full deterministic template catalog."""
    rng = random.Random(seed)
    catalog: list[Template] = []
    counter = 0
    for base_lang in ("ru", "kk"):
        for label, subtype, source_type, per_fold in TEMPLATE_PLAN:
            for fold in range(N_FOLDS):
                for j in range(per_fold):
                    counter += 1
                    hard_neg = subtype in HARD_NEGATIVE_SUBTYPES
                    tid = f"{label[:3]}-{subtype[:6]}-{source_type[:4]}-{base_lang}-f{fold}-{j}-{counter:04d}"
                    layout = rng.choice(LAYOUTS[source_type])
                    deco = {
                        "bullet": rng.choice(BULLETS),
                        "emoji": rng.choice(EMOJI_SETS),
                        "separator": rng.choice([". ", "! ", ". ", ", "]),
                        "end_emoji": rng.choice(["", "", "🔥", "🙏", "❤️", "👇", "😉"]),
                    }
                    if source_type == "screenshot":
                        from safenet.generator.screens import screen_spec

                        tpl = Template(
                            tid, label, subtype, base_lang, source_type, "screen", fold, "low",
                            hard_neg, screen_index=j, deco=deco,
                        )
                        catalog.append(dataclasses.replace(tpl, strength=screen_spec(tpl, base_lang)["strength"]))
                        continue
                    by_strength = STRUCTURES[subtype][source_type]
                    if label == "none":
                        strength = "low"
                    else:
                        strength = "medium" if rng.random() < 0.3 else "high"
                    structures = by_strength[strength]
                    structure = structures[(fold + j * 2 + rng.randrange(len(structures))) % len(structures)]
                    used: dict[str, int] = {}
                    blocks = []
                    for block in structure:
                        rank = used.get(block, rng.randrange(4))
                        if block in used:
                            rank += 1
                        used[block] = rank
                        blocks.append((block, rank))
                    catalog.append(Template(
                        tid, label, subtype, base_lang, source_type, layout, fold, strength, hard_neg,
                        blocks=tuple(blocks), deco=deco,
                    ))
    return catalog
