"""Context of a checked post: account, chat / group and the link to the original post.

The account is read from the link when it is not given (``https://t.me/airolejon/15156`` -> ``@airolejon``,
platform Telegram), so posts of one channel meet on one node of the account graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from django.core.exceptions import ValidationError
from django.core.validators import URLValidator

MAX_NAME = 100
MAX_URL = 500

HOSTS = {
    "t.me": "telegram", "telegram.me": "telegram", "telegram.dog": "telegram",
    "instagram.com": "instagram", "tiktok.com": "tiktok",
    "vk.com": "vk", "m.vk.com": "vk",
    "youtube.com": "youtube", "m.youtube.com": "youtube", "youtu.be": "youtube",
    "wa.me": "whatsapp", "chat.whatsapp.com": "whatsapp",
}
# first path segments that are pages of the service, not account names
SERVICE_PATHS = {
    "telegram": {"s", "c", "joinchat", "addstickers", "addemoji", "share", "proxy", "iv"},
    "instagram": {"p", "reel", "reels", "stories", "explore", "tv", "accounts"},
    "vk": {"wall", "feed", "video", "photo", "im", "away.php", "search"},
}
HANDLE_PLATFORMS = {"telegram", "instagram", "tiktok", "youtube"}
HANDLE = re.compile(r"[A-Za-z0-9_.]{3,64}")
LINK = re.compile(r"(https?://)?[\w-]+(\.[\w-]+)+/", re.I)
_validate_url = URLValidator(schemes=["http", "https"])


class ContextError(ValueError):
    """The link to the original post is not a valid http(s) URL."""


@dataclass
class PostContext:
    account_id: str = ""
    group_id: str = ""
    source_url: str = ""
    platform: str = ""


def account_from_url(url: str) -> tuple[str, str]:
    """``(platform, account)`` of a post link; empty strings when they cannot be told."""
    parts = urlsplit(url if re.match(r"https?://", url, re.I) else f"https://{url}")
    platform = HOSTS.get((parts.hostname or "").lower().removeprefix("www."), "")
    segments = [s for s in parts.path.split("/") if s]
    if platform == "telegram" and segments[:1] == ["s"]:
        segments = segments[1:]
    if not platform or not segments:
        return platform, ""
    first = segments[0]
    if platform == "vk":
        wall = re.fullmatch(r"wall(-?\d+)_\d+", first)
        if wall:
            return platform, f"vk:{wall.group(1)}"
        return platform, "" if first in SERVICE_PATHS["vk"] else f"vk:{first.lower()}"
    if platform in ("tiktok", "youtube"):
        return platform, first.lower() if first.startswith("@") else ""
    if platform in SERVICE_PATHS and (first in SERVICE_PATHS[platform] or first.startswith("+")):
        return platform, ""
    if platform in ("telegram", "instagram") and HANDLE.fullmatch(first):
        return platform, f"@{first.lower()}"
    return platform, ""  # WhatsApp links carry a phone number or an invite, not a name


def normalize_name(value: str, platform: str) -> str:
    """One spelling per account or chat: ``AiroLejon`` / ``@airolejon`` / ``t.me/airolejon`` -> ``@airolejon``."""
    value = (value or "").strip()
    if not value:
        return ""
    if LINK.match(value):
        link_platform, account = account_from_url(value)
        if account:
            return account
        platform = platform or link_platform
    if value.startswith("@"):
        return value.lower()[:MAX_NAME]
    if platform == "vk" and HANDLE.fullmatch(value):
        return f"vk:{value.lower()}"
    if platform in HANDLE_PLATFORMS and HANDLE.fullmatch(value):
        return f"@{value.lower()}"
    return value[:MAX_NAME]


def build_context(*, account: str = "", group: str = "", source_url: str = "", platform: str = "") -> PostContext:
    """Validate and normalise what the analyst (or an integration) said about the post."""
    source_url = (source_url or "").strip()
    url_platform, url_account = "", ""
    if source_url:
        if not re.match(r"https?://", source_url, re.I):
            source_url = f"https://{source_url}"
        try:
            if len(source_url) > MAX_URL:
                raise ValidationError("too long")
            _validate_url(source_url)
        except ValidationError as exc:
            raise ContextError(source_url) from exc
        url_platform, url_account = account_from_url(source_url)
    platform = platform or url_platform
    return PostContext(
        account_id=normalize_name(account, platform) or url_account,
        group_id=normalize_name(group, platform),
        source_url=source_url,
        platform=platform,
    )
