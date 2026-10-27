"""Public Telegram posts and channels, read from their web preview on ``t.me``.

The analyst pastes a link and the platform takes the post's text, the links hidden under its words and
buttons, the channel and the publication time. Hidden links matter most: they are what ties channels to one
operator, and copying a post by hand loses them.

Only ``https://t.me/...`` is requested, and the address is rebuilt from the parsed channel name and post
number, never taken from the input as it is. ``SAFENET_FETCH_ENABLED=0`` turns this off (closed networks).
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from django.conf import settings

LINK_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:t\.me|telegram\.me|telegram\.dog)/(?:s/)?"
    r"(?P<channel>[A-Za-z][A-Za-z0-9_]{3,31})(?:/(?P<post>\d{1,9}))?/?(?:[?#].*)?$",
    re.I,
)
SERVICE_NAMES = {"joinchat", "addstickers", "addemoji", "share", "proxy", "socks", "login", "setlanguage", "iv"}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) SafeNetKZ/1.0"
MAX_LINKS = 10
LINK_MARK = "🔗"

_BLOCK = re.compile(r'data-post="(?P<post>[^"]+)".*?(?=data-post="|\Z)', re.S)
_TEXT = re.compile(r'<div class="tgme_widget_message_text[^"]*js-message_text[^"]*"[^>]*>(.*?)</div>', re.S)
_HREF = re.compile(r'<a[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<label>.*?)</a>', re.S)
_BUTTON = re.compile(r'class="tgme_widget_message_inline_button[^"]*"[^>]*href="(?P<href>[^"]+)"')
_TIME = re.compile(r'<time[^>]*datetime="(?P<dt>[^"]+)"')
_TITLE = re.compile(r'<div class="tgme_channel_info_header_title"[^>]*>(.*?)</div>', re.S)
_DESCRIPTION = re.compile(r'<div class="tgme_channel_info_description"[^>]*>(.*?)</div>', re.S)


class FetchError(Exception):
    """The post or channel could not be read; ``code`` names the interface text that explains why."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class TgPost:
    channel: str
    post_id: int
    visible_text: str
    links: list[str] = field(default_factory=list)
    published_at: datetime | None = None
    has_photo: bool = False

    @property
    def account(self) -> str:
        return f"@{self.channel}"

    @property
    def url(self) -> str:
        return f"https://t.me/{self.channel}/{self.post_id}"

    @property
    def case_id(self) -> str:
        return f"tg-{self.channel}-{self.post_id}"[:64]

    @property
    def text(self) -> str:
        """What is analysed: the visible text and, below it, the links hidden under words and buttons."""
        return with_links(self.visible_text, self.links)


@dataclass
class TgChannel:
    channel: str
    title: str = ""
    description: str = ""
    description_links: list[str] = field(default_factory=list)
    posts: list[TgPost] = field(default_factory=list)

    @property
    def account(self) -> str:
        return f"@{self.channel}"

    @property
    def url(self) -> str:
        return f"https://t.me/{self.channel}"

    @property
    def about_text(self) -> str:
        return with_links("\n".join(p for p in (self.title, self.description) if p), self.description_links)


def with_links(text: str, links: list[str]) -> str:
    return "\n\n".join(p for p in (text.strip(), "\n".join(f"{LINK_MARK} {u}" for u in links)) if p)


def parse_link(url: str) -> tuple[str, int | None]:
    """``(channel, post number or None)`` of a t.me link; ``FetchError("fetch_bad_link")`` otherwise."""
    m = LINK_RE.match((url or "").strip())
    if not m or m.group("channel").lower() in SERVICE_NAMES:
        raise FetchError("fetch_bad_link")
    return m.group("channel").lower(), int(m.group("post")) if m.group("post") else None


def is_telegram_link(url: str) -> bool:
    try:
        parse_link(url)
    except FetchError:
        return False
    return True


# ------------------------------------------------------------------ parsing
def _clean(fragment: str) -> str:
    fragment = re.sub(r"<br\s*/?>", "\n", fragment)
    fragment = re.sub(r"<[^>]+>", "", fragment)
    return html.unescape(fragment).strip()


def _tidy_link(href: str) -> str | None:
    """Absolute http(s) link without tracking (utm_*) parameters; Telegram's own links are dropped."""
    href = html.unescape(html.unescape(href)).strip()
    parts = urlsplit(href)
    host = (parts.hostname or "").lower().removeprefix("www.")
    if parts.scheme not in ("http", "https") or not host or host in ("t.me", "telegram.me", "telegram.org"):
        return None
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _hidden_links(fragment: str, visible: str, extra: list[str] = ()) -> list[str]:
    """Links whose address is not written in the visible text (under words, behind buttons)."""
    out: list[str] = []
    for href in [m.group("href") for m in _HREF.finditer(fragment)] + list(extra):
        link = _tidy_link(href)
        if link and link not in out and urlsplit(link).netloc.lower() not in visible.lower():
            out.append(link)
    return out[:MAX_LINKS]


def _post(channel: str, post_id: int, block: str) -> TgPost | None:
    text_html = _TEXT.search(block)
    visible = _clean(text_html.group(1)) if text_html else ""
    buttons = [m.group("href") for m in _BUTTON.finditer(block)]
    links = _hidden_links(text_html.group(1) if text_html else "", visible, buttons)
    if not visible and not links:
        return None
    when = _TIME.search(block)
    published = None
    if when:
        try:
            published = datetime.fromisoformat(when.group("dt"))
        except ValueError:
            published = None
    return TgPost(channel, post_id, visible, links, published, "tgme_widget_message_photo" in block)


def parse_posts(doc: str, channel: str) -> list[TgPost]:
    posts = []
    for m in _BLOCK.finditer(doc):
        name, _, number = m.group("post").partition("/")
        if name.lower() != channel or not number.isdigit():
            continue
        post = _post(channel, int(number), m.group(0))
        if post is not None:
            posts.append(post)
    return posts


def parse_channel(doc: str, channel: str) -> TgChannel:
    if "tgme_channel_info" not in doc:
        raise FetchError("fetch_not_found")  # a user, a private group or no such channel
    title = _TITLE.search(doc)
    description = _DESCRIPTION.search(doc)
    desc_html = description.group(1) if description else ""
    desc_text = _clean(desc_html)
    return TgChannel(channel=channel, title=_clean(title.group(1)) if title else "", description=desc_text,
                     description_links=_hidden_links(desc_html, desc_text), posts=parse_posts(doc, channel))


# ------------------------------------------------------------------ network
def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "ru,kk,en"})
    try:
        with urllib.request.urlopen(request, timeout=settings.SAFENET_FETCH_TIMEOUT) as response:
            return response.read(4_000_000).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise FetchError("fetch_not_found" if exc.code == 404 else "fetch_failed") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FetchError("fetch_failed") from exc


def _enabled() -> None:
    if not settings.SAFENET_FETCH_ENABLED:
        raise FetchError("fetch_disabled")


def fetch_post(channel: str, post_id: int) -> TgPost:
    _enabled()
    doc = _get(f"https://t.me/{channel}/{post_id}?embed=1&mode=tme")
    if "tgme_widget_message_error" in doc:
        raise FetchError("fetch_not_found")
    posts = [p for p in parse_posts(doc, channel) if p.post_id == post_id]
    if not posts:
        raise FetchError("fetch_empty")  # a photo or video without text
    return posts[0]


def fetch_channel(channel: str) -> TgChannel:
    """The channel's title, description and its latest posts (the ~20 that t.me shows)."""
    _enabled()
    return parse_channel(_get(f"https://t.me/s/{channel}"), channel)
