"""Text normalisation with a position map back to the original text.

Every normalised character remembers the ``[start, end)`` range of original
characters it came from, so evidence found in the normalised (or folded) text can
be highlighted in the original. Pipeline:

1. NFKC + lower case, dashes/quotes unified, invisible characters dropped;
2. spaced-out words joined («к а з и н о» -> «казино», «k-a-s-p-i» -> «kaspi»);
3. homoglyphs / leet inside Cyrillic words -> Cyrillic («к@зинo» -> «казино»);
4. Latin transliteration -> Cyrillic (Russian or Kazakh rules: q→қ, sh→ш, ä→ә ...);
5. language detection;
6. emoji -> word tokens (💰 -> деньги/ақша, 🎰 -> казино, ...);
7. whitespace collapse; folded copy (Kazakh letters -> Russian look-alikes, same length);
8. entity extraction (amounts, percents, multipliers, links, cards, phones, promo codes, Kaspi).

Entities found in the original text (links, cards, phones, promo codes) are
protected: later steps never transliterate or "fix" them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from safenet.preprocess import tables as T
from safenet.preprocess.entities import Entity, extract_normalized_entities, extract_raw_entities
from safenet.preprocess.lang import detect_language

PROTECTED_KINDS = {"url", "domain", "shortener", "card", "phone", "promo_code", "ref_param"}


@dataclass
class NormalizedText:
    """Result of :func:`normalize`."""

    original: str
    text: str
    folded: str
    starts: list[int]
    ends: list[int]
    lang: str
    translit: str | None
    entities: list[Entity] = field(default_factory=list)

    def to_original(self, start: int, end: int) -> tuple[int, int]:
        """Map a ``[start, end)`` span of the normalised text to the original text."""
        if not self.text:
            return 0, 0
        start = max(0, min(start, len(self.text) - 1))
        end = max(start + 1, min(end, len(self.text)))
        return self.starts[start], self.ends[end - 1]

    def original_fragment(self, start: int, end: int) -> str:
        s, e = self.to_original(start, end)
        return self.original[s:e]

    def by_kind(self, kind: str) -> list[Entity]:
        return [e for e in self.entities if e.kind == kind]

    def amounts(self) -> list[float]:
        return [e.value for e in self.by_kind("amount") if e.value is not None]

    def entity_dict(self) -> dict[str, list[str]]:
        """Entity strings grouped like the dataset ``entities`` field."""
        out: dict[str, list[str]] = {"cards": [], "phones": [], "domains": [], "promo_codes": [], "links": []}
        mapping = {"card": "cards", "phone": "phones", "domain": "domains", "promo_code": "promo_codes",
                   "url": "links", "shortener": "links"}
        for ent in self.entities:
            if ent.kind == "domain" and ent.meta.get("host") in T.SHORTENERS:
                continue
            key = mapping.get(ent.kind)
            if key and ent.text not in out[key]:
                out[key].append(ent.text)
        return out


class _Buffer:
    """Characters with the original ``[start, end)`` range of each one."""

    __slots__ = ("chars", "starts", "ends", "protected")

    def __init__(self) -> None:
        self.chars: list[str] = []
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.protected: list[bool] = []

    def add(self, ch: str, start: int, end: int, protected: bool = False) -> None:
        self.chars.append(ch)
        self.starts.append(start)
        self.ends.append(end)
        self.protected.append(protected)

    def copy_from(self, other: "_Buffer", i: int) -> None:
        self.add(other.chars[i], other.starts[i], other.ends[i], other.protected[i])

    def text(self) -> str:
        return "".join(self.chars)


# ---------------------------------------------------------------- step 1
def _initial_buffer(text: str, protected: set[int]) -> _Buffer:
    buf = _Buffer()
    for i, ch in enumerate(text):
        if ch in T.INVISIBLE or ch == "\r":
            continue
        if ch in T.DASHES:
            buf.add("-", i, i + 1, i in protected)
            continue
        if ch in T.QUOTES:
            buf.add('"', i, i + 1, i in protected)
            continue
        if unicodedata.combining(ch) and buf.chars:
            composed = unicodedata.normalize("NFC", buf.chars[-1] + ch)
            if len(composed) == 1:
                buf.chars[-1] = composed
                buf.ends[-1] = i + 1
            continue
        norm = ch.lower() if ch == "№" else unicodedata.normalize("NFKC", ch).lower()
        for c in norm:
            buf.add(c, i, i + 1, i in protected)
    return buf


# ---------------------------------------------------------------- step 2
_TOKEN = r"(?:sh|ch|zh|kh|ts|ya|yu|yo|gh|ia|iu|[^\W\d_]|[@$0])"
# Letters separated by one repeated separator; with «.»/«-»/«*»/«_» the last piece may carry a
# suffix that was written together («r.a.n.d.o.m.a.i.z.e.rmen»).
_SPACED_RE = re.compile(
    rf"(?<![\w@$]){_TOKEN}(?P<sep>[ .\-*_]){_TOKEN}(?:(?P=sep){_TOKEN})+"
    rf"(?:(?P=sep)[^\W\d_]{{2,}})?(?![\w@$])"
)


def _join_spaced(buf: _Buffer) -> _Buffer:
    text = buf.text()
    drop: set[int] = set()
    for m in _SPACED_RE.finditer(text):
        sep = m.group("sep")
        tokens = m.group(0).split(sep)
        if sum(1 for t in tokens for c in t if c.isalpha()) < 3:
            continue
        if len(tokens) < (4 if sep == " " else 3):
            continue
        if any(buf.protected[i] for i in range(m.start(), m.end())):
            continue
        seps = [i for i in range(m.start(), m.end()) if text[i] == sep]
        if sep == " " and len(tokens[-1]) > 2:
            seps = seps[:-1]  # a real word after spaced letters is not a suffix
        drop.update(seps)
    if not drop:
        return buf
    out = _Buffer()
    for i in range(len(buf.chars)):
        if i not in drop:
            out.copy_from(buf, i)
    return out


# ---------------------------------------------------------------- step 3/4
_WORD_RE = re.compile(r"[\w@$'’]+")


def _is_cyr(ch: str) -> bool:
    return ch in T.CYRILLIC


def _is_lat(ch: str) -> bool:
    return ("a" <= ch <= "z") or ch in T.KK_TRANSLIT_CHARS


def _translit_mode(buf: _Buffer, text: str) -> str | None:
    """``"ru"`` / ``"kk"`` when the text is mostly written in Latin letters, else ``None``."""
    lat = cyr = 0
    for ch, prot in zip(buf.chars, buf.protected):
        if prot:
            continue
        if _is_cyr(ch):
            cyr += 1
        elif _is_lat(ch):
            lat += 1
    if lat < 3 or lat <= cyr:
        return None
    words = [m.group(0) for m in _WORD_RE.finditer(text)]
    if any(ch in T.KK_TRANSLIT_CHARS for ch in text) or any(
        w.lstrip("@").startswith(T.KK_TRANSLIT_STEMS) for w in words
    ):
        return "kk"
    return "ru"


def _transliterate(word: str, mode: str) -> list[tuple[str, int, int]]:
    """Latin -> Cyrillic; returns ``(char, i0, i1)`` with source ranges inside ``word``."""
    multi = dict(T.TRANSLIT_MULTI)
    multi.update(T.TRANSLIT_KK_MULTI if mode == "kk" else T.TRANSLIT_RU_MULTI)
    single = dict(T.TRANSLIT_SINGLE)
    if mode == "kk":
        single.update(T.TRANSLIT_KK_OVERRIDES)
    out: list[tuple[str, int, int]] = []
    j = 0
    n = len(word)
    while j < n:
        ch = word[j]
        matched = False
        for size in (4, 3, 2):
            chunk = word[j:j + size]
            if len(chunk) == size and chunk in multi:
                out.append((multi[chunk], j, j + size))
                j += size
                matched = True
                break
        if matched:
            continue
        if ch == "y":
            prev = word[j - 1] if j else ""
            prev_vowel = prev in T.LATIN_VOWELS or prev in T.VOWELS_CYR
            rep = "й" if prev_vowel else "ы"
            if out and out[-1][0] == "й" and prev == "y":
                rep = "ы"
            out.append((rep, j, j + 1))
        elif ch in single:
            rep = single[ch]
            if rep:
                out.append((rep, j, j + 1))
        else:
            out.append((ch, j, j + 1))
        j += 1
    return out


def _fix_word(word: str, mode: str | None) -> list[tuple[str, int, int]] | None:
    """Return replacement characters for one word, or ``None`` to keep it unchanged."""
    n_cyr = sum(1 for c in word if _is_cyr(c))
    n_lat = sum(1 for c in word if _is_lat(c))
    if n_cyr:
        if not n_lat and not any(c in T.LEET_CYR for c in word):
            return None
        has_kk = any(c in T.KK_LETTERS for c in word)
        out = []
        changed = False
        for j, c in enumerate(word):
            rep = c
            if c in T.HOMOGLYPHS:
                rep = "і" if (c == "i" and has_kk) else T.HOMOGLYPHS[c]
            elif c == "i":
                rep = "і" if has_kk else "и"
            elif c in "@$":
                left = word[j - 1] if j else ""
                right = word[j + 1] if j + 1 < len(word) else ""
                if _is_cyr(left) or _is_cyr(right):
                    rep = T.LEET_CYR[c]
            elif c in T.LEET_CYR and c.isdigit() and n_cyr >= 2:
                left = word[j - 1] if j else ""
                right = word[j + 1] if j + 1 < len(word) else ""
                if (left.isalpha() or left in "@$") and not right.isdigit():
                    rep = T.LEET_CYR[c]
            elif _is_lat(c):
                tr = _transliterate(c, "ru")
                rep = tr[0][0] if tr else c
            if rep != c:
                changed = True
            out.append((rep, j, j + 1))
        return out if changed else None
    if not n_lat:
        return None
    if word.startswith("@") and mode is None:
        return None
    # Leet inside Latin words: «k@zin0» -> «kazino».
    chars = list(word)
    if n_lat >= 2:
        for j, c in enumerate(chars):
            if c in T.LEET_LAT and c != "4" and j and chars[j - 1].isalpha():
                right = chars[j + 1] if j + 1 < len(chars) else ""
                if not right.isdigit() and not (c.isdigit() and j >= 2 and chars[j - 2].isdigit()):
                    chars[j] = T.LEET_LAT[c]
    core = "".join(chars)
    if core in T.LATIN_WORDS:
        rep = T.LATIN_WORDS[core]
        return [(ch, 0, len(word)) for ch in rep]
    if mode is not None:
        return _transliterate(core, mode)
    candidate = _transliterate(core, "ru")
    if "".join(c for c, _, _ in candidate).startswith(T.KEYWORD_STEMS):
        return candidate
    if core != word:
        return [(c, j, j + 1) for j, c in enumerate(core)]
    return None


def _fix_words(buf: _Buffer, mode: str | None) -> _Buffer:
    text = buf.text()
    out = _Buffer()
    pos = 0
    for m in _WORD_RE.finditer(text):
        s, e = m.start(), m.end()
        for i in range(pos, s):
            out.copy_from(buf, i)
        pos = e
        if any(buf.protected[s:e]):
            for i in range(s, e):
                out.copy_from(buf, i)
            continue
        repl = _fix_word(m.group(0), mode)
        if repl is None:
            for i in range(s, e):
                out.copy_from(buf, i)
            continue
        for ch, j0, j1 in repl:
            out.add(ch, buf.starts[s + j0], buf.ends[s + j1 - 1])
    for i in range(pos, len(buf.chars)):
        out.copy_from(buf, i)
    return out


# ---------------------------------------------------------------- step 6/7
def _emoji_tokens(buf: _Buffer, lang: str, mode: str | None) -> _Buffer:
    if not any(c in T.EMOJI_TOKENS for c in buf.chars):
        return buf
    use_kk = lang == "kk" or (lang == "translit" and mode == "kk")
    out = _Buffer()
    i = 0
    n = len(buf.chars)
    while i < n:
        ch = buf.chars[i]
        if ch not in T.EMOJI_TOKENS:
            out.copy_from(buf, i)
            i += 1
            continue
        j = i
        while j + 1 < n and buf.chars[j + 1] == ch:
            j += 1
        ru, kk = T.EMOJI_TOKENS[ch]
        token = f"{ru} {kk}" if lang == "mixed" else (kk if use_kk else ru)
        start, end = buf.starts[i], buf.ends[j]
        for c in f" {token} ":
            out.add(c, start, end)
        i = j + 1
    return out


def _collapse_spaces(buf: _Buffer) -> _Buffer:
    out = _Buffer()
    for i, ch in enumerate(buf.chars):
        if ch in " \t \n":
            is_newline = ch == "\n"
            if not out.chars:
                continue
            last = out.chars[-1]
            if last in " \n":
                if is_newline and last == " ":
                    out.chars[-1] = "\n"
                continue
            out.add("\n" if is_newline else " ", buf.starts[i], buf.ends[i])
        else:
            out.copy_from(buf, i)
    while out.chars and out.chars[-1] in " \n":
        for lst in (out.chars, out.starts, out.ends, out.protected):
            lst.pop()
    return out


def fold(text: str) -> str:
    """Map Kazakh-specific letters (and ё, э, ъ) to Russian look-alikes; keeps length."""
    return text.lower().translate(T.FOLD)


def normalize(text: str) -> NormalizedText:
    """Normalise ``text`` and extract entities; see the module docstring."""
    text = text or ""
    raw_entities = extract_raw_entities(text)
    protected = {i for e in raw_entities if e.kind in PROTECTED_KINDS for i in range(e.start, e.end)}
    buf = _initial_buffer(text, protected)
    buf = _join_spaced(buf)
    mode = _translit_mode(buf, buf.text())
    buf = _fix_words(buf, mode)
    lang = "translit" if mode else detect_language(buf.text())
    buf = _emoji_tokens(buf, lang, mode)
    buf = _collapse_spaces(buf)
    normalized = buf.text()
    folded = normalized.translate(T.FOLD)
    result = NormalizedText(
        original=text, text=normalized, folded=folded, starts=buf.starts, ends=buf.ends,
        lang=lang, translit=mode,
    )
    result.entities = raw_entities + extract_normalized_entities(result, raw_entities)
    return result
