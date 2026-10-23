"""Template context: interface texts of the selected language."""

from __future__ import annotations

from safenet import DISCLAIMER_KK, DISCLAIMER_RU, __version__
from app.dashboard import i18n


def ui(request) -> dict:
    lang = i18n.get_lang(request)
    return {
        "lang": lang,
        "other_lang": "ru" if lang == "kk" else "kk",
        "t": i18n.UI[lang],
        "track_names": i18n.TRACKS[lang],
        "source_names": i18n.SOURCES[lang],
        "lang_names": i18n.LANG_NAMES[lang],
        "platform_names": i18n.PLATFORM_NAMES,
        "entity_names": i18n.ENTITY_KINDS[lang],
        "disclaimer": DISCLAIMER_KK if lang == "kk" else DISCLAIMER_RU,
        "version": __version__,
    }
