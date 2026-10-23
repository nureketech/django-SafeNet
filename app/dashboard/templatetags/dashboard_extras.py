"""Small template helpers (registered as builtins in settings)."""

import re

from django import template

from safenet.common import mask_card, mask_phone

register = template.Library()

CARD = re.compile(r"(?<!\d)\d{4}(?:[ -]?\d{4}){3}(?!\d)")
PHONE = re.compile(r"(?<![\d+])(?:\+7|8)[ (-]*7\d\d[) -]*\d{3}[ -]*\d{2}[ -]*\d{2}(?!\d)")


@register.filter
def get_item(mapping, key):
    """``{{ mapping|get_item:key }}`` — dictionary lookup with a variable key."""
    try:
        return mapping.get(key, key)
    except AttributeError:
        return key


@register.filter
def mask_numbers(text):
    """Card and phone numbers in a snippet as masks (pages that list many cases at once)."""
    text = CARD.sub(lambda m: mask_card(m.group(0)), str(text or ""))
    return PHONE.sub(lambda m: mask_phone(m.group(0)), text)


@register.filter
def pct(value, digits=1):
    """0.9734 -> '97.3%'."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "—"
