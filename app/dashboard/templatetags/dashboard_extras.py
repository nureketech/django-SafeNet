"""Small template helpers (registered as builtins in settings)."""

from django import template

register = template.Library()


@register.filter
def get_item(mapping, key):
    """``{{ mapping|get_item:key }}`` — dictionary lookup with a variable key."""
    try:
        return mapping.get(key, key)
    except AttributeError:
        return key


@register.filter
def pct(value, digits=1):
    """0.9734 -> '97.3%'."""
    try:
        return f"{float(value) * 100:.{int(digits)}f}%"
    except (TypeError, ValueError):
        return "—"
