"""Audit trail: who did what, from where (``logs/audit.log`` in production)."""

from __future__ import annotations

import logging

from django.conf import settings
from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver

logger = logging.getLogger("safenet.audit")


def client_ip(request) -> str:
    if request is None:
        return "-"
    if getattr(settings, "SECURE_PROXY_SSL_HEADER", None):  # behind a reverse proxy
        forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "-")


def _clean(value) -> str:
    return str(value).replace("\n", " ").replace("\r", " ")[:200]


def audit(request, action: str, *, actor: str | None = None, **fields) -> None:
    """One line per action: ``action=... user=... ip=... key=value ...``."""
    if actor is None:
        user = getattr(request, "user", None)
        actor = user.get_username() if user is not None and user.is_authenticated else "-"
    extra = " ".join(f"{k}={_clean(v)}" for k, v in fields.items())
    logger.info("action=%s user=%s ip=%s %s", action, _clean(actor), client_ip(request), extra)


@receiver(user_logged_in)
def _logged_in(sender, request, user, **kwargs):
    audit(request, "login", actor=user.get_username())


@receiver(user_logged_out)
def _logged_out(sender, request, user, **kwargs):
    audit(request, "logout", actor=user.get_username() if user else "-")


@receiver(user_login_failed)
def _login_failed(sender, credentials, request=None, **kwargs):
    audit(request, "login_failed", actor=credentials.get("username", "-"))
