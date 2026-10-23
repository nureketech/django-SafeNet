"""Sign-in with a temporary lock after repeated failures (per username and address)."""

from __future__ import annotations

from django import forms
from django.conf import settings
from django.contrib.auth import views as auth_views
from django.contrib.auth.forms import AuthenticationForm
from django.core.cache import cache

from app.dashboard import i18n
from app.dashboard.audit import audit, client_ip


def _lock_key(username: str, ip: str) -> str:
    return f"safenet:login-failures:{username.strip().lower()}:{ip}"


class LoginForm(AuthenticationForm):
    def __init__(self, request=None, *args, **kwargs):
        super().__init__(request, *args, **kwargs)
        t = i18n.UI[i18n.get_lang(request)] if request is not None else i18n.UI["kk"]
        self.fields["username"].label = t["login_username"]
        self.fields["password"].label = t["login_password"]
        self.error_messages = {**self.error_messages, "invalid_login": t["login_invalid"],
                               "inactive": t["login_inactive"], "locked": t["login_locked"]}

    def clean(self):
        username = self.data.get("username", "")
        key = _lock_key(username, client_ip(self.request))
        if cache.get(key, 0) >= settings.SAFENET_LOGIN_MAX_FAILURES:
            audit(self.request, "login_locked", actor=username)
            raise forms.ValidationError(self.error_messages["locked"], code="locked")
        try:
            cleaned = super().clean()
        except forms.ValidationError:
            failures = cache.get(key, 0) + 1
            cache.set(key, failures, timeout=settings.SAFENET_LOGIN_LOCK_MINUTES * 60)
            raise
        cache.delete(key)
        return cleaned


class LoginView(auth_views.LoginView):
    template_name = "dashboard/login.html"
    authentication_form = LoginForm
    redirect_authenticated_user = True
