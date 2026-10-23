"""Production hardening (beyond the TZ): sign-in, roles, audit, API tokens, limits, settings."""

from __future__ import annotations

import io
import logging
import os
import secrets
import subprocess
import sys

import pytest
from django.core.cache import cache
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.urls import reverse

import tasks
from app.dashboard.models import ApiToken, Case, Feedback
from app.settings import load_env_file
from safenet import DISCLAIMER_KK, paths
from safenet.common import find_forbidden
from tests.conftest import TEST_PASSWORD, make_user, require

PYRAMID = "Вложи 10 000 — через неделю получишь 30 000! Гарантированно, без риска. Пиши в лс"
TAGLINES = ("Схемалардың белгілерін анықтау — шешімді маман қабылдайды",
            "Выявление признаков схем — решение принимает специалист")
DASHBOARD_URLS = ["/", "/check/", "/network/", "/metrics/", "/case/x/"]


@pytest.fixture(autouse=True)
def _clean_cache():
    cache.clear()  # login failures and API throttling live in the cache
    yield
    cache.clear()


def bearer(key: str) -> dict:
    return {"HTTP_AUTHORIZATION": f"Bearer {key}"}


def post_json(client, url, body: str, **extra):
    return client.post(url, data=body, content_type="application/json", **extra)


def html(response) -> str:
    return response.content.decode("utf-8")


# ------------------------------------------------------------------ sign-in
@pytest.mark.django_db
@pytest.mark.parametrize("url", DASHBOARD_URLS)
def test_anonymous_is_sent_to_login(client, url):
    r = client.get(url)
    assert r.status_code == 302 and r["Location"].startswith(f"{reverse('login')}?next=")


@pytest.mark.django_db
def test_anonymous_api_and_docs(client):
    assert post_json(client, "/analyze", '{"text": "x"}').status_code == 401
    assert client.get("/docs").status_code == 302
    assert client.get("/openapi.json").status_code == 302
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["thresholds"] is None and health["model_loaded"] is None


@pytest.mark.django_db
@pytest.mark.parametrize("lang", ["kk", "ru"])
def test_login_page(client, lang):
    page = html(client.get(reverse("login"), {"lang": lang}))
    assert 'type="password"' in page and 'name="csrfmiddlewaretoken"' in page
    assert f'href="{reverse("check")}"' not in page and 'data-role="usermenu"' not in page  # no navigation
    assert not find_forbidden(page) and not any(t in page for t in TAGLINES)
    assert ("Жүйеге кіру" if lang == "kk" else "Вход в систему") in page


def test_login_and_logout(client, analyst):
    r = client.post(reverse("login"), {"username": "analyst1", "password": TEST_PASSWORD})
    assert r.status_code == 302 and r["Location"] == reverse("queue")
    page = html(client.get(reverse("queue")))
    assert 'data-role="usermenu"' in page and "analyst1" in page
    assert not any(t in page for t in TAGLINES)
    r = client.post(reverse("logout"))
    assert r.status_code == 302 and r["Location"] == reverse("login")
    assert client.get(reverse("queue")).status_code == 302


def test_login_ignores_foreign_next(client, analyst):
    r = client.post(reverse("login") + "?next=https://evil.example/", {"username": "analyst1",
                                                                       "password": TEST_PASSWORD,
                                                                       "next": "https://evil.example/"})
    assert r.status_code == 302 and "evil.example" not in r["Location"]


def test_lockout_after_repeated_failures(client, analyst, settings):
    settings.SAFENET_LOGIN_MAX_FAILURES = 3
    for _ in range(3):
        r = client.post(reverse("login"), {"username": "analyst1", "password": "wrong-password-1"})
        assert r.status_code == 200 and "Пайдаланушы аты немесе құпиясөз қате." in html(r)
    r = client.post(reverse("login"), {"username": "analyst1", "password": TEST_PASSWORD})
    assert r.status_code == 200 and "Кіру әрекеттері тым көп" in html(r)
    assert client.get(reverse("queue")).status_code == 302
    cache.clear()  # the lock expires
    r = client.post(reverse("login"), {"username": "analyst1", "password": TEST_PASSWORD})
    assert r.status_code == 302


def test_set_lang_does_not_redirect_off_site(client, analyst):
    client.force_login(analyst)
    r = client.get(reverse("set_lang", args=["ru"]), HTTP_REFERER="https://evil.example/phish")
    assert r.status_code == 302 and r["Location"] == reverse("queue")


# ------------------------------------------------------------------- roles
@pytest.fixture
def case(db):
    from app.dashboard.services import save_case
    from safenet.pipeline import analyze

    require(paths.MODEL_PATH, "make train")
    return save_case(analyze({"id": "prod-case-1", "text": PYRAMID, "source_type": "post"}), origin="check")


def test_observer_reads_but_cannot_decide(client, observer, case, settings, tmp_path):
    settings.SAFENET_FEEDBACK_PATH = str(tmp_path / "feedback.csv")
    client.force_login(observer)
    assert client.get(reverse("queue")).status_code == 200
    page = html(client.get(reverse("case", args=[case.id])))
    assert 'class="fb-form"' not in page and DISCLAIMER_KK in page
    r = client.post(reverse("case_feedback", args=[case.id]), {"verdict": "confirmed"})
    assert r.status_code == 403 and 'data-role="error-403"' in html(r)
    assert not Feedback.objects.exists()
    assert 'name="save"' not in html(client.get(reverse("check")))
    assert client.post(reverse("check"), {"text": PYRAMID, "save": "1"}).status_code == 403
    assert client.post(reverse("check"), {"text": PYRAMID}).status_code == 200


def test_user_without_role_is_refused(client, django_user_model, db):
    client.force_login(make_user(django_user_model, "nobody", None))
    for url in ["/", "/check/", "/network/", "/metrics/"]:
        r = client.get(url)
        assert r.status_code == 403 and 'data-role="error-403"' in html(r), url


def test_feedback_is_attributed_and_audited(client, analyst, case, settings, tmp_path, caplog):
    settings.SAFENET_FEEDBACK_PATH = str(tmp_path / "feedback.csv")
    client.force_login(analyst)
    with caplog.at_level(logging.INFO, logger="safenet.audit"):
        client.get(reverse("case", args=[case.id]))
        client.post(reverse("case_feedback", args=[case.id]), {"verdict": "confirmed", "comment": "тексерілді"})
    fb = Feedback.objects.get()
    assert fb.user == analyst and fb.username == "analyst1"
    lines = [r.getMessage() for r in caplog.records if r.name == "safenet.audit"]
    assert any("action=case_view user=analyst1" in line and f"case={case.id}" in line for line in lines)
    assert any("action=feedback user=analyst1" in line and "verdict=confirmed" in line for line in lines)


def test_old_feedback_csv_is_moved_aside(client, analyst, case, settings, tmp_path):
    path = tmp_path / "feedback.csv"
    path.write_text("timestamp,case_id,verdict\n2026-01-01,x,confirmed\n", encoding="utf-8")
    settings.SAFENET_FEEDBACK_PATH = str(path)
    client.force_login(analyst)
    client.post(reverse("case_feedback", args=[case.id]), {"verdict": "missed"})
    assert path.read_text(encoding="utf-8").splitlines()[0].endswith(",comment,user")
    assert len(list(tmp_path.glob("feedback-*.csv"))) == 1


# --------------------------------------------------------------------- API
def test_api_tokens(client, db):
    require(paths.MODEL_PATH, "make train")
    token, key = ApiToken.issue("collector")
    body = '{"text": "' + PYRAMID + '"}'
    assert post_json(client, "/analyze", body).status_code == 401
    assert post_json(client, "/analyze", body, **bearer("sn_wrong")).status_code == 401
    r = post_json(client, "/analyze", body, **bearer(key))
    assert r.status_code == 200 and r.json()["risk_level"] == "high"
    token.refresh_from_db()
    assert token.last_used_at is not None and key not in (token.key_hash, token.key_prefix)
    token.is_active = False
    token.save()
    assert post_json(client, "/analyze", body, **bearer(key)).status_code == 401


def test_api_session_needs_view_permission(client, django_user_model, observer):
    require(paths.MODEL_PATH, "make train")
    body = '{"text": "' + PYRAMID + '"}'
    client.force_login(make_user(django_user_model, "nobody", None))
    assert post_json(client, "/analyze", body).status_code == 401
    client.force_login(observer)
    assert post_json(client, "/analyze", body).status_code == 200
    save = '{"text": "' + PYRAMID + '", "save": true}'
    assert post_json(client, "/analyze", save).status_code == 403  # observers cannot add cases


def test_api_save_puts_result_into_queue(client, api_token, analyst, settings, tmp_path):
    require(paths.MODEL_PATH, "make train")
    settings.SAFENET_FEEDBACK_PATH = str(tmp_path / "feedback.csv")
    body = '{"id": "tg-1001", "text": "' + PYRAMID + '", "platform": "telegram", "save": true}'
    r = post_json(client, "/analyze", body, **bearer(api_token))
    assert r.status_code == 200 and r.json()["id"] == "tg-1001"
    case = Case.objects.get(pk="tg-1001")
    assert case.origin == "api" and case.risk_level == "high" and case.platform == "telegram"
    client.force_login(analyst)
    client.post(reverse("case_feedback", args=[case.id]), {"verdict": "confirmed"})
    post_json(client, "/analyze", body, **bearer(api_token))  # re-analysis of the same id
    assert Case.objects.filter(pk="tg-1001").count() == 1
    assert Feedback.objects.get().case_id == "tg-1001"  # the verdict stays linked
    items = '[{"text": "' + PYRAMID + '", "save": true}, {"text": "Свежий хлеб, 300 тг", "save": false}]'
    r = post_json(client, "/batch", items, **bearer(api_token))
    assert r.status_code == 200
    saved = r.json()[0]["id"]
    assert saved.startswith("api-") and Case.objects.filter(pk=saved, origin="api").exists()
    assert Case.objects.filter(origin="api").count() == 2


def test_api_rate_limit(client, api_token, settings):
    settings.SAFENET_API_RATE = "3/m"
    codes = [post_json(client, "/batch", "[]", **bearer(api_token)).status_code for _ in range(4)]
    assert codes[:3] == [422, 422, 422] and codes[3] == 429


def test_health_details_with_token(client, api_token):
    data = client.get("/health", **bearer(api_token)).json()
    assert data["status"] == "ok" and set(data["thresholds"]) == {"medium", "high"}


def test_openapi_has_no_verdict_wording(client, analyst):
    client.force_login(analyst)
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    text = schema.content.decode("utf-8")
    assert not find_forbidden(text) and "bearer" in text.lower()


def test_create_api_token_command(analyst):
    out = io.StringIO()
    call_command("create_api_token", "telegram-collector", "--user", "analyst1", stdout=out)
    key = next(line for line in out.getvalue().splitlines() if line.startswith(ApiToken.PREFIX))
    token = ApiToken.objects.get()
    assert token.user == analyst and token.key_hash == ApiToken.hash_key(key) and ApiToken.authenticate(key) == token
    with pytest.raises(Exception):
        call_command("create_api_token", "x", "--user", "ghost", stdout=io.StringIO())


# ---------------------------------------------------------------- operations
def test_purge_cases_command(db):
    for origin in ("dataset", "demo", "check", "api"):
        Case.objects.create(id=f"c-{origin}", risk_score=50, risk_level="medium", primary_track="lottery",
                            result={}, origin=origin)
    Feedback.objects.create(case_id="c-demo", case_ref="c-demo", verdict="confirmed")
    out = io.StringIO()
    call_command("purge_cases", "--origin", "dataset", "demo", stdout=out)
    assert "would be deleted" in out.getvalue() and Case.objects.count() == 4
    call_command("purge_cases", "--origin", "dataset", "demo", "--yes", stdout=io.StringIO())
    assert set(Case.objects.values_list("origin", flat=True)) == {"check", "api"}
    assert Feedback.objects.get().case_ref == "c-demo"  # history kept


def test_error_pages(client, analyst):
    r = client.get("/no-such-page/")
    assert r.status_code == 404 and 'data-role="error-404"' in html(r)
    client.force_login(analyst)
    r = client.get(reverse("case", args=["missing"]))
    assert r.status_code == 404 and 'data-role="error-404"' in html(r)


def test_security_headers(client, analyst):
    client.force_login(analyst)
    r = client.get(reverse("queue"))
    assert r["X-Frame-Options"] == "DENY" and r["X-Content-Type-Options"] == "nosniff"
    assert r["Referrer-Policy"] == "same-origin"


def test_upload_size_limit(client, analyst, settings):
    settings.SAFENET_MAX_IMAGE_BYTES = 100
    client.force_login(analyst)
    big = SimpleUploadedFile("big.png", b"\x89PNG" + b"0" * 500, content_type="image/png")
    r = client.post(reverse("check"), {"text": "", "image": big})
    assert r.status_code == 422 and "10 МБ" in html(r)


# ----------------------------------------------------------------- settings
def _settings_probe(env: dict, tmp_path, code: str) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if not k.startswith(("SAFENET_", "DJANGO_"))}
    full |= {"SAFENET_LOG_DIR": str(tmp_path / "logs"), "PYTHONUTF8": "1", **env}
    return subprocess.run([sys.executable, "-c", code], cwd=paths.ROOT, env=full, capture_output=True,
                          text=True, timeout=120)


PROBE = ("import app.settings as s; print(s.DEBUG, s.SESSION_COOKIE_SECURE, s.CSRF_COOKIE_SECURE, "
         "s.SECURE_HSTS_SECONDS, s.SECURE_SSL_REDIRECT, 'whitenoise.middleware.WhiteNoiseMiddleware' in s.MIDDLEWARE)")


def test_production_settings_require_secrets(tmp_path):
    r = _settings_probe({"SAFENET_ENV": "production"}, tmp_path, PROBE)
    assert r.returncode != 0 and "DJANGO_SECRET_KEY is required" in r.stderr
    for weak in ("short", "k" * 60, "django-insecure-" + secrets.token_urlsafe(48)):
        r = _settings_probe({"SAFENET_ENV": "production", "DJANGO_SECRET_KEY": weak}, tmp_path, PROBE)
        assert r.returncode != 0 and "at least 50 random characters" in r.stderr, weak
    r = _settings_probe({"SAFENET_ENV": "production", "DJANGO_SECRET_KEY": secrets.token_urlsafe(64)}, tmp_path,
                        PROBE)
    assert r.returncode != 0 and "DJANGO_ALLOWED_HOSTS is required" in r.stderr
    r = _settings_probe({"SAFENET_ENV": "production", "DJANGO_SECRET_KEY": secrets.token_urlsafe(64),
                         "DJANGO_DEBUG": "1", "DJANGO_ALLOWED_HOSTS": "x.kz"}, tmp_path, PROBE)
    assert r.returncode != 0 and "DJANGO_DEBUG must be off" in r.stderr


def test_production_settings_are_secure(tmp_path):
    env = {"SAFENET_ENV": "production", "DJANGO_SECRET_KEY": secrets.token_urlsafe(64),
           "DJANGO_ALLOWED_HOSTS": "safenet.example.kz"}
    r = _settings_probe(env, tmp_path, PROBE)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "True", "True", "31536000", "True", "True"]
    check = subprocess.run([sys.executable, "manage.py", "check", "--deploy", "--fail-level", "WARNING"],
                           cwd=paths.ROOT, capture_output=True, text=True, timeout=180,
                           env={**{k: v for k, v in os.environ.items() if not k.startswith(("SAFENET_", "DJANGO_"))},
                                **env, "SAFENET_LOG_DIR": str(tmp_path / "logs"), "PYTHONUTF8": "1"})
    assert check.returncode == 0, check.stdout + check.stderr
    assert (tmp_path / "logs").is_dir()


def test_development_defaults(tmp_path):
    r = _settings_probe({}, tmp_path, PROBE)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["True", "False", "False", "0", "False", "False"]


def test_env_file_loader(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# comment\nexport SAFENET_T1=one\nSAFENET_T2 = \"two words\"\nSAFENET_T3='x=y'\n"
                    "SAFENET_T4=from-file\nnot a line\n", encoding="utf-8")
    for name in ("SAFENET_T1", "SAFENET_T2", "SAFENET_T3"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SAFENET_T4", "from-env")
    loaded = load_env_file(path)
    assert loaded == {"SAFENET_T1": "one", "SAFENET_T2": "two words", "SAFENET_T3": "x=y"}
    assert os.environ["SAFENET_T4"] == "from-env"  # the environment wins
    for name in loaded:
        monkeypatch.delenv(name)


def test_make_serve_runs_production_server(monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != "SAFENET_ENV"})  # no leak
    steps, served = [], []
    monkeypatch.setattr(tasks, "run_all", lambda *cmds: steps.extend(c[2:] for c in cmds) or 0)
    monkeypatch.setattr(tasks, "run", lambda *cmd: served.append(cmd) or 0)
    assert tasks.main(["serve", "PORT=9000"]) == 0
    assert os.environ["SAFENET_ENV"] == "production"
    assert [s[0] for s in steps] == ["migrate", "collectstatic", "check"] and "--deploy" in steps[2]
    assert served[0][1:3] == ("-m", "waitress") and "--listen=0.0.0.0:9000" in served[0]
    assert served[0][-1] == "app.wsgi:application"
