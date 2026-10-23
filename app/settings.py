"""Django settings for SafeNet KZ.

Two modes, chosen by ``SAFENET_ENV``:

* ``development`` (default) — debug on, an insecure fallback key, plain HTTP; used by
  ``make app`` / ``make demo`` and the test-suite;
* ``production`` — debug off, ``DJANGO_SECRET_KEY`` and ``DJANGO_ALLOWED_HOSTS`` are
  required, secure cookies and HSTS (unless ``SAFENET_HTTPS=0``), static files through
  WhiteNoise, file logs; used by ``make serve``. See ``docs/DEPLOYMENT.md``.

Values come from the environment; a ``.env`` file in the project root is read first
(existing environment variables win).
"""

from __future__ import annotations

import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent


def load_env_file(path: Path) -> dict[str, str]:
    """``KEY=value`` lines of ``path`` (``#`` comments, optional quotes); does not override ``os.environ``."""
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


load_env_file(BASE_DIR / ".env")


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: str = "") -> list[str]:
    return [item.strip() for item in env(name, default).split(",") if item.strip()]


SAFENET_ENV = env("SAFENET_ENV", "development").lower()
if SAFENET_ENV not in ("development", "production"):
    raise ImproperlyConfigured("SAFENET_ENV must be 'development' or 'production'")
PRODUCTION = SAFENET_ENV == "production"

DEBUG = env_bool("DJANGO_DEBUG", not PRODUCTION)
if PRODUCTION and DEBUG:
    raise ImproperlyConfigured("DJANGO_DEBUG must be off in production")

SECRET_KEY = env("DJANGO_SECRET_KEY")
if not SECRET_KEY:
    if PRODUCTION:
        raise ImproperlyConfigured("DJANGO_SECRET_KEY is required in production (see .env.example)")
    SECRET_KEY = "dev-only-insecure-key-never-use-in-production"
elif PRODUCTION and (len(SECRET_KEY) < 50 or len(set(SECRET_KEY)) < 5 or SECRET_KEY.startswith("django-insecure-")):
    raise ImproperlyConfigured("DJANGO_SECRET_KEY must be at least 50 random characters long")

ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "" if PRODUCTION else "127.0.0.1,localhost,testserver")
if PRODUCTION and not ALLOWED_HOSTS:
    raise ImproperlyConfigured("DJANGO_ALLOWED_HOSTS is required in production")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "ninja",
    "app.dashboard",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # static files from STATIC_ROOT (after collectstatic); the dev server serves them itself
    *(["whitenoise.middleware.WhiteNoiseMiddleware"] if PRODUCTION else []),
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.auth.middleware.LoginRequiredMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "app.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "app.dashboard.context.ui",
            ],
            "builtins": ["app.dashboard.templatetags.dashboard_extras"],
        },
    },
]

WSGI_APPLICATION = "app.wsgi.application"

# ------------------------------------------------------------------ database
# SQLite by default; PostgreSQL for production: SAFENET_DB_ENGINE=postgresql (+ psycopg, requirements-prod.txt).
_db_engine = env("SAFENET_DB_ENGINE", "sqlite").lower()
if _db_engine in ("postgres", "postgresql"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": env("SAFENET_DB_NAME", "safenet"),
            "USER": env("SAFENET_DB_USER", "safenet"),
            "PASSWORD": env("SAFENET_DB_PASSWORD"),
            "HOST": env("SAFENET_DB_HOST", "localhost"),
            "PORT": env("SAFENET_DB_PORT", "5432"),
            "CONN_MAX_AGE": 60,
        }
    }
elif _db_engine == "sqlite":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": env("SAFENET_DB", str(BASE_DIR / "data" / "safenet.sqlite3")),
        }
    }
else:
    raise ImproperlyConfigured("SAFENET_DB_ENGINE must be 'sqlite' or 'postgresql'")

# ------------------------------------------------------------------ auth
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "queue"
LOGOUT_REDIRECT_URL = "login"
AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", "OPTIONS": {"min_length": 10}},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]
SESSION_COOKIE_AGE = int(env("SAFENET_SESSION_HOURS", "8")) * 3600
SESSION_EXPIRE_AT_BROWSER_CLOSE = True
# failed logins per username + address before a temporary lock
SAFENET_LOGIN_MAX_FAILURES = int(env("SAFENET_LOGIN_MAX_FAILURES", "5"))
SAFENET_LOGIN_LOCK_MINUTES = int(env("SAFENET_LOGIN_LOCK_MINUTES", "15"))

# ------------------------------------------------------------------ security
HTTPS = env_bool("SAFENET_HTTPS", PRODUCTION)
SESSION_COOKIE_SECURE = HTTPS
CSRF_COOKIE_SECURE = HTTPS
SECURE_SSL_REDIRECT = env_bool("SAFENET_SSL_REDIRECT", HTTPS)
SECURE_HSTS_SECONDS = int(env("SAFENET_HSTS_SECONDS", "31536000" if HTTPS else "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = HTTPS
SECURE_HSTS_PRELOAD = env_bool("SAFENET_HSTS_PRELOAD", False)
# Preload-list submission is hard to undo, so it is an explicit opt-in rather than a deploy warning.
SILENCED_SYSTEM_CHECKS = [] if SECURE_HSTS_PRELOAD else ["security.W021"]
if env_bool("SAFENET_BEHIND_PROXY", False):  # TLS terminated by nginx / IIS
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
    USE_X_FORWARDED_HOST = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
X_FRAME_OPTIONS = "DENY"  # the graph view opts in to SAMEORIGIN itself

# ------------------------------------------------------------------ i18n
LANGUAGE_CODE = "kk"
TIME_ZONE = "Asia/Almaty"
USE_I18N = True
USE_TZ = True

# ------------------------------------------------------------------ static files
STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {
        "BACKEND": ("whitenoise.storage.CompressedManifestStaticFilesStorage" if PRODUCTION
                    else "django.contrib.staticfiles.storage.StaticFilesStorage"),
    },
}

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Uploaded screenshots are analysed in memory and never stored.
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
SAFENET_MAX_IMAGE_BYTES = 10 * 1024 * 1024

# ------------------------------------------------------------------ SafeNet
# Analyst feedback: every verdict is appended here as well as stored in the DB.
SAFENET_FEEDBACK_PATH = env("SAFENET_FEEDBACK_PATH", str(BASE_DIR / "data" / "feedback" / "feedback.csv"))
# Demo helpers (example texts on the check page); off unless asked for.
SAFENET_DEMO_MODE = env_bool("SAFENET_DEMO_MODE", False)
# API: requests per token ("<n>/<s|m|h|d>").
SAFENET_API_RATE = env("SAFENET_API_RATE", "600/m")

# ------------------------------------------------------------------ logging
LOG_DIR = Path(env("SAFENET_LOG_DIR", str(BASE_DIR / "logs")))
_handlers = {"console": {"class": "logging.StreamHandler", "formatter": "plain"}}
if PRODUCTION:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _handlers |= {
        "file": {"class": "logging.handlers.RotatingFileHandler", "filename": str(LOG_DIR / "safenet.log"),
                 "maxBytes": 10 * 1024 * 1024, "backupCount": 10, "encoding": "utf-8", "formatter": "plain"},
        "audit": {"class": "logging.handlers.RotatingFileHandler", "filename": str(LOG_DIR / "audit.log"),
                  "maxBytes": 10 * 1024 * 1024, "backupCount": 50, "encoding": "utf-8", "formatter": "plain"},
    }
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
    "handlers": _handlers,
    "root": {"handlers": ["console", "file"] if PRODUCTION else ["console"], "level": "INFO"},
    "loggers": {
        "django": {"handlers": ["console", "file"] if PRODUCTION else ["console"], "level": "INFO",
                   "propagate": False},
        "django.server": {"level": "INFO"},
        # production: a separate audit file only; development: through the root console handler
        "safenet.audit": ({"handlers": ["audit"], "level": "INFO", "propagate": False} if PRODUCTION
                          else {"level": "INFO"}),
    },
}
