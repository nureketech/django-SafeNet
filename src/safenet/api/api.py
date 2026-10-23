"""HTTP API (django-ninja), mounted at the site root by ``app/urls.py``.

* ``POST /analyze``        — JSON ``{text, source_type?, platform?, account_id?, group_id?, source_url?, id?, save?}``
                             -> AnalysisResult
* ``POST /analyze/image``  — multipart: ``image`` (+ optional ``text``, ``source_type``, ..., ``save``) -> AnalysisResult
* ``POST /batch``          — JSON list of analyze items -> list[AnalysisResult]
* ``GET  /health``         — service status (details only for authenticated callers)
* ``GET  /docs``           — OpenAPI documentation (Swagger UI) for signed-in users, schema at ``/openapi.json``

Authentication: ``Authorization: Bearer <token>`` (tokens: ``manage.py create_api_token``) or
the dashboard session of a user with the ``dashboard.view_case`` permission. Requests are
rate-limited per token (``SAFENET_API_RATE``). ``save: true`` puts the result into the
analyst queue (origin ``api``). ``account_id``, ``group_id`` and ``source_url`` (the account is read
from a t.me / Instagram / TikTok / VK / YouTube link when not given) link saved cases in the live
account graph, which also gives the network bonus. Invalid input returns HTTP 422 with the details.
"""

from __future__ import annotations

import hashlib
import io
from typing import Literal

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest
from ninja import File, Form, NinjaAPI, Schema
from ninja.errors import HttpError
from ninja.files import UploadedFile
from ninja.security import HttpBearer, SessionAuth
from ninja.throttling import AuthRateThrottle
from PIL import Image, UnidentifiedImageError
from pydantic import Field

from safenet import DISCLAIMER_KK, DISCLAIMER_RU, __version__, paths
from safenet.pipeline import analyze, analyze_many
from safenet.scoring.config import load_config
from safenet.scoring.models import AnalysisResult

SourceType = Literal["post", "chat_message", "stream_transcript", "screenshot", "account_bio"]
Platform = Literal["telegram", "whatsapp", "instagram", "tiktok", "youtube", "vk"]
MAX_BATCH = 500


class TokenAuth(HttpBearer):
    """``Authorization: Bearer sn_...`` — an active :class:`ApiToken`."""

    def authenticate(self, request: HttpRequest, token: str):
        from app.dashboard.models import ApiToken

        return ApiToken.authenticate(token)


class AnalystSession(SessionAuth):
    """Dashboard session (for ``/docs`` → Try it out) of a user allowed to view cases."""

    def authenticate(self, request: HttpRequest, key):
        user = super().authenticate(request, key)
        return user if user is not None and user.has_perm("dashboard.view_case") else None


class ApiRateThrottle(AuthRateThrottle):
    """Per-token (or per-user) limit read from ``settings.SAFENET_API_RATE`` on every request."""

    def __init__(self):
        super().__init__(settings.SAFENET_API_RATE)

    def allow_request(self, request: HttpRequest) -> bool:
        self.rate = settings.SAFENET_API_RATE
        self.num_requests, self.duration = self.parse_rate(self.rate)
        return super().allow_request(request)


api = NinjaAPI(
    title="SafeNet KZ API",
    version=__version__,
    description=(
        "Әлеуметтік желі контентіндегі ақылы ұтыс, онлайн-казино жарнамасы және қаржы пирамидасы белгілерін "
        "анықтау. Жауапта тәуекел деңгейі, белгілер, дәлелдер және ескерту қайтарылады. "
        "Аутентификация: `Authorization: Bearer <token>`. "
        f"{DISCLAIMER_KK} / {DISCLAIMER_RU}"
    ),
    docs_url="/docs",
    docs_decorator=login_required,
    urls_namespace="safenet-api",
    auth=[TokenAuth(), AnalystSession()],
    throttle=[ApiRateThrottle()],
)


@api.exception_handler(HttpError)
def http_error(request: HttpRequest, exc: HttpError):
    """Unparsable bodies are invalid input too: 422 instead of ninja's 400."""
    status = exc.status_code
    if status == 400 and str(exc).startswith("Cannot parse request body"):
        status = 422
    return api.create_response(request, {"detail": str(exc)}, status=status)


def _caller(request: HttpRequest) -> str:
    auth = getattr(request, "auth", None)
    if auth is None:
        return "-"
    return f"token:{auth.name}" if hasattr(auth, "key_prefix") else f"user:{auth.get_username()}"


def _may_save(request: HttpRequest) -> None:
    """Tokens may save; dashboard users need the ``dashboard.add_case`` permission."""
    auth = getattr(request, "auth", None)
    if auth is not None and not hasattr(auth, "key_prefix") and not auth.has_perm("dashboard.add_case"):
        raise HttpError(403, "saving to the queue requires the dashboard.add_case permission")


def _context(account_id: str | None, group_id: str | None, source_url: str | None, platform: str | None):
    from app.dashboard.sources import ContextError, build_context

    try:
        return build_context(account=account_id or "", group=group_id or "", source_url=source_url or "",
                             platform=platform or "")
    except ContextError as exc:
        raise HttpError(422, "source_url must be an http(s) link") from exc


def _lookup():
    """Network bonus from the live graph of real saved cases (never the synthetic research graph)."""
    from app.dashboard.network import live_lookup

    return live_lookup()


def _item(data: dict, context) -> dict:
    return {**data, "platform": context.platform or None, "account_id": context.account_id or None}


def _store(result: AnalysisResult, item_id: str | None, text: str, context) -> AnalysisResult:
    from app.dashboard.services import save_case

    key = f"{result.source_type}|{text}|{result.ocr_text or ''}"
    if context.account_id or context.source_url:
        key += f"|{context.account_id}|{context.source_url}"
    result.id = (item_id or f"api-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:12]}")[:64]
    save_case(result, origin="api", group_id=context.group_id, source_url=context.source_url)
    return result


def _audit(request: HttpRequest, action: str, **fields) -> None:
    from app.dashboard.audit import audit

    audit(request, action, actor=_caller(request), **fields)


class AnalyzeIn(Schema):
    text: str = Field(min_length=1, max_length=20000, description="Пост, хабарлама, эфир транскрипциясы немесе био")
    id: str | None = Field(default=None, max_length=64)
    source_type: SourceType | None = None
    platform: Platform | None = None
    account_id: str | None = Field(default=None, max_length=100, description="Аккаунт / арна, мысалы @channel")
    group_id: str | None = Field(default=None, max_length=100, description="Топ / чат")
    source_url: str | None = Field(default=None, max_length=500, description="Посттың сілтемесі")
    save: bool = Field(default=False, description="Нәтижені аналитиктер кезегіне қосу")


class HealthOut(Schema):
    status: str
    version: str
    model_loaded: bool | None = None
    graph_loaded: bool | None = None
    ocr_engine: str | None = None
    thresholds: dict[str, float] | None = None


@api.get("/health", response=HealthOut, tags=["service"], summary="Сервис күйі", auth=None)
def health(request: HttpRequest):
    out = {"status": "ok", "version": __version__}
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not (request.user.is_authenticated or TokenAuth().authenticate(request, token)):
        return out  # monitoring without credentials sees only that the service is up
    from app.dashboard.network import live_graph
    from safenet.multimodal.ocr import available_engine

    cfg = load_config()
    return out | {
        "model_loaded": paths.MODEL_PATH.exists(),
        "graph_loaded": live_graph() is not None,
        "ocr_engine": available_engine()[0],
        "thresholds": {"medium": cfg.medium, "high": cfg.high},
    }


@api.post("/analyze", response=AnalysisResult, tags=["analysis"], summary="Мәтінді талдау")
def analyze_text(request: HttpRequest, payload: AnalyzeIn):
    if payload.save:
        _may_save(request)
    context = _context(payload.account_id, payload.group_id, payload.source_url, payload.platform)
    result = analyze(_item(payload.model_dump(exclude={"save", "group_id", "source_url"}), context),
                     network_lookup=_lookup())
    if payload.save:
        _store(result, payload.id, payload.text, context)
    _audit(request, "api_analyze", level=result.risk_level, saved=result.id if payload.save else "-")
    return result


@api.post("/analyze/image", response=AnalysisResult, tags=["analysis"],
          summary="Скриншотты талдау (OCR) + қосымша пост мәтіні")
def analyze_image(
    request: HttpRequest,
    image: File[UploadedFile],
    text: Form[str] = "",
    source_type: Form[SourceType | None] = None,
    platform: Form[Platform | None] = None,
    account_id: Form[str | None] = None,
    group_id: Form[str | None] = None,
    source_url: Form[str | None] = None,
    id: Form[str | None] = None,
    save: Form[bool] = False,
):
    limit = settings.SAFENET_MAX_IMAGE_BYTES
    if image.size and image.size > limit:
        raise HttpError(422, f"image is larger than {limit // (1024 * 1024)} MB")
    if save:
        _may_save(request)
    context = _context(account_id, group_id, source_url, platform)
    data = image.read()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HttpError(422, f"not a valid image: {exc}") from exc
    item = _item({"text": text or "", "source_type": source_type or ("post" if text else "screenshot")}, context)
    result = analyze(item, image=data, network_lookup=_lookup())
    if save:
        _store(result, id, text or "", context)
    _audit(request, "api_analyze_image", level=result.risk_level, saved=result.id if save else "-")
    return result


@api.post("/batch", response=list[AnalysisResult], tags=["analysis"], summary="Бірнеше жазбаны талдау")
def batch(request: HttpRequest, items: list[AnalyzeIn]):
    if not items:
        raise HttpError(422, "items must not be empty")
    if len(items) > MAX_BATCH:
        raise HttpError(422, f"at most {MAX_BATCH} items per request")
    if any(i.save for i in items):
        _may_save(request)
    contexts = [_context(i.account_id, i.group_id, i.source_url, i.platform) for i in items]
    results = analyze_many([_item(i.model_dump(exclude={"save", "group_id", "source_url"}), c)
                            for i, c in zip(items, contexts)], use_ocr=False, network_lookup=_lookup())
    for item, context, result in zip(items, contexts, results):
        if item.save:
            _store(result, item.id, item.text, context)
    _audit(request, "api_batch", items=len(items), saved=sum(i.save for i in items))
    return results
