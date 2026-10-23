"""HTTP API (django-ninja), mounted at the site root by ``app/urls.py``.

* ``POST /analyze``        — JSON ``{text, source_type?, platform?, account_id?, id?}`` -> AnalysisResult
* ``POST /analyze/image``  — multipart: ``image`` (+ optional ``text``, ``source_type``, ...) -> AnalysisResult
* ``POST /batch``          — JSON list of analyze items -> list[AnalysisResult]
* ``GET  /health``         — service status
* ``GET  /docs``           — OpenAPI documentation (Swagger UI), schema at ``/openapi.json``

Invalid input returns HTTP 422 with the validation details.
"""

from __future__ import annotations

import io
from typing import Literal

from django.http import HttpRequest
from ninja import File, Form, NinjaAPI, Schema
from ninja.errors import HttpError
from ninja.files import UploadedFile
from PIL import Image, UnidentifiedImageError
from pydantic import Field

from safenet import DISCLAIMER_KK, DISCLAIMER_RU, __version__, paths
from safenet.pipeline import analyze, analyze_many
from safenet.scoring.config import load_config
from safenet.scoring.models import AnalysisResult

SourceType = Literal["post", "chat_message", "stream_transcript", "screenshot", "account_bio"]
Platform = Literal["telegram", "whatsapp", "instagram", "tiktok", "youtube", "vk"]
MAX_BATCH = 500
MAX_IMAGE_BYTES = 10 * 1024 * 1024

api = NinjaAPI(
    title="SafeNet KZ API",
    version=__version__,
    description=(
        "Әлеуметтік желі контентіндегі заңсыз схемалардың **белгілерін** анықтау (лотерея, онлайн-казино, "
        "пирамида). Жүйе үкім шығармайды: белгілерді, тәуекел деңгейін және түсіндірмені қайтарады. "
        f"{DISCLAIMER_KK} / {DISCLAIMER_RU}"
    ),
    docs_url="/docs",
    urls_namespace="safenet-api",
)


@api.exception_handler(HttpError)
def http_error(request: HttpRequest, exc: HttpError):
    """Unparsable bodies are invalid input too: 422 instead of ninja's 400."""
    status = exc.status_code
    if status == 400 and str(exc).startswith("Cannot parse request body"):
        status = 422
    return api.create_response(request, {"detail": str(exc)}, status=status)


class AnalyzeIn(Schema):
    text: str = Field(min_length=1, max_length=20000, description="Пост, хабарлама, эфир транскрипциясы немесе био")
    id: str | None = Field(default=None, max_length=100)
    source_type: SourceType | None = None
    platform: Platform | None = None
    account_id: str | None = Field(default=None, max_length=100)


class HealthOut(Schema):
    status: str
    version: str
    model_loaded: bool
    graph_loaded: bool
    ocr_engine: str | None
    thresholds: dict[str, float]


@api.get("/health", response=HealthOut, tags=["service"], summary="Сервис күйі")
def health(request: HttpRequest):
    from safenet.graph.lookup import load_graph
    from safenet.multimodal.ocr import available_engine

    cfg = load_config()
    return {
        "status": "ok",
        "version": __version__,
        "model_loaded": paths.MODEL_PATH.exists(),
        "graph_loaded": load_graph() is not None,
        "ocr_engine": available_engine()[0],
        "thresholds": {"medium": cfg.medium, "high": cfg.high},
    }


@api.post("/analyze", response=AnalysisResult, tags=["analysis"], summary="Мәтінді талдау")
def analyze_text(request: HttpRequest, payload: AnalyzeIn):
    return analyze(payload.model_dump())


@api.post("/analyze/image", response=AnalysisResult, tags=["analysis"],
          summary="Скриншотты талдау (OCR) + қосымша пост мәтіні")
def analyze_image(
    request: HttpRequest,
    image: File[UploadedFile],
    text: Form[str] = "",
    source_type: Form[SourceType | None] = None,
    platform: Form[Platform | None] = None,
    account_id: Form[str | None] = None,
):
    if image.size and image.size > MAX_IMAGE_BYTES:
        raise HttpError(422, f"image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB")
    data = image.read()
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HttpError(422, f"not a valid image: {exc}") from exc
    item = {"text": text or "", "source_type": source_type or ("post" if text else "screenshot"),
            "platform": platform, "account_id": account_id}
    return analyze(item, image=data)


@api.post("/batch", response=list[AnalysisResult], tags=["analysis"], summary="Бірнеше жазбаны талдау")
def batch(request: HttpRequest, items: list[AnalyzeIn]):
    if not items:
        raise HttpError(422, "items must not be empty")
    if len(items) > MAX_BATCH:
        raise HttpError(422, f"at most {MAX_BATCH} items per request")
    return analyze_many([i.model_dump() for i in items], use_ocr=False)
