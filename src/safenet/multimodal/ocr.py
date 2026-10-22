"""OCR for screenshots: Tesseract (``rus+kaz+eng``) with an EasyOCR fallback.

Image preprocessing before OCR: grayscale, upscaling of small images, inversion of
dark themes, median denoising, removal of solid colour bars, deskew (projection
profile search), Otsu binarisation.

Configuration (environment variables):

* ``TESSERACT_CMD`` — path to ``tesseract`` (default: ``PATH`` or
  ``C:\\Program Files\\Tesseract-OCR\\tesseract.exe``);
* ``TESSDATA_PREFIX`` — language data folder (default: ``models/tessdata`` when it exists);
* ``SAFENET_OCR_ENGINE`` — ``auto`` (default) | ``tesseract`` | ``easyocr``.

When no engine is available, :class:`OCRUnavailable` explains how to install one.
"""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from safenet import paths

TESSERACT_LANGS = ("rus", "kaz", "eng")
WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
INSTALL_HINT = (
    "Install Tesseract with Russian/Kazakh data (Windows: `winget install UB-Mannheim.TesseractOCR`, "
    "then put rus/kaz .traineddata into models/tessdata; Linux: `apt install tesseract-ocr tesseract-ocr-rus "
    "tesseract-ocr-kaz`) or `pip install easyocr`."
)


class OCRUnavailable(RuntimeError):
    """No OCR engine can be used; the message says why and how to fix it."""


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0..1
    engine: str
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- preprocessing
def load_image(image) -> Image.Image:
    """Accept a path, bytes, a file-like object or a PIL image."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if hasattr(image, "read"):
        return Image.open(image)
    path = Path(image)
    if not path.is_absolute() and not path.exists():
        path = paths.ROOT / path
    return Image.open(path)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    sum_all = float(np.dot(np.arange(256), hist))
    weight_bg = sum_bg = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / weight_fg
        var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def estimate_skew(binary: np.ndarray, max_angle: float = 5.0, step: float = 0.25) -> float:
    """Angle (degrees) that makes text lines horizontal: best row-profile sharpness."""
    img = Image.fromarray((binary * 255).astype(np.uint8))
    if img.width > 600:
        img = img.resize((600, max(1, int(img.height * 600 / img.width))))
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + step / 2, step):
        rotated = np.asarray(img.rotate(float(angle), resample=Image.NEAREST, fillcolor=0)) > 0
        profile = rotated.sum(axis=1).astype(float)
        score = float(np.sum(np.diff(profile) ** 2))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def preprocess_image(image) -> tuple[Image.Image, dict]:
    """Return a clean black-on-white image for OCR and what was done to it."""
    img = load_image(image)
    img = ImageOps.exif_transpose(img).convert("L")
    info: dict = {}
    if img.width < 700:
        factor = 2
        img = img.resize((img.width * factor, img.height * factor), Image.LANCZOS)
        info["upscaled"] = factor
    gray = np.asarray(img)
    if gray.mean() < 110:
        img = ImageOps.invert(img)
        info["inverted"] = True
    img = img.filter(ImageFilter.MedianFilter(3))
    gray = np.asarray(img)
    t = otsu_threshold(gray)
    ink = gray < t  # True = dark text pixel
    full_rows = ink.mean(axis=1) > 0.85
    ink[full_rows] = False  # solid header bars / dividers are not text
    angle = estimate_skew(ink)
    info["skew"] = angle
    if abs(angle) >= 0.25:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=255)
        gray = np.asarray(img)
        t = otsu_threshold(gray)
        ink = gray < t
        ink[ink.mean(axis=1) > 0.85] = False
    clean = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8))
    info["threshold"] = t
    return clean, info


# ---------------------------------------------------------------- engines
def _tesseract_cmd() -> str | None:
    cmd = os.environ.get("TESSERACT_CMD")
    if cmd:
        return cmd if Path(cmd).exists() or shutil.which(cmd) else None
    found = shutil.which("tesseract")
    if found:
        return found
    return str(WINDOWS_TESSERACT) if WINDOWS_TESSERACT.exists() else None


def _tessdata_dir() -> str | None:
    env = os.environ.get("TESSDATA_PREFIX")
    if env:
        return env
    local = paths.MODELS_DIR / "tessdata"
    return str(local) if local.exists() else None


@lru_cache(maxsize=1)
def tesseract_status() -> tuple[bool, str, str]:
    """``(available, languages, reason)``."""
    cmd = _tesseract_cmd()
    if cmd is None:
        return False, "", "Tesseract binary not found"
    try:
        import pytesseract
    except ImportError:
        return False, "", "pytesseract is not installed"
    pytesseract.pytesseract.tesseract_cmd = cmd
    config = f'--tessdata-dir "{_tessdata_dir()}"' if _tessdata_dir() else ""
    try:
        available = set(pytesseract.get_languages(config=config))
    except Exception as exc:  # noqa: BLE001 - any failure means "not usable"
        return False, "", f"Tesseract cannot run: {exc}"
    langs = [lang for lang in TESSERACT_LANGS if lang in available]
    if "rus" not in langs and "kaz" not in langs:
        return False, "", f"Tesseract has no Russian/Kazakh language data (found: {sorted(available)})"
    return True, "+".join(langs), ""


@lru_cache(maxsize=1)
def easyocr_status() -> tuple[bool, str]:
    if importlib.util.find_spec("easyocr") is None:
        return False, "easyocr is not installed"
    return True, ""


def available_engine() -> tuple[str | None, str]:
    """The engine that would be used and, if none, the combined reason."""
    choice = os.environ.get("SAFENET_OCR_ENGINE", "auto").lower()
    t_ok, _, t_reason = tesseract_status()
    e_ok, e_reason = easyocr_status()
    if choice in ("auto", "tesseract") and t_ok:
        return "tesseract", ""
    if choice in ("auto", "easyocr") and e_ok:
        return "easyocr", ""
    return None, f"{t_reason}; {e_reason}. {INSTALL_HINT}"


def _ocr_tesseract(img: Image.Image) -> OCRResult:
    import pytesseract

    _, langs, _ = tesseract_status()
    tessdata = _tessdata_dir()
    config = "--oem 1 --psm 6" + (f' --tessdata-dir "{tessdata}"' if tessdata else "")
    text = pytesseract.image_to_string(img, lang=langs, config=config)
    data = pytesseract.image_to_data(img, lang=langs, config=config, output_type=pytesseract.Output.DICT)
    confs = [float(c) for c in data.get("conf", []) if str(c).strip() not in ("", "-1")]
    confidence = float(np.mean(confs) / 100) if confs else 0.0
    return OCRResult(text=_clean(text), confidence=round(confidence, 3), engine="tesseract", meta={"langs": langs})


@lru_cache(maxsize=1)
def _easyocr_reader():
    import easyocr

    return easyocr.Reader(["ru", "en"], gpu=False, verbose=False)


def _ocr_easyocr(img: Image.Image) -> OCRResult:
    results = _easyocr_reader().readtext(np.asarray(img.convert("RGB")), detail=1, paragraph=False)
    results.sort(key=lambda r: (round(min(p[1] for p in r[0]) / 20), min(p[0] for p in r[0])))
    lines: list[list[str]] = []
    last_y = None
    for box, text, _conf in results:
        y = min(p[1] for p in box)
        if last_y is None or abs(y - last_y) > 20:
            lines.append([])
        lines[-1].append(text)
        last_y = y
    confidence = float(np.mean([r[2] for r in results])) if results else 0.0
    return OCRResult(text=_clean("\n".join(" ".join(line) for line in lines)), confidence=round(confidence, 3),
                     engine="easyocr")


def _clean(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def ocr_image(image, engine: str | None = None) -> OCRResult:
    """Recognise the text of a screenshot; raises :class:`OCRUnavailable` without an engine."""
    if engine is None:
        engine, reason = available_engine()
        if engine is None:
            raise OCRUnavailable(reason)
    clean, info = preprocess_image(image)
    if engine == "tesseract":
        ok, _, reason = tesseract_status()
        if not ok:
            raise OCRUnavailable(f"{reason}. {INSTALL_HINT}")
        result = _ocr_tesseract(clean)
    elif engine == "easyocr":
        ok, reason = easyocr_status()
        if not ok:
            raise OCRUnavailable(f"{reason}. {INSTALL_HINT}")
        result = _ocr_easyocr(clean)
    else:
        raise ValueError(f"unknown OCR engine {engine!r}")
    result.meta.update(info)
    return result
