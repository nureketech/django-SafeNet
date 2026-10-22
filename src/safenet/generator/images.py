"""Render synthetic screenshots with Pillow (text in the image == record ``text``)."""

from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

THEMES = {
    "light": {"bg": (250, 250, 250), "fg": (25, 25, 25), "header": [(220, 60, 60), (40, 120, 200), (90, 160, 90)]},
    "dark": {"bg": (28, 30, 36), "fg": (236, 236, 236), "header": [(70, 70, 90), (120, 40, 40), (40, 90, 120)]},
    "chat": {"bg": (226, 236, 246), "fg": (20, 20, 20), "header": [(60, 140, 200), (30, 160, 120)]},
}


@lru_cache(maxsize=1)
def font_path() -> str:
    """DejaVuSans (shipped with matplotlib) covers Cyrillic, Kazakh letters and ₸."""
    import matplotlib

    candidates = [
        Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError("No Cyrillic-capable TTF font found (install DejaVuSans)")


def render_screenshot(text: str, path: Path, rng: random.Random) -> dict:
    """Draw ``text`` as a phone-like screenshot and save it as PNG.

    Returns the rendering parameters (theme, noise, rotation) for the report.
    """
    size = rng.choice([26, 28, 30, 32])
    font = ImageFont.truetype(font_path(), size=size)
    lines = text.split("\n")
    pad = 36
    line_h = int(size * 1.65)
    header_h = 56
    width = int(max(font.getlength(line) for line in lines)) + 2 * pad + 20
    width = max(width, 560)
    height = header_h + pad * 2 + line_h * len(lines)
    theme_name = rng.choice(["light", "light", "dark", "chat"])
    theme = THEMES[theme_name]
    img = Image.new("RGB", (width, height), theme["bg"])
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, header_h], fill=rng.choice(theme["header"]))
    y = header_h + pad
    for line in lines:
        if theme_name == "chat" and line.strip():
            w = int(font.getlength(line))
            draw.rounded_rectangle([pad - 12, y - 6, pad + w + 12, y + size + 10], radius=12, fill=(255, 255, 255))
        draw.text((pad, y), line, font=font, fill=theme["fg"])
        y += line_h

    noise = rng.random() < 0.4
    angle = rng.uniform(-2.5, 2.5) if rng.random() < 0.35 else 0.0
    blur = rng.random() < 0.2
    if noise:
        arr = np.asarray(img).astype(np.int16)
        gen = np.random.default_rng(rng.randrange(2**32))
        arr += gen.normal(0, 12, arr.shape).astype(np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if angle:
        img = img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=theme["bg"])
    if blur:
        img = img.filter(ImageFilter.GaussianBlur(0.6))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)
    return {"theme": theme_name, "noise": noise, "angle": round(angle, 2), "blur": blur}
