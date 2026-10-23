"""Prepared showcase cases for ``make demo`` (TZ, stage 11).

The cases live in ``config/demo_cases.yaml``. The payment screenshot is rendered
from its ``image_text`` (``python -m safenet.demo`` re-creates it) and read back by
OCR during the demo, exactly like an uploaded screenshot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

from safenet import paths
from safenet.scoring.models import AnalysisResult

DEMO_CONFIG = paths.CONFIG_DIR / "demo_cases.yaml"
DEMO_DIR = paths.DATA_DIR / "demo"


@dataclass
class DemoCase:
    id: str
    title_kk: str
    title_ru: str
    source_type: str
    platform: str
    expected_level: str
    expected_track: str | None = None
    expected_level_without_ocr: str | None = None
    text: str = ""
    account_id: str = ""
    network: str = ""
    image: str = ""
    image_text: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def image_path(self) -> Path | None:
        return paths.ROOT / self.image if self.image else None

    def expected(self, ocr_available: bool) -> str:
        if self.image and not ocr_available and self.expected_level_without_ocr:
            return self.expected_level_without_ocr
        return self.expected_level


def load_cases(path: Path = DEMO_CONFIG) -> list[DemoCase]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]
    known = set(DemoCase.__dataclass_fields__)
    return [DemoCase(**{k: v for k, v in c.items() if k in known},
                     extra={k: v for k, v in c.items() if k not in known}) for c in raw]


def render_receipt(text: str, path: Path) -> Path:
    """A clean phone-like payment screen: thin top stripe, large amount, one field per line."""
    from safenet.generator.images import font_path

    lines = text.split("\n")
    title_font = ImageFont.truetype(font_path(), 34)
    amount_font = ImageFont.truetype(font_path(), 58)
    body_font = ImageFont.truetype(font_path(), 30)
    fonts = [title_font, amount_font] + [body_font] * (len(lines) - 2)
    pad, gap = 48, 30
    width = max(720, max(int(f.getlength(line)) for f, line in zip(fonts, lines)) + 2 * pad)
    height = 24 + pad + sum(f.size + gap for f in fonts) + pad
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 24], fill=(214, 48, 49))
    y = 24 + pad
    for font, line in zip(fonts, lines):
        color = (20, 20, 20) if font is not body_font else (60, 60, 60)
        draw.text((pad, y), line, font=font, fill=color)
        y += font.size + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG", optimize=True)
    return path


def ensure_images(cases: list[DemoCase], force: bool = False) -> list[Path]:
    """Render the screenshots of the demo cases that are missing (or all with ``force``)."""
    made = []
    for case in cases:
        if case.image_path and case.image_text and (force or not case.image_path.exists()):
            made.append(render_receipt(case.image_text, case.image_path))
    return made


def network_account(track: str) -> str | None:
    """First account of the largest planted cluster whose dominant track is ``track``."""
    from safenet.graph.lookup import load_graph

    model = load_graph()
    if model is None:
        return None
    clusters = [c for c in model.clusters.values() if c.top_track == track and c.has_bonus_entity]
    if not clusters:
        return None
    best = max(clusters, key=lambda c: (len(c.accounts), c.risk_share, c.cluster_id))
    return sorted(best.accounts)[0]


def analyse(cases: list[DemoCase] | None = None) -> list[tuple[DemoCase, AnalysisResult]]:
    """Run the full pipeline on every demo case (screenshots through OCR when an engine exists)."""
    from safenet.multimodal import ocr as ocr_module
    from safenet.pipeline import analyze

    cases = load_cases() if cases is None else cases
    ensure_images(cases)
    ocr_ok = ocr_module.available_engine()[0] is not None
    out = []
    for case in cases:
        account = (network_account(case.network) if case.network else None) or case.account_id or f"acc_{case.id}"
        item = {"id": case.id, "text": case.text, "source_type": case.source_type, "platform": case.platform,
                "account_id": account}
        if case.image_path and ocr_ok:
            result = analyze(item, image=case.image_path)
        elif case.image_path:  # no OCR engine: the rendered text stands in for the recognised one
            result = analyze({**item, "text": "\n".join(filter(None, [case.text, case.image_text]))})
            result.warnings.append("OCR қозғалтқышы жоқ: скриншот мәтіні дайын мәтіннен алынды")
        else:
            result = analyze(item)
        out.append((case, result))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render demo screenshots and print the demo cases' results")
    parser.add_argument("--render", action="store_true", help="re-render the screenshots even if they exist")
    args = parser.parse_args()
    from safenet.multimodal import ocr as ocr_module

    cases = load_cases()
    for path in ensure_images(cases, force=args.render):
        print("rendered", path.relative_to(paths.ROOT))
    ocr_ok = ocr_module.available_engine()[0] is not None
    for case, result in analyse(cases):
        expected = case.expected(ocr_ok)
        print(f"{case.id:18} {result.risk_score:3d} {result.risk_level:6} {result.primary_track:8} "
              f"expected {expected:6} {'ok' if result.risk_level == expected else 'MISMATCH'}")


if __name__ == "__main__":
    main()
