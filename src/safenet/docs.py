"""Keep the generated parts of the documentation in sync with configs and reports.

Blocks between ``<!-- auto:NAME -->`` and ``<!-- /auto:NAME -->`` are rewritten from
``config/signals.yaml``, ``config/scoring.yaml``, ``models/thresholds.json``,
``reports/metrics.json`` and ``reports/benchmark.json``::

    python -m safenet.docs           # rewrite the blocks
    python -m safenet.docs --check   # exit 1 when a block is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from safenet import paths

TRACK_KK = {"lottery": "ақылы ұтыс", "casino": "казино", "pyramid": "пирамида", "common": "ортақ",
            "negative": "төмендететін"}
DOC_BLOCKS = {
    "README.md": ("metrics",),
    "docs/PITCH.md": ("metrics",),
    "docs/METHODOLOGY.md": ("signals", "combos", "scoring", "thresholds"),
    "docs/ARCHITECTURE.md": ("bench",),
}
BLOCK_RE = re.compile(r"(<!-- auto:(?P<name>[\w-]+) -->\n)(?P<body>.*?)(<!-- /auto:(?P=name) -->)", re.S)


def _f(x: float) -> str:
    return f"{x:.3f}"


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ blocks
def block_signals() -> str:
    signals = yaml.safe_load(paths.SIGNALS_CONFIG.read_text(encoding="utf-8"))["signals"]
    rows = ["| id | трек | атауы | салмақ | негізгі | қалай анықталады (signals.yaml) |", "|---|---|---|---:|:-:|---|"]
    for s in signals:
        weight = f"−{s['weight']}" if s["track"] == "negative" else str(s["weight"])
        how = s["description"].replace("|", "/")
        if s.get("computed"):
            how += f" *(есептелетін: `{s['computed']}`)*"
        rows.append(f"| `{s['id']}` | {TRACK_KK[s['track']]} | {s['name_kk']} | {weight} | "
                    f"{'★' if s['core'] else ''} | {how} |")
    n_neg = sum(s["track"] == "negative" for s in signals)
    n_core = sum(bool(s["core"]) for s in signals)
    return "\n".join([f"Барлығы **{len(signals)}** белгі: {n_core} негізгі (★), {len(signals) - n_core - n_neg} "
                      f"қосымша, {n_neg} төмендететін.", "", *rows])


def _scoring() -> dict:
    return yaml.safe_load(paths.SCORING_CONFIG.read_text(encoding="utf-8"))


def block_combos() -> str:
    rows = ["| id | трек | шарт | бонус |", "|---|---|---|---:|"]
    for c in _scoring()["combos"]:
        cond = " + ".join(f"`{s}`" for s in c["all"])
        if c.get("any"):
            cond += " + біреуі: " + " / ".join(f"`{s}`" for s in c["any"])
        rows.append(f"| `{c['id']}` | {TRACK_KK[c['track']]} | {cond} | +{c['bonus']} |")
    return "\n".join(rows)


def block_scoring() -> str:
    cfg = _scoring()
    return "\n".join([
        "| параметр | мәні | мағынасы |",
        "|---|---:|---|",
        f"| `rule_cap` | {cfg['rule_cap']} | ережелер бөлігінің жоғарғы шегі |",
        f"| `ml_weight` | {cfg['ml_weight']} | ML ықтималдығының салмағы |",
        f"| `network_max` | {cfg['network_max']} | желі бонусының жоғарғы шегі |",
        f"| `no_core_cap` | {cfg['no_core_cap']} | негізгі белгі болмаса ұпайдың жоғарғы шегі |",
        f"| `require_evidence_for_medium` | {str(cfg['require_evidence_for_medium']).lower()} | "
        "дәлелсіз жазба `medium`-ға жетпейді |",
    ])


def block_thresholds() -> str:
    th = _json(paths.THRESHOLDS_PATH)
    tuning = _scoring()["tuning"]
    val = th["val"]
    return "\n".join([
        f"- **medium = {th['medium']:.0f}**, **high = {th['high']:.0f}** (таңдалған жинақ: `{th['source']}`).",
        f"- medium шарттары: precision ≥ {tuning['min_precision']}, қиын негативтердегі FPR ≤ "
        f"{tuning['max_hard_negative_fpr']}; осы шартта recall барынша жоғары. Іздеу аралығы "
        f"{tuning['medium_range'][0]}–{tuning['medium_range'][1]}.",
        f"- high шарттары: `no_core_cap` ({_scoring()['no_core_cap']})-тен жоғары, «high» precision ≥ "
        f"{tuning['min_high_precision']}, эталон деңгейлермен келісім барынша жоғары. Іздеу аралығы "
        f"{tuning['high_range'][0]}–{tuning['high_range'][1]}.",
        f"- val-дағы нәтиже: precision {_f(val['precision'])}, recall {_f(val['recall'])}, "
        f"қиын негативтердегі FPR {_pct(val['hard_negative_fpr'])}.",
    ])


def block_metrics() -> str:
    hyb = _json(paths.REPORTS_DIR / "metrics.json")["hybrid"]
    rows = ["| жинақ | n | precision | recall | F1 | FPR | FPR (қиын негатив) |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for split in ("test", "gold"):
        b = hyb[split]
        rows.append(f"| {split} | {int(b['n'])} | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | "
                    f"{_pct(b['fpr'])} | {_pct(b['hard_negative_fpr'])} |")
    return "\n".join(rows + ["", "Дереккөз: [`reports/metrics.md`](reports/metrics.md) (`make eval`)."])


def block_bench() -> str:
    path = paths.REPORTS_DIR / "benchmark.json"
    if not path.exists():
        return "_`python -m safenet.eval.bench` әлі іске қосылмаған._"
    b = _json(path)
    t, c, cap = b["timings"], b["cascade"], b["capacity"]
    ocr = t["ocr"]["seconds_per_image"]
    cores = cap["cores"]
    return "\n".join([
        "| өлшем | мәні |",
        "|---|---:|",
        f"| ережелер (нормализация + белгілер), мс/жазба | {t['rules_ms']:.2f} |",
        f"| ML (батч), мс/жазба | {t['ml_ms']:.2f} |",
        f"| толық талдау (OCR-сыз), мс/жазба | {t['full_ms']:.2f} |",
        f"| OCR, мс/скриншот | {ocr * 1000:.0f} |" if ocr else "| OCR | өлшенбеді |",
        f"| каскад: схемалардың келесі сатыға жіберілгені | {_pct(c['scheme_recall'])} |",
        f"| каскад: қарапайым `none` сүзілгені | {_pct(c['plain_benign_dropped'])} |",
        f"| каскад: барлық `none` сүзілгені (қиын негативтермен) | {_pct(c['benign_dropped'])} |",
        f"| мысал: шыңдағы ағын, жазба/с | {cap['peak_items_per_s']:.0f} |",
        f"| мысал: ML-ға жіберілетін үлес | {_pct(cap['forwarded_share'])} |",
        f"| мысал: CPU ядролары — ережелер / ML / OCR | {cores['rules']:.1f} / {cores['ml']:.1f} / "
        f"{cores['ocr']:.1f} |" if cores["ocr"] is not None else
        f"| мысал: CPU ядролары — ережелер / ML | {cores['rules']:.1f} / {cores['ml']:.1f} |",
        "",
        f"Машина: {b['machine']['system']}, {b['machine']['cpu_count']} логикалық ядро, бір ағын. "
        "Толығырақ: [`reports/benchmark.md`](../reports/benchmark.md).",
    ])


BLOCKS = {"signals": block_signals, "combos": block_combos, "scoring": block_scoring,
          "thresholds": block_thresholds, "metrics": block_metrics, "bench": block_bench}


def render(text: str, relative_to_docs: bool = False) -> str:
    """Return ``text`` with every auto block regenerated."""

    def repl(m: re.Match) -> str:
        body = BLOCKS[m.group("name")]()
        if relative_to_docs:
            body = body.replace("](reports/", "](../reports/")
        return f"{m.group(1)}{body}\n{m.group(4)}"

    return BLOCK_RE.sub(repl, text)


def sync(check: bool = False) -> list[str]:
    """Rewrite (or, with ``check``, only compare) the documents; return the ones out of date."""
    stale = []
    for name in DOC_BLOCKS:
        path = paths.ROOT / name
        if not path.exists():
            stale.append(name)
            continue
        text = path.read_text(encoding="utf-8")
        new = render(text, relative_to_docs=name.startswith("docs/"))
        if new != text:
            stale.append(name)
            if not check:
                path.write_text(new, encoding="utf-8")
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = sync(check=args.check)
    if args.check:
        print("out of date: " + ", ".join(stale) if stale else "docs are up to date")
        sys.exit(1 if stale else 0)
    print("updated: " + ", ".join(stale) if stale else "docs are up to date")


if __name__ == "__main__":
    main()
