"""Throughput measurements for the scaling plan (docs/ARCHITECTURE.md).

``python -m safenet.eval.bench`` times every pipeline stage on the test split and
measures how much of the stream a cheap rules-only first step would drop. It writes
``reports/benchmark.md`` and ``reports/benchmark.json``. Timings depend on the
machine; the shares are deterministic.
"""

from __future__ import annotations

import json
import os
import platform
import time

import pandas as pd

from safenet import paths
from safenet.signals import detect

OCR_SAMPLE = 10
# Assumptions of the capacity example (not measurements; stated in the report).
STREAM_PER_DAY = 10_000_000
PEAK_FACTOR = 3.0
SCHEME_SHARE = 0.02
IMAGE_SHARE = 0.10


def _timed(fn, repeat: int = 3) -> float:
    """Best wall time of ``repeat`` runs, in seconds."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def forwards(text: str) -> bool:
    """Cheap first step of the cascade: any positive signal fires (regex only, no ML/OCR)."""
    return any(not h.negative for h in detect(text))


def cascade_shares(df: pd.DataFrame) -> dict:
    fwd = df.text.fillna("").map(forwards)
    schemes = df.label != "none"
    benign = ~schemes
    hard = benign & df.is_hard_negative.astype(bool)
    return {
        "n": int(len(df)),
        "forwarded_share": float(fwd.mean()),
        "scheme_recall": float(fwd[schemes].mean()),
        "benign_dropped": float((~fwd[benign]).mean()),
        "plain_benign_dropped": float((~fwd[benign & ~hard]).mean()),
        "hard_negative_dropped": float((~fwd[hard]).mean()),
        "n_schemes": int(schemes.sum()),
        "n_benign": int(benign.sum()),
        "n_hard_negatives": int(hard.sum()),
    }


def measure() -> dict:
    from safenet.ml import load_model, predict_proba
    from safenet.multimodal import ocr as ocr_module
    from safenet.pipeline import analyze_many
    from safenet.preprocess import normalize

    df = pd.read_parquet(paths.split_path("test"))
    texts_df = df[df.source_type != "screenshot"]
    texts = texts_df.text.tolist()
    items = [{"id": r.id, "text": r.text, "source_type": r.source_type, "platform": r.platform,
              "account_id": r.account_id} for r in texts_df.itertuples()]
    model = load_model()
    analyze_many(items[:20], use_ocr=False)  # warm-up: model, graph and regex caches

    n = len(texts)
    t_rules = _timed(lambda: [detect(normalize(t)) for t in texts])
    t_ml = _timed(lambda: predict_proba(texts, model=model))
    t_full = _timed(lambda: analyze_many(items, use_ocr=False))

    ocr = {"engine": None, "seconds_per_image": None, "n_images": 0}
    engine = ocr_module.available_engine()[0]
    shots = df[df.source_type == "screenshot"].head(OCR_SAMPLE)
    if engine and len(shots):
        images = [paths.ROOT / p for p in shots.image_path]
        t_ocr = _timed(lambda: [ocr_module.ocr_image(p) for p in images], repeat=1)
        ocr = {"engine": engine, "seconds_per_image": t_ocr / len(images), "n_images": len(images)}

    cascade = cascade_shares(df)
    timings = {
        "n_texts": n,
        "rules_ms": t_rules / n * 1000,
        "ml_ms": t_ml / n * 1000,
        "full_ms": t_full / n * 1000,
        "ocr": ocr,
    }
    return {
        "machine": {"python": platform.python_version(), "system": f"{platform.system()} {platform.release()}",
                    "processor": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()},
        "timings": timings,
        "cascade": cascade,
        "capacity": capacity(timings, cascade),
    }


def capacity(timings: dict, cascade: dict) -> dict:
    """CPU cores needed for an example stream (single-threaded workers, no batching gains assumed)."""
    rate = STREAM_PER_DAY / 86400 * PEAK_FACTOR
    forwarded = SCHEME_SHARE * cascade["scheme_recall"] + (1 - SCHEME_SHARE) * (1 - cascade["benign_dropped"])
    ocr_s = timings["ocr"]["seconds_per_image"]
    cores = {
        "rules": rate * timings["rules_ms"] / 1000,
        "ml": rate * forwarded * timings["ml_ms"] / 1000,
        "ocr": rate * IMAGE_SHARE * ocr_s if ocr_s else None,
    }
    cores["without_cascade"] = rate * timings["full_ms"] / 1000
    return {
        "assumptions": {"stream_per_day": STREAM_PER_DAY, "peak_factor": PEAK_FACTOR,
                        "scheme_share": SCHEME_SHARE, "image_share": IMAGE_SHARE},
        "peak_items_per_s": rate,
        "forwarded_share": forwarded,
        "cores": cores,
    }


def render(result: dict) -> str:
    t, c, cap = result["timings"], result["cascade"], result["capacity"]
    m = result["machine"]
    ocr = t["ocr"]
    a = cap["assumptions"]
    cores = cap["cores"]
    lines = [
        "# Өнімділік және масштабтау бағасы",
        "",
        "`python -m safenet.eval.bench` арқылы жасалды. Уақыттар машинаға тәуелді, үлестер — тұрақты.",
        f"Машина: {m['system']}, {m['processor']}, {m['cpu_count']} логикалық ядро, Python {m['python']}; "
        "бір процесс, бір ағын.",
        "",
        f"## Бір жазбаның өңделу уақыты (test, {t['n_texts']} мәтін, 3 өлшеудің ең жақсысы)",
        "",
        "| кезең | мс / жазба |",
        "|---|---:|",
        f"| ережелер: нормализация + белгілер (regex) | {t['rules_ms']:.2f} |",
        f"| ML: TF-IDF + белгі-фичалар + LogReg (батч) | {t['ml_ms']:.2f} |",
        f"| толық талдау: ережелер + ML + скоринг + граф | {t['full_ms']:.2f} |",
    ]
    if ocr["seconds_per_image"]:
        lines.append(f"| OCR ({ocr['engine']}, {ocr['n_images']} скриншот) | {ocr['seconds_per_image'] * 1000:.0f} |")
    else:
        lines.append("| OCR | қозғалтқыш жоқ — өлшенбеді |")
    lines += [
        "",
        "## Каскадтың бірінші сатысы: «кемінде бір оң белгі»",
        "",
        "Арзан қадам тек regex-белгілерді тексереді. Ешбір оң белгі іске қоспаған жазба ML мен OCR-ға жіберілмейді.",
        "",
        "| көрсеткіш | мәні |",
        "|---|---:|",
        f"| test жазбалары | {c['n']} |",
        f"| келесі сатыға жіберілгені (test) | {c['forwarded_share'] * 100:.1f}% |",
        f"| схемалардың жіберілгені (каскад recall, {c['n_schemes']} схема) | {c['scheme_recall'] * 100:.1f}% |",
        f"| `none` жазбаларының сүзілгені ({c['n_benign']}) | {c['benign_dropped'] * 100:.1f}% |",
        f"| — қарапайым `none` | {c['plain_benign_dropped'] * 100:.1f}% |",
        f"| — қиын негативтер ({c['n_hard_negatives']}) | {c['hard_negative_dropped'] * 100:.1f}% |",
        "",
        "test-те схемалар үлесі жоғары, ал қиын негативтер әдейі схемаға ұқсатып жазылған. Сондықтан нақты ағында "
        "сүзілетін үлес жоғарыдағы `none` бағасынан төмен болмайды деп күтеміз.",
        "",
        "## Сыйымдылық мысалы",
        "",
        f"Болжамдар (өлшем емес): тәулігіне {a['stream_per_day']:,} жазба".replace(",", " ")
        + f", шың = орташадан ×{a['peak_factor']:.0f}, схемалар үлесі {a['scheme_share'] * 100:.0f}%, "
        f"скриншоттар үлесі {a['image_share'] * 100:.0f}%.",
        "",
        f"- Шыңдағы ағын: **{cap['peak_items_per_s']:.0f} жазба/с**; каскадтан кейін ML-ға "
        f"**{cap['forwarded_share'] * 100:.1f}%** жіберіледі.",
        "",
        "| саты | қажет CPU ядролары (шыңда) |",
        "|---|---:|",
        f"| ережелер (барлық ағын) | {cores['rules']:.1f} |",
        f"| ML (тек жіберілгені) | {cores['ml']:.1f} |",
        f"| OCR (барлық скриншот) | {cores['ocr']:.1f} |" if cores["ocr"] is not None else "| OCR | — |",
        f"| салыстыру: каскадсыз толық талдау | {cores['without_cascade']:.1f} |",
        "",
        "Ядролар саны = шыңдағы ағын × бір жазбаның уақыты. Бұл — бір ағынды Python өлшемі бойынша "
        "консервативті баға: батчтау мен векторлау оны азайтады.",
        "",
    ]
    return "\n".join(lines)


def run() -> dict:
    result = measure()
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
    (paths.REPORTS_DIR / "benchmark.md").write_text(render(result), encoding="utf-8")
    return result


if __name__ == "__main__":
    out = run()
    print(json.dumps({"timings": out["timings"], "cascade": out["cascade"]}, ensure_ascii=False, indent=2))
