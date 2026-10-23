"""``make eval``: metrics on test and gold, ablation, plots, error analysis.

Everything written to ``reports/metrics.md`` is deterministic (no timings), so a
re-run produces the same file. Gold is used here for evaluation only.
"""

from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from safenet import LABELS, TRACKS, paths  # noqa: E402
from safenet.eval import metrics as M  # noqa: E402
from safenet.eval.collect import Record, collect  # noqa: E402
from safenet.eval.variants import VARIANT_NAMES, VARIANTS, Scored, choose_threshold, flagged, score_records  # noqa: E402
from safenet.graph.lookup import load_graph  # noqa: E402
from safenet.scoring.config import load_config  # noqa: E402

TARGETS = {"precision": 0.85, "recall": 0.80, "hard_negative_fpr": 0.10}
TRACK_KK = {"lottery": "лотерея", "casino": "казино", "pyramid": "пирамида", "none": "белгілер жоқ"}
REPORT_FILES = ("metrics.md", "metrics.json", "errors.md", "confusion_test.png", "confusion_gold.png",
                "pr_curve.png", "ablation.png")


def _arrays(records: list[Record]) -> tuple[np.ndarray, np.ndarray]:
    return np.array([r.label for r in records]), np.array([r.is_hard_negative for r in records])


def evaluate() -> dict:
    cfg = load_config()
    graph = load_graph()
    data = {
        "val": pd.read_parquet(paths.split_path("val")),
        "test": pd.read_parquet(paths.split_path("test")),
        "gold": pd.read_parquet(paths.GOLD_PATH),
    }
    records = {
        "val": collect(data["val"], graph=None, use_ocr=False),
        "test": collect(data["test"], graph=graph, use_ocr=True),
        "gold": collect(data["gold"], graph=graph, use_ocr=True),
    }

    # thresholds: production (stage 5) for hybrid variants; rules-only / ML-only tuned on val the same way
    thresholds = {v: cfg.medium for v in ("hybrid", "hybrid_ocr", "hybrid_graph")}
    threshold_notes = {v: "жұмыс шегі" for v in thresholds}
    val_labels, val_hn = _arrays(records["val"])
    for variant in ("rules", "ml"):
        chosen = choose_threshold(score_records(records["val"], variant, cfg), val_labels, val_hn, cfg.tuning)
        thresholds[variant] = float(chosen["threshold"])
        threshold_notes[variant] = "" if chosen["constraints_met"] else "шарт орындалмады, max F1"

    results: dict = {"thresholds": thresholds, "threshold_notes": threshold_notes, "splits": {}}
    scored_cache: dict[tuple[str, str], Scored | None] = {}
    for split in ("test", "gold"):
        recs = records[split]
        labels, hn = _arrays(recs)
        split_out = {"variants": {}}
        for variant in VARIANTS:
            scored = score_records(recs, variant, cfg, graph)
            scored_cache[(split, variant)] = scored
            if scored is None:
                split_out["variants"][variant] = None
                continue
            flags = flagged(scored, thresholds[variant])
            pred = M.predicted_class(flags, scored.track)
            split_out["variants"][variant] = {
                "binary": M.binary(labels, hn, flags),
                "tracks": M.per_track(labels, pred),
            }
        hybrid = scored_cache[(split, "hybrid")]
        flags = flagged(hybrid, cfg.medium)
        pred = M.predicted_class(flags, hybrid.track)
        split_out["confusion"] = M.confusion(labels, pred)
        split_out["explanations"] = M.explanation_quality(recs, flags)
        split_out["by_lang"] = M.by_group(recs, flags, lambda r: r.lang)
        split_out["by_obfuscation"] = M.by_group(recs, flags, lambda r: "бар" if r.obfuscated else "жоқ")
        split_out["by_source"] = M.by_group(recs, flags, lambda r: r.source_type)
        split_out["levels"] = {
            v: M.level_agreement(recs, scored_cache[(split, v)].score, scored_cache[(split, v)].has_evidence,
                                 cfg.medium, cfg.high)
            for v in ("hybrid", "hybrid_graph") if scored_cache[(split, v)] is not None
        }
        y = labels != "none"
        score_for_curve = np.where(hybrid.has_evidence, hybrid.score, np.minimum(hybrid.score, cfg.medium - 1))
        split_out["pr_curve"] = precision_recall_curve(y, score_for_curve)
        split_out["threshold_table"] = [
            {"threshold": t, **{k: v for k, v in M.binary(labels, hn, flagged(hybrid, t)).items()
                                if k in ("precision", "recall", "f1", "hard_negative_fpr")}}
            for t in sorted(set(range(10, 75, 5)) | {int(cfg.medium), int(cfg.high)})
        ]
        results["splits"][split] = split_out
    results["records"] = records
    results["scored"] = scored_cache
    results["cfg"] = cfg
    return results


# ------------------------------------------------------------------ writing
def _f(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def _pct(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def _plot_confusion(cm: pd.DataFrame, title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=120)
    ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)), [TRACK_KK[c] for c in cm.columns], rotation=20)
    ax.set_yticks(range(len(cm.index)), [TRACK_KK[c] for c in cm.index])
    ax.set_xlabel("болжам")
    ax.set_ylabel("эталон")
    vmax = cm.values.max() or 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm.values[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > vmax / 2 else "#222222")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def _plot_pr(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=120)
    for split, color in (("test", "#2f6db5"), ("gold", "#d0663a")):
        precision, recall, _ = results["splits"][split]["pr_curve"]
        ax.plot(recall, precision, color=color, label=split)
        b = results["splits"][split]["variants"]["hybrid"]["binary"]
        ax.scatter([b["recall"]], [b["precision"]], color=color, zorder=3)
    ax.axhline(TARGETS["precision"], color="#999999", linestyle="--", linewidth=1)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision–recall (гибрид); нүкте — таңдалған шек")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def _plot_ablation(results: dict, path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=120)
    width = 0.38
    xs = np.arange(len(VARIANTS))
    for k, (split, color) in enumerate((("test", "#2f6db5"), ("gold", "#d0663a"))):
        vals = [(results["splits"][split]["variants"][v] or {}).get("binary", {}).get("f1", 0.0) for v in VARIANTS]
        ax.bar(xs + (k - 0.5) * width, vals, width, label=split, color=color)
    ax.set_xticks(xs, [VARIANT_NAMES[v].replace(" (ережелер + ML)", "") for v in VARIANTS], rotation=12)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 (белгілер бар/жоқ)")
    ax.set_title("Абляция")
    ax.legend(loc="lower center", ncol=2, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)


def _group_table(df: pd.DataFrame, title: str) -> list[str]:
    lines = [f"| {title} | n | precision | recall | F1 | FPR | FPR (қиын негатив) |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines += [f"| {r.group} | {r.n} | {_f(r.precision)} | {_f(r.recall)} | {_f(r.f1)} | {_pct(r.fpr)} | "
              f"{_pct(r.hard_negative_fpr)} |" for r in df.itertuples()]
    return lines


def write_metrics(results: dict) -> dict:
    cfg = results["cfg"]
    test, gold = results["splits"]["test"], results["splits"]["gold"]
    th = results["thresholds"]
    hyb = {s: results["splits"][s]["variants"]["hybrid"]["binary"] for s in ("test", "gold")}
    lines = [
        "# Бағалау: метрикалар",
        "",
        "Барлық сандар бағалау кезінде автоматты түрде есептеледі, қайта есептегенде нәтиже өзгермейді. **test** — "
        "синтетикалық деректердің бөлек бөлігі: оның шаблондары мен фразалары модельді оқытуға да, баптауға да "
        f"қолданылмаған. **gold** — қолмен жазылған {hyb['gold']['n']} хабарлама, олар тек бағалауға қолданылады.",
        "",
        f"«Белгілер бар» — деңгейі «Орташа» немесе одан жоғары (ұпай ≥ {cfg.medium:.0f}); «Жоғары» деңгей — ұпай ≥ "
        f"{cfg.high:.0f}. Шектер оқытуға қолданылмаған бөлек баптау жинағы бойынша таңдалған.",
        "",
        "## Негізгі нәтижелер (гибрид)",
        "",
        "| жинақ | n | precision | recall | F1 | FPR | FPR (қиын негатив) |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {s} | {b['n']} | {_f(b['precision'])} | {_f(b['recall'])} | {_f(b['f1'])} | {_pct(b['fpr'])} | "
          f"{_pct(b['hard_negative_fpr'])} |" for s, b in hyb.items()],
        "",
        "Test жинағындағы мақсатты мәндер: "
        + "; ".join([
            f"precision ≥ {TARGETS['precision']} — {'✅' if hyb['test']['precision'] >= TARGETS['precision'] else '❌'}",
            f"recall ≥ {TARGETS['recall']} — {'✅' if hyb['test']['recall'] >= TARGETS['recall'] else '❌'}",
            f"қиын негативтердегі FPR ≤ {TARGETS['hard_negative_fpr']} — "
            f"{'✅' if hyb['test']['hard_negative_fpr'] <= TARGETS['hard_negative_fpr'] else '❌'}",
        ]) + ".",
        "",
        "## Тректер бойынша (гибрид)",
        "",
        "Болжанатын класс: деңгейі «Орташа» немесе одан жоғары болса — ең ықтимал трек, әйтпесе «белгілер жоқ».",
        "",
        "| трек | test precision | test recall | test F1 | gold precision | gold recall | gold F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *[f"| {t} | {_f(test['variants']['hybrid']['tracks'][t]['precision'])} | "
          f"{_f(test['variants']['hybrid']['tracks'][t]['recall'])} | {_f(test['variants']['hybrid']['tracks'][t]['f1'])} | "
          f"{_f(gold['variants']['hybrid']['tracks'][t]['precision'])} | {_f(gold['variants']['hybrid']['tracks'][t]['recall'])} | "
          f"{_f(gold['variants']['hybrid']['tracks'][t]['f1'])} |" for t in TRACKS],
        f"| macro-F1 | | | {_f(test['variants']['hybrid']['tracks']['macro_f1'])} | | | "
        f"{_f(gold['variants']['hybrid']['tracks']['macro_f1'])} |",
        "",
        "## Абляция",
        "",
        "Әр нұсқаның шегі баптау жинағында бірдей критериймен таңдалды: precision ≥ 0.85 және қиын негативтердегі "
        f"FPR ≤ 10% болғанда recall барынша жоғары. Гибрид нұсқалары жүйенің жұмыс шегін (ұпай ≥ {cfg.medium:.0f}) "
        "қолданады.",
        "",
        "| нұсқа | шек | test P | test R | test F1 | test FPR (қиын) | gold P | gold R | gold F1 | gold FPR (қиын) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        note = results["threshold_notes"][v]
        row = [VARIANT_NAMES[v], f"{th[v]:.0f}" + (f" ({note})" if note else "")]
        for s in ("test", "gold"):
            res = results["splits"][s]["variants"][v]
            if res is None:
                row += ["—"] * 4
            else:
                b = res["binary"]
                row += [_f(b["precision"]), _f(b["recall"]), _f(b["f1"]), _pct(b["hard_negative_fpr"])]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "- «Гибрид + OCR»: test скриншоттарының мәтіні суреттен OCR арқылы оқылады (эталон мәтін қолданылмайды). "
        "Gold-та скриншот жоқ, сондықтан нәтиже гибридпен бірдей.",
        "- «Гибрид + граф»: желі бонусы тек «Орташа» немесе одан жоғары деңгейдегі жазбаларға беріледі, сондықтан "
        "«белгілер бар/жоқ» метрикалары гибридпен бірдей. Бонус «Орташа» мен «Жоғары» деңгейлердің арасындағы "
        "шекараға ғана әсер етеді (төменде).",
        "",
        "![Абляция](ablation.png)",
        "",
        "## Тәуекел деңгейлері (test)",
        "",
        "Болжанған деңгейдің (Төмен / Орташа / Жоғары) эталон деңгейімен сәйкестігі:",
        "",
        "| нұсқа | деңгейлер дәлдігі | деңгейлер macro-F1 |",
        "|---|---:|---:|",
        *[f"| {VARIANT_NAMES[v]} | {_f(m['accuracy'])} | {_f(m['macro_f1'])} |" for v, m in test["levels"].items()],
        "",
        "## Түсіндірмелер сапасы",
        "",
        "Дұрыс белгіленген схемалардың ішінде іске қосылған белгілердің кемінде біреуі эталон белгілермен "
        "сәйкес келетін жағдайлар үлесі:",
        "",
        "| жинақ | жағдайлар | ≥ 1 сәйкестік | орташа Жаккар |",
        "|---|---:|---:|---:|",
        *[f"| {s} | {results['splits'][s]['explanations']['n']} | "
          f"{_pct(results['splits'][s]['explanations']['share_with_overlap'])} | "
          f"{_f(results['splits'][s]['explanations']['mean_jaccard'])} |" for s in ("test", "gold")],
        "",
        "## Тілдер бойынша (гибрид)",
        "",
        "### test",
        "",
        *_group_table(test["by_lang"], "тіл"),
        "",
        "### gold",
        "",
        *_group_table(gold["by_lang"], "тіл"),
        "",
        "## Бүркемелеу бойынша (гибрид, test)",
        "",
        *_group_table(test["by_obfuscation"], "бүркемелеу"),
        "",
        "## Дереккөз түрі бойынша (гибрид, test)",
        "",
        *_group_table(test["by_source"], "дереккөз"),
        "",
        "## Precision–recall қисығы және шектер кестесі",
        "",
        "![PR](pr_curve.png)",
        "",
        "| шек | test precision | test recall | test F1 | test FPR (қиын) | gold precision | gold recall | gold F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gold_table = {r["threshold"]: r for r in gold["threshold_table"]}
    for r in test["threshold_table"]:
        g = gold_table[r["threshold"]]
        mark = " ◀" if r["threshold"] == int(cfg.medium) else ""
        lines.append(f"| {r['threshold']}{mark} | {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['f1'])} | "
                     f"{_pct(r['hard_negative_fpr'])} | {_f(g['precision'])} | {_f(g['recall'])} | {_f(g['f1'])} |")
    lines += [
        "",
        "◀ — жүйенің жұмыс шегі.",
        "",
        "## Қателер матрицасы (гибрид)",
        "",
        "![test](confusion_test.png) ![gold](confusion_gold.png)",
        "",
        "Толығырақ: [қателерді талдау](errors.md).",
        "",
    ]
    (paths.REPORTS_DIR / "metrics.md").write_text("\n".join(lines), encoding="utf-8")

    # 6 digits: values re-formatted elsewhere (dashboard, README) must round like metrics.md
    summary = {
        "thresholds": {k: float(v) for k, v in th.items()},
        "hybrid": {s: {k: round(float(v), 6) for k, v in hyb[s].items()} for s in hyb},
        "ablation_f1": {s: {v: (round(results["splits"][s]["variants"][v]["binary"]["f1"], 6)
                                if results["splits"][s]["variants"][v] else None) for v in VARIANTS}
                        for s in ("test", "gold")},
        "explanations": {s: results["splits"][s]["explanations"] for s in ("test", "gold")},
        "track_macro_f1": {s: round(results["splits"][s]["variants"]["hybrid"]["tracks"]["macro_f1"], 6)
                           for s in ("test", "gold")},
    }
    (paths.REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    _plot_confusion(test["confusion"], "test (гибрид)", paths.REPORTS_DIR / "confusion_test.png")
    _plot_confusion(gold["confusion"], "gold (гибрид)", paths.REPORTS_DIR / "confusion_gold.png")
    _plot_pr(results, paths.REPORTS_DIR / "pr_curve.png")
    _plot_ablation(results, paths.REPORTS_DIR / "ablation.png")
    return summary


# ------------------------------------------------------------- error analysis
def _reason_fp(r: Record, score: float) -> str:
    pos = [h for h in r.hits if not h.negative]
    neg = [h.id for h in r.hits if h.negative]
    core = [h.id for h in pos if h.core]
    top = max(TRACKS, key=lambda t: r.proba.get(t, 0.0))
    parts = []
    if r.is_hard_negative:
        parts.append("қиын негатив")
    if core:
        parts.append(f"негізгі белгі іске қосылды ({', '.join(core)})")
    else:
        parts.append("тек әлсіз белгілер + ML")
    if neg:
        parts.append(f"теріс белгі ({', '.join(neg)}) ұпайды жеткілікті төмендетпеді")
    else:
        parts.append("теріс белгі табылмады (сөздікте ондай контекст жоқ)")
    parts.append(f"ML: {top} {r.proba.get(top, 0.0):.2f}")
    return "; ".join(parts)


def _reason_fn(r: Record, score: float, threshold: float) -> str:
    pos = [h for h in r.hits if not h.negative]
    neg = [h.id for h in r.hits if h.negative]
    top = max(TRACKS, key=lambda t: r.proba.get(t, 0.0))
    if not pos:
        reason = "оң белгі табылмады (сөздікте мұндай тұжырым жоқ)"
    elif neg:
        reason = f"теріс белгі ({', '.join(neg)}) ұпайды төмендетті"
    elif not any(h.core for h in pos):
        reason = f"тек әлсіз белгілер ({', '.join(h.id for h in pos)}), негізгі белгі жоқ"
    else:
        reason = "ұпай шекке жетпеді"
    return f"{reason}; ML: {top} {r.proba.get(top, 0.0):.2f}; ұпай {score:.0f} < {threshold:.0f}"


def write_errors(results: dict) -> None:
    cfg = results["cfg"]
    fps, fns = [], []
    for split in ("gold", "test"):
        recs = results["records"][split]
        scored = results["scored"][(split, "hybrid")]
        flags = flagged(scored, cfg.medium)
        for r, f, s in zip(recs, flags, scored.score):
            if f and r.label == "none":
                fps.append((split, r, s))
            elif not f and r.label != "none":
                fns.append((split, r, s))
    fps.sort(key=lambda x: (x[0] != "gold", -x[2], x[1].id))
    fns.sort(key=lambda x: (x[0] != "gold", x[2], x[1].id))
    lines = [
        "# Қателерді талдау (гибрид)",
        "",
        f"Барлығы: жалған іске қосылулар — {len(fps)}, өткізіп алулар — {len(fns)} (test + gold). Төменде әр "
        "түрден 10-ға дейін мысал берілген (алдымен gold). Себептер іске қосылған белгілер мен модель "
        "ықтималдықтары бойынша автоматты түрде шығарылады.",
        "",
        "## Жалған іске қосылулар (FP)",
        "",
        "| # | жинақ | id | ұпай | мәтін | себебі |",
        "|---:|---|---|---:|---|---|",
    ]
    for i, (split, r, s) in enumerate(fps[:10], 1):
        lines.append(f"| {i} | {split} | {r.id} | {s:.0f} | {r.text[:160].replace(chr(10), ' ⏎ ').replace('|', '/')} | "
                     f"{_reason_fp(r, s)} |")
    lines += ["", "## Өткізіп алулар (FN)", "", "| # | жинақ | id | трек | ұпай | мәтін | себебі |",
              "|---:|---|---|---|---:|---|---|"]
    for i, (split, r, s) in enumerate(fns[:10], 1):
        lines.append(f"| {i} | {split} | {r.id} | {r.label} | {s:.0f} | "
                     f"{r.text[:160].replace(chr(10), ' ⏎ ').replace('|', '/')} | {_reason_fn(r, s, cfg.medium)} |")
    fn_reasons = Counter(_reason_fn(r, s, cfg.medium).split(";")[0].split(" (")[0] for _, r, s in fns)
    fp_kinds = Counter(
        ("қиын негатив" if r.is_hard_negative else "қалыпты контент")
        + (", теріс белгі бар" if any(h.negative for h in r.hits) else ", теріс белгі жоқ")
        for _, r, _ in fps
    )
    fn_screens = sum(r.source_type == "screenshot" for _, r, _ in fns)
    lines += [
        "",
        "## Негізгі қорытындылар (санақ)",
        "",
        f"Өткізіп алулар себебі бойынша (барлығы {len(fns)}, оның ішінде скриншоттар — {fn_screens}):",
        "",
        *[f"- {reason}: {n}" for reason, n in fn_reasons.most_common()],
        "",
        f"Жалған іске қосылулар түрі бойынша (барлығы {len(fps)}):",
        "",
        *[f"- {kind}: {n}" for kind, n in fp_kinds.most_common()],
        "",
        "Жақсарту бағыттары: өткізіп алынған тұжырымдарды (мысалы, «командаға салым», «слот үшін» деген төлем "
        "чектері) аналитиктің кері байланысы (дашбордтағы «Өткізіп жіберу» батырмасы) арқылы сөздікке қосу; "
        "ескерту мен қаржылық сауаттылық мәтіндері үшін теріс белгінің тұжырымдарын кеңейту.",
        "",
    ]
    (paths.REPORTS_DIR / "errors.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> dict:
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = evaluate()
    summary = write_metrics(results)
    write_errors(results)
    return summary
