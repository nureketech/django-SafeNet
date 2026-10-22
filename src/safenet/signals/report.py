"""Per-signal statistics on a split; ``reports/signals_precision.md``.

* precision vs track — share of records where the signal fired that belong to the
  signal's own track (schemes for ``common`` signals, ``none`` for negative ones);
* precision / recall vs ``gold_signals`` — agreement with the signals the generator
  inserted (used for explainability).
"""

from __future__ import annotations

import pandas as pd

from safenet import paths
from safenet.common import to_list
from safenet.signals.detector import SignalDetector

CORE_MIN_PRECISION = 0.7


def signal_stats(df: pd.DataFrame, detector: SignalDetector | None = None) -> pd.DataFrame:
    detector = detector or SignalDetector()
    fired = [{h.id for h in detector.detect(text)} for text in df["text"]]
    rows = []
    for sig in detector.signals:
        has = pd.Series([sig.id in f for f in fired], index=df.index)
        gold = df["gold_signals"].map(lambda g, sid=sig.id: sid in to_list(g))
        if sig.track in ("lottery", "casino", "pyramid"):
            own = df["label"] == sig.track
        elif sig.track == "negative":
            own = df["label"] == "none"
        else:
            own = df["label"] != "none"
        n = int(has.sum())
        rows.append({
            "signal": sig.id,
            "track": sig.track,
            "core": sig.core,
            "fired": n,
            "precision_track": round(float((has & own).sum() / n), 3) if n else None,
            "precision_gold": round(float((has & gold).sum() / n), 3) if n else None,
            "recall_gold": round(float((has & gold).sum() / gold.sum()), 3) if gold.sum() else None,
        })
    return pd.DataFrame(rows)


def _md(stats: pd.DataFrame) -> list[str]:
    lines = [
        "| белгі | трек | core | іске қосылды | precision (трек) | precision (gold_signals) | recall (gold_signals) |",
        "|---|---|:-:|---:|---:|---:|---:|",
    ]
    for r in stats.itertuples():
        fmt = lambda v: "—" if v is None or pd.isna(v) else f"{v:.2f}"  # noqa: E731
        lines.append(
            f"| {r.signal} | {r.track} | {'✔' if r.core else ''} | {r.fired} | {fmt(r.precision_track)} | "
            f"{fmt(r.precision_gold)} | {fmt(r.recall_gold)} |"
        )
    return lines


def write_report() -> dict[str, pd.DataFrame]:
    detector = SignalDetector()
    out = {}
    lines = [
        "# Белгілердің дәлдігі (precision)",
        "",
        "`python -m safenet.signals` арқылы есептелген. Белгілер сөздігі тек **train** бойынша бапталды. **val** "
        "жинағы жүйелі қателерді табу үшін қаралды. **test** пен **gold** бұл кезеңде қолданылған жоқ.",
        "",
        f"Талап: әр core-белгінің өз трегіне қатысты precision-ы train жинағында ≥ {CORE_MIN_PRECISION}.",
        "",
    ]
    for split in ("train", "val"):
        df = pd.read_parquet(paths.split_path(split))
        stats = signal_stats(df, detector)
        out[split] = stats
        lines += [f"## {split} ({len(df)} жазба)", "", *_md(stats), ""]
    core = out["train"][out["train"]["core"]]
    lines += [
        "## Core-белгілер (train)",
        "",
        *[f"- {r.signal}: precision = {r.precision_track:.2f} "
          f"{'✅' if r.precision_track >= CORE_MIN_PRECISION else '❌'}" for r in core.itertuples()],
        "",
    ]
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.REPORTS_DIR / "signals_precision.md").write_text("\n".join(lines), encoding="utf-8")
    return out
