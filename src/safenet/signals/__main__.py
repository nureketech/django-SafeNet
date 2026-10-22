"""``python -m safenet.signals`` — write ``reports/signals_precision.md``."""

from safenet.signals.report import write_report

if __name__ == "__main__":
    stats = write_report()
    print(stats["train"].to_string(index=False))
