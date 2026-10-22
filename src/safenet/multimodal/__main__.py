"""``python -m safenet.multimodal`` — write ``reports/ocr_report.md``."""

from safenet.multimodal.benchmark import write_report

if __name__ == "__main__":
    summary = write_report()
    print(summary)
