"""``python -m safenet.eval`` (``make eval``) — write reports/metrics.md, errors.md, plots."""

import json

from safenet.eval.report import run

if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
