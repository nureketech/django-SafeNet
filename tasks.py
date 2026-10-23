"""Cross-platform task runner used by both ``Makefile`` and ``make.bat``.

Usage::

    python tasks.py <target> [<target> ...] [N=<n>] [PORT=<port>]

Several targets run in order and stop at the first failure, like ``make install data train eval app``.
Targets: install, tessdata, data, train, eval, bench, docs, api, app, demo, test, test-stage.
"""

from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(*args: str) -> int:
    """Run a command from the project root and return its exit code."""
    print("+", " ".join(args), flush=True)
    return subprocess.call(list(args), cwd=ROOT)


def run_all(*commands: tuple[str, ...]) -> int:
    """Run commands sequentially, stopping on the first failure."""
    for command in commands:
        code = run(*command)
        if code != 0:
            return code
    return 0


def target_install(_: dict[str, str]) -> int:
    return run_all(
        (PY, "-m", "pip", "install", "-r", "requirements.txt"),
        (PY, "-m", "pip", "install", "-e", "."),
    )


TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang}.traineddata"


def target_tessdata(_: dict[str, str]) -> int:
    """Download Russian/Kazakh Tesseract data into models/tessdata (eng/osd are copied from the install)."""
    import shutil
    import urllib.request

    target = ROOT / "models" / "tessdata"
    target.mkdir(parents=True, exist_ok=True)
    for lang in ("rus", "kaz"):
        dest = target / f"{lang}.traineddata"
        if not dest.exists():
            print(f"downloading {lang}.traineddata", flush=True)
            urllib.request.urlretrieve(TESSDATA_URL.format(lang=lang), dest)
    installed = shutil.which("tesseract")
    candidates = [Path(installed).parent / "tessdata"] if installed else []
    candidates.append(Path(r"C:\Program Files\Tesseract-OCR\tessdata"))
    for folder in candidates:
        for name in ("eng.traineddata", "osd.traineddata"):
            if (folder / name).exists() and not (target / name).exists():
                shutil.copy(folder / name, target / name)
    print("tessdata:", sorted(p.name for p in target.glob("*.traineddata")))
    return 0


def target_data(_: dict[str, str]) -> int:
    return run(PY, "-m", "safenet.generator")


def target_train(params: dict[str, str]) -> int:
    extra = ["--embeddings"] if params.get("EMBEDDINGS") == "1" else []
    return run_all(
        (PY, "-m", "safenet.ml.train", *extra),
        (PY, "-m", "safenet.signals"),
        (PY, "-m", "safenet.scoring.tune"),
        (PY, "-m", "safenet.graph", "--rescore"),
    )


def target_eval(_: dict[str, str]) -> int:
    return run(PY, "-m", "safenet.eval")


def target_bench(_: dict[str, str]) -> int:
    """Timings and cascade shares for docs/ARCHITECTURE.md (reports/benchmark.md)."""
    return run(PY, "-m", "safenet.eval.bench")


def target_docs(_: dict[str, str]) -> int:
    """Refresh the generated tables in README.md and docs/*.md."""
    return run(PY, "-m", "safenet.docs")


def _serve(port: str, open_browser: bool, load_cases: bool = True, page: str = "/check/") -> int:
    steps = [(PY, "manage.py", "migrate", "--noinput", "-v", "0")]
    if load_cases:
        steps.append((PY, "manage.py", "load_cases", "--if-empty"))
    code = run_all(*steps)
    if code != 0:
        return code
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}{page}")
    return run(PY, "manage.py", "runserver", f"127.0.0.1:{port}")


def target_api(params: dict[str, str]) -> int:
    """API only: http://127.0.0.1:8000/docs (OpenAPI), /analyze, /analyze/image, /batch, /health."""
    return _serve(params.get("PORT", "8000"), open_browser=params.get("OPEN", "0") == "1", load_cases=False,
                  page="/docs")


def target_app(params: dict[str, str]) -> int:
    return _serve(params.get("PORT", "8000"), open_browser=params.get("OPEN", "1") == "1")


def target_demo(params: dict[str, str]) -> int:
    """Queue (test split, if empty) + the 5 showcase cases; opens the queue filtered to them."""
    code = run_all(
        (PY, "manage.py", "migrate", "--noinput", "-v", "0"),
        (PY, "manage.py", "load_cases", "--if-empty"),
        (PY, "manage.py", "load_demo"),
    )
    if code != 0:
        return code
    return _serve(params.get("PORT", "8000"), open_browser=params.get("OPEN", "1") == "1", load_cases=False,
                  page="/?origin=demo")


def target_test(_: dict[str, str]) -> int:
    return run(PY, "-m", "pytest", "-q")


def target_test_stage(params: dict[str, str]) -> int:
    stage = int(params.get("N", "0"))
    files = sorted((ROOT / "tests").glob(f"test_stage{stage:02d}_*.py"))
    if not files:
        print(f"No tests found for stage {stage}", file=sys.stderr)
        return 2
    return run(PY, "-m", "pytest", "-q", *[str(f.relative_to(ROOT)) for f in files])


TARGETS = {
    "install": target_install,
    "tessdata": target_tessdata,
    "data": target_data,
    "train": target_train,
    "eval": target_eval,
    "bench": target_bench,
    "docs": target_docs,
    "api": target_api,
    "app": target_app,
    "demo": target_demo,
    "test": target_test,
    "test-stage": target_test_stage,
}


def parse_params(args: list[str]) -> dict[str, str]:
    """Parse ``KEY=value`` pairs (also tolerates ``KEY value`` and a bare stage number)."""
    params: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if "=" in arg:
            key, value = arg.split("=", 1)
            params[key.upper()] = value
        elif arg.isdigit():
            params["N"] = arg
        elif i + 1 < len(args) and arg.isupper():
            params[arg] = args[i + 1]
            i += 1
        i += 1
    return params


def main(argv: list[str]) -> int:
    names = []
    while argv and argv[0] in TARGETS:
        names.append(argv.pop(0))
    if not names:
        print(__doc__)
        return 1
    os.environ.setdefault("PYTHONUTF8", "1")
    params = parse_params(argv)
    for name in names:
        code = TARGETS[name](params)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
