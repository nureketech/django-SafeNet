"""Cross-platform task runner used by both ``Makefile`` and ``make.bat``.

Usage::

    python tasks.py <target> [N=<n>]

Targets: install, data, train, eval, api, app, test, test-stage, demo.
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


def target_data(_: dict[str, str]) -> int:
    return run(PY, "-m", "safenet.generator")


def target_train(params: dict[str, str]) -> int:
    extra = ["--embeddings"] if params.get("EMBEDDINGS") == "1" else []
    return run_all(
        (PY, "-m", "safenet.ml.train", *extra),
        (PY, "-m", "safenet.signals"),
        (PY, "-m", "safenet.scoring.tune"),
    )


def target_eval(_: dict[str, str]) -> int:
    return run(PY, "-m", "safenet.eval")


def _serve(port: str, open_browser: bool) -> int:
    code = run_all(
        (PY, "manage.py", "migrate", "--noinput", "-v", "0"),
        (PY, "manage.py", "load_cases", "--if-empty"),
    )
    if code != 0:
        return code
    if open_browser:
        webbrowser.open(f"http://127.0.0.1:{port}/check/")
    return run(PY, "manage.py", "runserver", f"127.0.0.1:{port}")


def target_api(params: dict[str, str]) -> int:
    return _serve(params.get("PORT", "8000"), open_browser=False)


def target_app(params: dict[str, str]) -> int:
    return _serve(params.get("PORT", "8000"), open_browser=params.get("OPEN", "1") == "1")


def target_demo(params: dict[str, str]) -> int:
    code = run_all(
        (PY, "manage.py", "migrate", "--noinput", "-v", "0"),
        (PY, "manage.py", "load_demo"),
    )
    if code != 0:
        return code
    return _serve(params.get("PORT", "8000"), open_browser=params.get("OPEN", "1") == "1")


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
    "data": target_data,
    "train": target_train,
    "eval": target_eval,
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
    if not argv or argv[0] not in TARGETS:
        print(__doc__)
        return 1
    os.environ.setdefault("PYTHONUTF8", "1")
    return TARGETS[argv[0]](parse_params(argv[1:]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
