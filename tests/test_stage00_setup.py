"""Stage 0: project skeleton."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from safenet import paths

REQUIRED_DIRS = [
    "config",
    "data/synthetic",
    "data/gold",
    "data/images",
    "data/feedback",
    "src/safenet/generator",
    "src/safenet/preprocess",
    "src/safenet/signals",
    "src/safenet/ml",
    "src/safenet/scoring",
    "src/safenet/multimodal",
    "src/safenet/graph",
    "src/safenet/eval",
    "src/safenet/api",
    "app",
    "models",
    "reports",
    "docs",
    "tests",
]

REQUIRED_FILES = ["TZ_kk.md", "PROGRESS.md", "README.md", "Makefile", "requirements.txt", "pyproject.toml"]
MAKE_TARGETS = ["install", "data", "train", "eval", "api", "app", "test", "test-stage", "demo"]


def test_import_safenet():
    import safenet

    assert safenet.__version__
    assert set(safenet.LABELS) == {"lottery", "casino", "pyramid", "none"}


@pytest.mark.parametrize("directory", REQUIRED_DIRS)
def test_directory_exists(directory):
    assert (paths.ROOT / directory).is_dir(), directory


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_file_exists(filename):
    assert (paths.ROOT / filename).is_file(), filename


def test_makefile_targets():
    text = (paths.ROOT / "Makefile").read_text(encoding="utf-8")
    for target in MAKE_TARGETS:
        assert f"\n{target}:" in text, target


def test_requirements_pinned():
    lines = [
        line.strip()
        for line in (paths.ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines
    assert all("==" in line for line in lines), "all dependencies must be pinned"


def test_make_test_stage_runs():
    if os.environ.get("SAFENET_NESTED_MAKE"):
        pytest.skip("nested run of make test-stage")
    if shutil.which("make"):
        command = ["make", "test-stage", "N=0", f"PY={sys.executable}"]
    elif os.name == "nt":
        command = ["cmd", "/c", str(paths.ROOT / "make.bat"), "test-stage", "N=0"]
    else:
        command = [sys.executable, str(paths.ROOT / "tasks.py"), "test-stage", "N=0"]
    env = {**os.environ, "SAFENET_NESTED_MAKE": "1"}
    proc = subprocess.run(command, cwd=paths.ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
