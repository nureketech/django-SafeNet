"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import pytest

from safenet import paths


def require(path, hint: str = "make data") -> None:
    """Skip the calling test with a clear reason when a build artifact is missing."""
    if not path.exists():
        pytest.skip(f"{path.relative_to(paths.ROOT)} жоқ — алдымен `{hint}` іске қосыңыз")


@pytest.fixture(scope="session")
def root():
    return paths.ROOT
