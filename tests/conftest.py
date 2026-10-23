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


# ------------------------------------------------------------- accounts (production access control)
TEST_PASSWORD = "correct-horse-battery-42"


def make_user(django_user_model, username: str, role: str | None):
    from django.contrib.auth.models import Group

    user = django_user_model.objects.create_user(username, password=TEST_PASSWORD)
    if role:
        user.groups.add(Group.objects.get(name=role))
    return user


@pytest.fixture
def analyst(db, django_user_model):
    from app.dashboard.roles import ROLE_ANALYST

    return make_user(django_user_model, "analyst1", ROLE_ANALYST)


@pytest.fixture
def observer(db, django_user_model):
    from app.dashboard.roles import ROLE_OBSERVER

    return make_user(django_user_model, "observer1", ROLE_OBSERVER)


@pytest.fixture
def api_token(db) -> str:
    from app.dashboard.models import ApiToken

    return ApiToken.issue("tests")[1]
