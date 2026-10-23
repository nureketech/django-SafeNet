"""User roles: Django groups with fixed permissions (created after every ``migrate``).

* ``analyst``  — works the queue: views cases, checks content, saves results, records verdicts;
* ``observer`` — read-only: views cases, the network, metrics and verdict history.

Administrators (superusers) manage accounts, roles and API tokens in ``/admin/``.
"""

from __future__ import annotations

ROLE_ANALYST = "analyst"
ROLE_OBSERVER = "observer"
ROLE_PERMISSIONS: dict[str, tuple[str, ...]] = {
    ROLE_ANALYST: ("view_case", "add_case", "view_feedback", "add_feedback"),
    ROLE_OBSERVER: ("view_case", "view_feedback"),
}


def ensure_roles(**_kwargs) -> None:
    """Create the role groups and (re)set their permissions; safe to run repeatedly."""
    from django.contrib.auth.models import Group, Permission

    for role, codenames in ROLE_PERMISSIONS.items():
        group, _ = Group.objects.get_or_create(name=role)
        perms = Permission.objects.filter(content_type__app_label="dashboard", codename__in=codenames)
        if perms.count() == len(codenames):  # permissions exist once the dashboard tables are migrated
            group.permissions.set(perms)
