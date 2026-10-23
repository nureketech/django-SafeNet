from django.apps import AppConfig
from django.db.models.signals import post_migrate


class DashboardConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "app.dashboard"
    label = "dashboard"
    verbose_name = "SafeNet KZ"

    def ready(self):
        from app.dashboard import audit  # noqa: F401  (connects the login signals)
        from app.dashboard.roles import ensure_roles

        post_migrate.connect(ensure_roles, sender=self)
