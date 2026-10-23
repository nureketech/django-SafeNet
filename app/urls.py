"""URL configuration for SafeNet KZ.

* analyst dashboard: ``/`` (case queue), ``/case/<id>/``, ``/check/``, ``/network/``, ``/metrics/``,
  ``/login/``, ``/logout/`` — every page needs a signed-in user (``LoginRequiredMiddleware``);
* API at the root so its paths are exactly those of the TZ: ``/analyze``, ``/analyze/image``,
  ``/batch``, ``/health``, ``/docs``. The API authenticates on its own (bearer token or session),
  so its views are exempt from the login redirect; ``/docs`` still requires a signed-in user.
"""

from django.contrib import admin
from django.contrib.auth.decorators import login_not_required
from django.urls import URLPattern, include, path

from safenet.api.api import api

api_patterns, api_app, api_namespace = api.urls
for pattern in api_patterns:
    if isinstance(pattern, URLPattern):
        pattern.callback = login_not_required(pattern.callback)

admin.site.site_header = "SafeNet KZ"
admin.site.site_title = "SafeNet KZ"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("app.dashboard.urls")),
    path("", include((api_patterns, api_app), namespace=api_namespace)),
]
