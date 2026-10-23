"""URL configuration for SafeNet KZ.

* analyst dashboard (stage 10): ``/`` (case queue), ``/case/<id>/``, ``/check/``, ``/network/``, ``/metrics/``;
* API (stage 9) at the root so its paths are exactly those of the TZ: ``/analyze``,
  ``/analyze/image``, ``/batch``, ``/health``, ``/docs``.
"""

from django.contrib import admin
from django.urls import include, path

from safenet.api.api import api

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("app.dashboard.urls")),
    path("", api.urls),
]
