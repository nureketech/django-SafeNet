"""URL configuration for SafeNet KZ.

The API (stage 9) is mounted at the root so its paths are exactly those of the
TZ: ``/analyze``, ``/analyze/image``, ``/batch``, ``/health``, ``/docs``.
"""

from django.contrib import admin
from django.urls import path

from safenet.api.api import api

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", api.urls),
]
