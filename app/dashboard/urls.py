from django.contrib.auth.views import LogoutView
from django.urls import path

from app.dashboard import views
from app.dashboard.auth import LoginView

urlpatterns = [
    path("", views.queue, name="queue"),
    path("login/", LoginView.as_view(), name="login"),
    path("logout/", LogoutView.as_view(), name="logout"),
    path("case/<str:case_id>/", views.case_detail, name="case"),
    path("case/<str:case_id>/feedback/", views.case_feedback, name="case_feedback"),
    path("case/<str:case_id>/image/", views.case_image, name="case_image"),
    path("check/", views.check, name="check"),
    path("network/", views.network, name="network"),
    path("network/<str:cluster_id>/graph/", views.network_graph, name="network_graph"),
    path("metrics/", views.metrics, name="metrics"),
    path("reports/<str:name>", views.report_file, name="report_file"),
    path("lang/<str:code>/", views.set_lang, name="set_lang"),
]
