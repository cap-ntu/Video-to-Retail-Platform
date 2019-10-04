from django.urls import re_path

from . import views

app_name = "dashboard"

urlpatterns = [
    re_path('^.*$', views.index, name="index"),
]
