from django.urls import path

from . import views

urlpatterns = [
    path('predict', views.Predict.as_view(), name='predict')
]
