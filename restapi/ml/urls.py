from django.urls import path
from ml import views



urlpatterns = [
    path('', views.start, name="start"),

]

