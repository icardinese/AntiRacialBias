from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('manual_input/', views.manual_input, name='manual_input'),
    path('document_scan/', views.document_scan, name='document_scan'),
    path('data_transparency/', views.data_transparency, name='data_transparency'),
    path('data_history/', views.data_history, name='data_history'),
]
