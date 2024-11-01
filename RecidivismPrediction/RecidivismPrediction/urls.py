from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('prediction.urls')),  # This connects to your app's urls.py
]
