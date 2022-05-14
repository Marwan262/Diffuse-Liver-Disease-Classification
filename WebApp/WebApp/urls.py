from django.conf.urls import url, include
from django.contrib import admin
from django.urls import path
from django.conf import settings 
from django.conf.urls.static import static

# from .urls import urlpatterns as endpoints_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('base.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# urlpatterns += endpoints_urlpatterns