from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from machine_learning import views

urlpatterns = [
    path('', views.home, name='home'),
    path("admin/", admin.site.urls),
    path('classification/', views.classification, name='classification'),
    path('regression/', views.regression, name='regression'),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)