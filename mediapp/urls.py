from django.contrib import admin
from django.urls import path, include
from core_app import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.index,name='index'),
    path('index.html', views.index, name='index'),
    path('detect.html', views.detect, name='detect'),
    path('breast_cancer.html', views.breast_cancer, name='breast_cancer'),
    path('malaria.html', views.malaria, name='malaria'),
    path('brain_tumor.html', views.brain_tumor, name='brain_tumor'),
    path('oct.html', views.oct, name='oct'),
    path('diabetes.html', views.diabetes, name='diabetes'),
]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
