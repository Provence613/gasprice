"""gasprice URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# 首先导入对应app的views文件
from predict import  views
urlpatterns = [
    # path('admin/', admin.site.urls),
    path(r'data/',views.data),
    path(r'show/',views.show),
    path(r'eval/',views.eval),
    path(r'pre/',views.pre),
    path(r'login/',views.login),
    path(r'register/',views.register),
    path(r'logout/',views.logout),

]
