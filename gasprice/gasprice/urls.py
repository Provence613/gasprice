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
from predict import  viewstime
from predict import  viewswallet
urlpatterns = [
    # path('admin/', admin.site.urls),
    # path(r'modeleval/',views.modeleval),
    path(r'chart/',views.chart),
    path(r'gasapi/',views.gasapi),
    path(r'datal/', views.datal),
    path(r'blockexplorerdata/',views.blockexplorerdata),
    path(r'data/',views.data),
    path(r'dataPage/',views.dataPage),
    path(r'homedata/', views.homedata),
    path(r'eval/',views.eval),
    path(r'pre/',views.pre),
    path(r'home/', views.home),
    path(r'about/', views.about),
    path(r'forecast/', viewstime.forecast),
    path(r'backtest/', viewstime.backtest),
    path(r'contact/', views.contact),
    path(r'blockexplorer/', views.blockexplorer),
    path(r'phome/', views.phome),
    path(r'pabout/', views.pabout),
    path(r'pforecast/', views.pforecast),
    path(r'pbacktest/', views.pbacktest),
    path(r'pcontact/', views.pcontact),
    path(r'pdataPage/',views.pdataPage),
    path(r'wallet/', viewswallet.wallet),
    path(r'pwallet/', viewswallet.pwallet),
]
