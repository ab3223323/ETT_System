from django.urls import path, include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from .views import *

urlpatterns = [
path('',views.index),
]
urlpatterns += staticfiles_urlpatterns()