from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_image, name='predict_image'),
    path('run_script/', views.run_script, name='run_script'),
]
