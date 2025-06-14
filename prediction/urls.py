from django.urls import path
from .views import predict_performance  # adjust if your function name is different

urlpatterns = [
    path('predicted/', predict_performance),
]
