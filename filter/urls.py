from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="filter-home"),
    path("about/", views.about, name="filter-about"),
    path("email/<int:pk>/", views.EmailDetailView.as_view(), name="email-detail"),
    path("email/spam/", views.spam, name="spam-email"),
    path("email/ham/", views.ham, name="ham-email"),
    path("email/verify/<int:pk>/", views.verify_email, name="verify-email"),
]
