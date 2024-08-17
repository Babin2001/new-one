from django.conf.urls import url
from rev import views

urlpatterns=[
    url('review/',views.review),
    ]