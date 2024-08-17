from django.conf.urls import url
from temp import views

urlpatterns=[
    url('user/',views.user),
]