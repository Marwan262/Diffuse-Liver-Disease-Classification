from unicodedata import name
from django.urls import path
from django.conf.urls import url, include
from rest_framework import routers
from . import views


router = routers.DefaultRouter()

urlpatterns = [
    path('', views.index, name = "index"),
    path('patients/', views.patients, name = "patients"),
    path('patients/<int:id>/', views.patient, name = "patient"),
    path('doctors/', views.doctors, name = "doctors"),
    path('doctors/<int:id>/', views.doctor, name = "doctor"),
    path('report/<int:id>', views.report, name = "report"),
    path('addpatient/', views.addpatient, name = "addpatient"),
    path('adddoctor/', views.adddoctor, name = "adddoctor"),
    path('addreport/<int:id>/', views.addreport, name = "addreport"),
    path('deletepatient/<int:id>/', views.deletepatient, name = "deletepatient"),
    path('deletedoctor/<int:id>/', views.deletedoctor, name = "deletedoctor"),
    path('deletereport/<int:id>/', views.deletereport, name = "deletereport"),
    path('deletecondition/<str:condition>/<int:id>/', views.deletecondition, name = "deletecondition"),
    path('restore/<int:id>/', views.restore, name = "restore"),
    path('archive/<int:id>/', views.archive, name = "archive"),
    path('newpass/<int:id>/<str:role>/', views.newpassword, name = "newpass"),
    path('login/', views.login, name = "login"),
    path('logout/', views.logout, name = "logout"),
    path('profile/', views.profile, name = "profile"),
    path('api/', include(router.urls)),
    path('classify/<int:id>/', views.get_prediction, name = "classify"),
    path('error/', views.error, name = "error"),
    path('pdf/<int:id>/', views.pdf, name = "pdf"),
] 