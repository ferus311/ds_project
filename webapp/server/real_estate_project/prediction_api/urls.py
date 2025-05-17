from django.urls import path
from . import views
 
urlpatterns = [
    path('predict_house_price/', views.predict_house_price_api, name='predict_house_price_api'),
    path('geocode_address/', views.geocode_address_api, name='geocode_address_api'),
] 