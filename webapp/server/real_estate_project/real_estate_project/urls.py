from django.contrib import admin
from django.urls import path, include
 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('prediction_api.urls')), # Thêm URL của app
] 