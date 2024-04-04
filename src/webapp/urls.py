from django.urls import path
from .views import index, page2, aboutus, csv_table

urlpatterns = [
    path('', index, name="webapp-index"),
    path('page2/', page2, name="page2"),
    path('aboutus/', aboutus, name="aboutus"),
    path('csv_table/', csv_table, name='csv_table'),
]