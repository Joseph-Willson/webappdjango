from django.urls import path
from .views import index, presentation, aboutus, csv_table, variables_choisis, verification, step1, step2, step5, step6, dashboard, presentation2, loader

urlpatterns = [
    path('', index, name="webapp-index"),
    path('presentation/', presentation, name="page2"),
    path('aboutus/', aboutus, name="aboutus"),
    path('variables_choisis/csv_table/', csv_table, name='csv_table'),
    path('variables_choisis/', variables_choisis, name='variables_choisis'),
    path('verification/', verification, name='verification'),
    path('step1/', step1, name='step1'),
    path('step2/', step2, name='step2'),
    path('step5/', step5, name='step5'),
    path('step6/', step6, name='step6'),
    path('dashboard/', dashboard, name='dashboard'),
    path('presentation2/', presentation2, name='presentation2'),
    path('loader/', loader, name='loader'),
    
]