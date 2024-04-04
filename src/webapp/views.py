from django.shortcuts import render
#from .models import Pizza
import pandas as pd
import plotly.graph_objs as go



def index(request):
    return render(request, "webapp/index.html")

def page2(request):
    return render(request, "webapp/page2.html")

def aboutus(request):
    return render(request, "webapp/aboutus.html")

def csv_table(request):
    df = pd.read_csv(r"./data/application_train_vf.csv", index_col=False)
    data = df.head(10).copy()

    name_type_suite_options = data["NAME_TYPE_SUITE"].unique()
    
    selected_name_type_suite = request.GET.get('name_type_suite', None)
    if selected_name_type_suite:
        data = data[data["NAME_TYPE_SUITE"] == selected_name_type_suite]

    data['date_mensuelle'] = pd.to_datetime(data['date_mensuelle'])
    
    graph = go.Figure()
    graph.add_trace(go.Scatter(x=data['date_mensuelle'], y=data['AMT_INCOME_TOTAL'], mode='lines+markers'))
    graph.update_layout(title='Revenu total en fonction de la date mensuelle',
                        xaxis_title='Date mensuelle',
                        yaxis_title='Revenu total',
                        hovermode='x')
    
    graph_html = graph.to_html(full_html=False)
    
    return render(request, 'webapp/csv_table.html', {'data': data, 'name_type_suite_options': name_type_suite_options, 'graph_html': graph_html})