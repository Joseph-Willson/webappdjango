from django.shortcuts import render
import pandas as pd
import plotly.graph_objs as go
import os
from .forms import MyForm
import json
import matplotlib.pyplot as plt
import io
import base64


def index(request):
    return render(request, "webapp/index.html")

def page2(request):
    return render(request, "webapp/page2.html")

def aboutus(request):
    return render(request, "webapp/aboutus.html")

def variables_choisis(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Traiter la donnée du formulaire
            field_value = form.cleaned_data['my_field']

            # Chemin vers le fichier où vous voulez enregistrer les données
            file_path = "../src/data/data.txt"

            # Ouvrir le fichier en mode écriture
            with open(file_path, "w") as file:
                # Écrire les données dans le fichier
                file.write(field_value)

    else:
        form = MyForm()

    

    return render(request, "webapp/variables_choisis.html", {'form': form})


def verification(request):
    graphs_base64 = []  # Pour stocker les images converties en base64

    with open('../src/data/data.txt', 'r') as file:
        content = file.read().split("-")

    df_discretized = pd.read_csv("../src/data/df_discretized.csv")

    df_discretized['date_mensuelle'] = pd.to_datetime(df_discretized['date_mensuelle'])
    df_discretized['annee'] = df_discretized['date_mensuelle'].dt.year

    for variable in content:
        grouped = df_discretized.groupby([variable, 'annee'])['TARGET'].mean().reset_index()
        pivot_df = grouped.pivot(index='annee', columns=variable, values='TARGET')

        plt.figure(figsize=(12, 6))
        for modalite in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[modalite], label=f'{variable} {modalite}')

        plt.title(f"Évolution de la moyenne de TARGET pour {variable} par annee")
        plt.xlabel('annee')
        plt.ylabel(f'Moyenne de TARGET pour {variable}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Sauvegarder le graphique en tant que fichier PNG
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)
        img_base64 = base64.b64encode(img_data.getvalue()).decode()
        graphs_base64.append(img_base64)

        plt.close()  # Fermer le graphique pour libérer la mémoire

    return render(request, "webapp/verification.html", {'graphs_base64': graphs_base64})
        






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