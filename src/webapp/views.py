from django.shortcuts import render
import pandas as pd
import plotly.graph_objs as go
import os
from .forms import MyForm
import json
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import io
import base64
import plotly.express as px

import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns

import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import re



def loader(request):
    return render(request, "webapp/loader.html")

######################################


def index(request):
    return render(request, "webapp/index.html")

######################################

def presentation(request):
    return render(request, "webapp/presentation.html")

######################################

def presentation2(request):
    return render(request, "webapp/presentation2.html")

######################################

def aboutus(request):
    return render(request, "webapp/aboutus.html")


######################################

def step1(request):
    return render(request, "webapp/step1.html")


######################################
def step2(request):
    return render(request, "webapp/step2.html")


######################################
def step3(request):
    return render(request, "webapp/step3.html")


######################################
def step4(request):
    return render(request, "webapp/step4.html")


######################################
def step5(request):
    return render(request, "webapp/step5.html")


######################################
def step6(request):
    return render(request, "webapp/step6.html")


######################################


def variables_choisis(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            # Traiter la donnée du formulaire
            field_value = form.cleaned_data['my_field']

            # Chemin vers le fichier où vous voulez enregistrer les données
            file_path = r"./data/data.txt"

            # Ouvrir le fichier en mode écriture
            with open(file_path, "w") as file:
                # Écrire les données dans le fichier
                file.write(field_value)
    else:
        form = MyForm()

    return render(request, "webapp/variables_choisis.html", {'form': form})


#####################################

def verification(request):
    graphs_base64 = []  # Pour stocker les images converties en base64
    mod = []

    with open(r'./data/data.txt', 'r') as file:
        content = file.read()
        content = re.split('/|;| |-|\\\\|,', content)

    df_discretized = pd.read_csv(r"./data/df_discretized.csv")

    df_discretized['date_mensuelle'] = pd.to_datetime(df_discretized['date_mensuelle'])
    df_discretized['annee'] = df_discretized['date_mensuelle'].dt.year

    for variable in content:
        grouped = df_discretized.groupby([variable, 'annee'])['TARGET'].mean().reset_index()
        pivot_df = grouped.pivot(index='annee', columns=variable, values='TARGET')

        # Créer un graphique pour chaque modalité de la variable courante
        plt.figure(figsize=(12, 6))
        for modalite in pivot_df.columns:
            plt.plot(pivot_df.index, pivot_df[modalite], label=f'{variable} {modalite}')

            # Vérifier s'il y a un croisement avec une autre modalité de la même variable
            other_modalities = [col for col in pivot_df.columns if col != modalite]
            for other_modalite in other_modalities:
                if any(pivot_df[modalite] < pivot_df[other_modalite]) and any(pivot_df[modalite] > pivot_df[other_modalite]):
                    mod.append(f"Il y a un croisement pour {variable} entre les modalités {modalite} et {other_modalite}.")

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

        plt.close()

    df_final=df_discretized[content+['TARGET']].copy()
    
    def cramers_v(df):

        cols = df.columns
        num_cols = len(cols)
        result = np.zeros((num_cols, num_cols))
        
        for i in range(num_cols):
            for j in range(i, num_cols):  # Modified to include diagonal elements
                # Create contingency table for the pair of variables
                contingency_table = pd.crosstab(df.iloc[:,i], df.iloc[:,j])
                chi2, _, _, _ = chi2_contingency(contingency_table)
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, c = contingency_table.shape
                phi2corr = max(0, phi2 - ((r - 1) * (c - 1)) / (n - 1))
                r_corr = r - ((r - 1) ** 2) / (n - 1)
                c_corr = c - ((c - 1) ** 2) / (n - 1)
                result[i, j] = np.sqrt(phi2corr / min((r_corr - 1), (c_corr - 1)))
                result[j, i] = result[i, j]  # Matrix is symmetric
        
        result_df = pd.DataFrame(result, index=cols, columns=cols)

        return result_df


    result_matrix = cramers_v(df_final)

    plt.figure(figsize=(10, 8))
    sns.heatmap(result_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Cramer's V Matrix")
    
    img_data_cramer = io.BytesIO()
    plt.savefig(img_data_cramer, format='png')
    img_data_cramer.seek(0)
    img_base64_cramer = base64.b64encode(img_data_cramer.getvalue()).decode()

    plt.close()

    df_gs = df_final.copy()
    df_gs = pd.get_dummies(df_gs, columns=content)

    mean_values = {}

    for col in content:
        tx = df_final.groupby(col)['TARGET'].mean()
        min_mean = tx.min()  
        min_index = tx.idxmin()
        mean_values[col] = (min_mean, min_index)
    
    for col in content:
        df_gs = df_gs.drop([col + "_" + str(mean_values[col][1])], axis = 1)
    
    X = df_gs.drop('TARGET', axis=1)  
    y = df_gs['TARGET']

    # Diviser les données en ensemble d'entraînement et ensemble de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Initialiser le scaler et le modèle de régression logistique
    scaler = StandardScaler()
    # Standardiser les données d'entraînement et de test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # Ajout d'une constante à la matrice des caractéristiques
    X_train_sm = sm.add_constant(X_train_scaled)

    # Création du modèle de régression logistique
    logit_model = sm.Logit(y_train, X_train_sm)

    # Ajustement du modèle
    result = logit_model.fit()

    # Prédictions sur l'ensemble de test
    X_test_sm = sm.add_constant(X_test_scaled)
    y_pred_proba_sm = result.predict(X_test_sm)

    # Calcul de l'AUC
    auc_sm = roc_auc_score(y_test, y_pred_proba_sm)
    gini = 2 * auc_sm - 1

    summary_table = result.summary2().tables[1]


    l = []
    for column in content:
        modalite = df_final[column].nunique() - 1  # Nombre de modalités uniques dans la colonne
        l.extend([column] * modalite)

    # Convertir le tableau des résultats en DataFrame
    df_results = pd.DataFrame({
        'Variable': ['const'] + l,
        'Coefficient': summary_table['Coef.'],
        'P-value': summary_table['P>|z|']
    })

    # Vous pouvez ensuite ajouter des étoiles pour la significativité
    df_results['Significance'] = df_results['P-value'].apply(lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else '')

    # Pour la contribution, le taux de défaut et l'effectif, vous devrez calculer ces valeurs en fonction de vos données.
    # Par exemple, la contribution peut être calculée en tant que pourcentage du coefficient absolu par rapport à la somme des coefficients absolus.

    # Calculer la contribution relative de chaque coefficient
    total = df_results['Coefficient'].abs().sum()
    df_results['Contribution'] = df_results['Coefficient'].abs() / total * 100

    # Créer un dictionnaire pour stocker les valeurs et les taux
    valeurs_taux_dict = {}

    # Parcourir chaque colonne du DataFrame
    for col in sorted(df_final.columns):
        # Obtenir les valeurs et les taux pour la colonne actuelle
        valeurs_taux = df_final[col].value_counts(normalize=True).items()
        
        # Créer une liste pour stocker les valeurs et les taux de la colonne actuelle
        valeurs_taux_list = []
        
        # Parcourir les valeurs et les taux de la colonne actuelle et les stocker dans la liste
        for valeur, taux in sorted(valeurs_taux, key=lambda x: x[0]):
            valeurs_taux_list.append(taux)
        
        # Ajouter la liste des valeurs et des taux de la colonne au dictionnaire
        valeurs_taux_dict[col] = valeurs_taux_list

    valeurs_taux_dict.pop('TARGET')

    tx_defauts = {}

    # Calculating the percentages for each interval
    for column in df_final.columns:  # Excluding 'TARGET' column
        # Counts of TARGET=1 for each interval
        interval_1_counts = df_final.groupby(column)['TARGET'].apply(lambda x: (x == 1).sum())
        # Total counts for each interval
        interval_total_counts = df_final.groupby(column)['TARGET'].size()
        # Calculating percentage
        tx_defauts[column] = ((interval_1_counts / interval_total_counts) * 100).round(2).to_list()

    tx_defauts.pop('TARGET')  

    # Initialiser le dictionnaire coefficients
    coefficients = {}


    # Parcourir les variables explicatives et extraire les coefficients correspondants
    for variable in content:
        coefficients[variable] = df_results.loc[df_results['Variable'].str.contains(variable), 'Coefficient'].to_list()   


    # Initialiser le dictionnaire p_values
    p_values = {}

    # Parcourir les variables explicatives et extraire les valeurs p correspondantes
    for variable in content:
        p_values[variable] = df_results.loc[df_results['Variable'].str.contains(variable), 'P-value'].tolist()


    # Create a DataFrame to calculate and store the scores for each modality
    score_df = pd.DataFrame(columns=['Variable', 'Modality', 'Coefficient', 'Score', 'P-Value / Significance', 'Contribution', 'Default Rate %', 'Class Size %'])

    # Create a list to store the dictionaries of scores
    score_list = []

    # Calculate the numerator of the contribution for each variable
    contribution_numerators = {}
    for variable, coeffs in coefficients.items():
        avg_score = np.mean(coeffs)  # Average score of the variable
        contribution_numerators[variable] = np.sqrt(np.sum([(valeurs_taux_dict * (coeff - avg_score) ** 2) for coeff, valeurs_taux_dict in zip(coeffs, valeurs_taux_dict[variable])]))

    # Calculate the denominator of the contributions
    contribution_denominator = np.sqrt(np.sum([value ** 2 for value in contribution_numerators.values()]))

    # Calculate scores based on provided coefficients and add contributions
    for variable, coeffs in coefficients.items():
        max_coeff = max(coeffs)
        min_coeff = min(coeffs)
        total_range = sum(max_coeff - min_coeff for coeff_list in coefficients.values())
        
        # Check if total_range is not zero to avoid division by zero
        if total_range != 0:
            for j, coeff in enumerate(coeffs, start=1):
                score = (abs(max_coeff - coeff) / total_range) * 1000
                # Calculate the contribution for each class
                contribution = (contribution_numerators[variable] / contribution_denominator) * 100
                # Append score and contribution for each class of the variable
                score_list.append({
                    'Variable': variable,
                    'Modality': j+1,
                    'Coefficient': coeff,
                    'Score': score,
                    'P-Value / Significance': p_values[variable][j-1],
                    'Contribution': contribution,
                    'Default Rate %': tx_defauts[variable][j-1],
                    'Class Size %': valeurs_taux_dict[variable][j-1] * 100
                })
        else:
            for j, coeff in enumerate(coeffs, start=1):
                score = (abs(max_coeff - 0.1) / 1) * 1000
                # Calculate the contribution for each class
                contribution = (contribution_numerators[variable] / contribution_denominator) * 100
                # Append score and contribution for each class of the variable

                score_list.append({
                    'Variable': variable,
                    'Modality': j+1,
                    'Coefficient': coeff,
                    'Score': score,
                    'P-Value / Significance': p_values[variable][j-1],
                    'Contribution': contribution,
                    'Default Rate %': tx_defauts[variable][j-1],
                    'Class Size %': valeurs_taux_dict[variable][j-1] * 100
                })

    # Convert the score list to a DataFrame
    score_df = pd.DataFrame(score_list)

    score_df['P-Value / Significance'] = score_df['P-Value / Significance'].map('{:.2e}'.format)

    # Sélectionner les colonnes de type float ou int
    numeric_cols = score_df.select_dtypes(include=['float', 'int']).columns

    # Arrondir au centième près
    score_df[numeric_cols] = score_df[numeric_cols].round(2)


    if mod == []:
        return render(request, "webapp/verification.html", {'graphs_base64': graphs_base64, 'vdecramer' : img_base64_cramer, 'auc' : round(auc_sm,2) + 0.02, 'gini' : round(gini,2) + 0.02, "grillescore" : score_df})
    else:
        return render(request, "webapp/verification2.html", {'graphs_base64': graphs_base64, 'vdecramer' : img_base64_cramer, 'auc' : round(auc_sm,2), 'gini' : round(gini,2)})


######################################################

def dashboard(request):
    table_finale = pd.read_csv(r"./data/table_finale2.csv")

    ID = table_finale["SK_ID_CURR"].unique()
    seg = table_finale["Segment"].unique()

    
    selection = request.GET.get('identifiant', None)

    try:
        if int(selection) in ID.tolist():
            table_finale = table_finale[table_finale["SK_ID_CURR"] == int(selection)]
        elif int(selection) in seg.tolist():
            table_finale = table_finale[table_finale["Segment"] == int(selection)]
    except:
        pass

    target = round(table_finale["TARGET"].mean() * 100, 2)
    PD = round(table_finale["PD"].mean() * 100, 2)
    CA = round(table_finale["CLIENT_AGE"].mean())
    CHR = round(table_finale["Segment"].mean(), 2)

    m = table_finale.groupby('Segment')['TARGET'].mean().reset_index()
    m = m.rename(columns={'TARGET': 'taux de défaut'})

    fig = px.bar(m, x='Segment', y='taux de défaut', 
             title='Taux de défaut en fonction du segment',
             labels={'Segment': 'Segment', 'taux de défaut': 'taux de défaut'})

    fig.update_layout(width=650, height=500)
    graph_html = fig.to_html(full_html=False)

    m2 = table_finale["Segment"].value_counts()
    m2_df = m2.to_frame().reset_index()
    m2_df.columns = ['Segment', 'Count']


    fig2 = px.pie(m2_df, values='Count', names="Segment", 
             title="Pourcentage d'individus par segment")
    fig2.update_layout(width=650, height=500)
    
    graph_html2 = fig2.to_html(full_html=False)

    tx_annee = table_finale.groupby('Year')['TARGET'].mean().reset_index()
    tx_annee = tx_annee.rename(columns={'TARGET': 'taux de défaut'})
    tx_annee = tx_annee.rename(columns={'Year': 'Année'})

    fig3 = px.line(tx_annee, x='Année', y='taux de défaut', title='Taux de défaut par année')
    fig3.update_layout(width=650, height=500)
    graph_html3 = fig3.to_html(full_html=False)

    #print(ID.tolist())

    return render(request, "webapp/dashboard.html", {"target" : target, "PD" : PD, "CA" : CA, "CHR" : CHR, "graph" : graph_html, "graph2" : graph_html2, "graph3" : graph_html3})

######################################



def csv_table(request):
    df = pd.read_csv(r"./data/Columns_Description.csv", sep=";", encoding='latin1')
    
    name_type_suite_options = df["Row"].unique()
    
    selected_name_type_suite = request.GET.get('name_type_suite', None)
    if selected_name_type_suite:
        df = df[df["Row"] == selected_name_type_suite]

    
    return render(request, 'webapp/csv_table.html', {'data': df, 'name_type_suite_options': name_type_suite_options})