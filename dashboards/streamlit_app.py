# =============================================================
# DASHBOARD CHURN PREDICTION - STREAMLIT
# =============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import requests
import shap

# =============================================================
# CONFIGURATION DE LA PAGE
# =============================================================
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🎯",
    layout="wide"
)

# =============================================================
# CHARGEMENT DES DONNÉES ET DU MODÈLE
# =============================================================
@st.cache_data
def load_data():
    return pd.read_csv('data/processed/telco_churn_processed.csv')

@st.cache_resource
def load_model():
    with open('models/xgb_churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

df = load_data()
model, metadata = load_model()
SEUIL = metadata['seuil_optimal']

# =============================================================
# BARRE LATÉRALE - NAVIGATION
# =============================================================
st.sidebar.image("https://img.icons8.com/color/96/000000/customer-insight.png")
st.sidebar.title("🎯 Churn Prediction")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [" Vue Globale", "🔍 Prédiction Client", " Performance Modèle"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("** Dataset**")
st.sidebar.markdown(f"- {df.shape[0]:,} clients")
st.sidebar.markdown(f"- {df.shape[1]} variables")
st.sidebar.markdown(f"- Seuil optimal : {SEUIL}")

# =============================================================
# PAGE 1 : VUE GLOBALE
# =============================================================
if page == " Vue Globale":
    st.title(" Vue Globale du Churn")
    st.markdown("---")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)

    total_clients = len(df)
    total_churners = df['Churn'].sum()
    taux_churn = df['Churn'].mean() * 100
    revenu_risque = total_churners * 200

    col1.metric(" Total Clients", f"{total_clients:,}")
    col2.metric(" Churners", f"{total_churners:,}")
    col3.metric(" Taux de Churn", f"{taux_churn:.1f}%")
    col4.metric(" Revenu à Risque", f"{revenu_risque:,}€")

    st.markdown("---")

    # Graphiques
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution du Churn")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2ecc71', '#e74c3c']
        df['Churn'].value_counts().plot(
            kind='bar', ax=ax, color=colors)
        ax.set_xticklabels(['Non-Churn', 'Churn'], rotation=0)
        ax.set_ylabel("Nombre de clients")
        st.pyplot(fig)

    with col2:
        st.subheader("Churn par Type de Contrat")
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Reconstituer le type de contrat
        df['Contract_Type'] = 'Month-to-month'
        df.loc[df['Contract_One year'] == 1, 'Contract_Type'] = 'One year'
        df.loc[df['Contract_Two year'] == 1, 'Contract_Type'] = 'Two year'
        
        contrat_churn = df.groupby('Contract_Type')['Churn'].mean() * 100
        contrat_churn.plot(kind='bar', ax=ax, 
                          color=['#e74c3c', '#f39c12', '#2ecc71'])
        ax.set_ylabel("Taux de Churn (%)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

# =============================================================
# PAGE 2 : PRÉDICTION CLIENT
# =============================================================
elif page == "🔍 Prédiction Client":
    st.title("🔍 Prédiction du Churn Client")
    st.markdown("Renseignez les informations du client pour prédire son risque de churn.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader(" Démographie")
        gender = st.selectbox("Genre", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", ["Non", "Oui"])
        partner = st.selectbox("En couple", ["Non", "Oui"])
        dependents = st.selectbox("Personnes à charge", ["Non", "Oui"])
        tenure = st.slider("Ancienneté (mois)", 0, 72, 12)

    with col2:
        st.subheader(" Services")
        phone = st.selectbox("Service téléphonique", ["Non", "Oui"])
        multiple = st.selectbox("Lignes multiples", ["Non", "Oui"])
        internet = st.selectbox("Internet", ["DSL", "Fiber optic", "Non"])
        security = st.selectbox("Sécurité en ligne", ["Non", "Oui"])
        backup = st.selectbox("Sauvegarde en ligne", ["Non", "Oui"])
        device = st.selectbox("Protection appareil", ["Non", "Oui"])
        tech = st.selectbox("Support technique", ["Non", "Oui"])
        tv = st.selectbox("Streaming TV", ["Non", "Oui"])
        movies = st.selectbox("Streaming Films", ["Non", "Oui"])

    with col3:
        st.subheader(" Contrat & Paiement")
        contract = st.selectbox("Contrat", 
                               ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Facturation électronique", ["Non", "Oui"])
        payment = st.selectbox("Méthode de paiement", [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ])
        monthly = st.number_input("Charges mensuelles (€)", 
                                  min_value=0.0, max_value=200.0, value=65.0)
        total = st.number_input("Charges totales (€)", 
                                min_value=0.0, value=monthly * tenure)

    st.markdown("---")

    if st.button(" Prédire le Churn", type="primary"):
        # Préparer les données
        def encode(val):
            return 1 if val == "Oui" else 0

        data = {
            'gender': 1 if gender == "Male" else 0,
            'SeniorCitizen': encode(senior),
            'Partner': encode(partner),
            'Dependents': encode(dependents),
            'tenure': tenure,
            'PhoneService': encode(phone),
            'MultipleLines': encode(multiple),
            'OnlineSecurity': encode(security),
            'OnlineBackup': encode(backup),
            'DeviceProtection': encode(device),
            'TechSupport': encode(tech),
            'StreamingTV': encode(tv),
            'StreamingMovies': encode(movies),
            'PaperlessBilling': encode(paperless),
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
            'InternetService_No': 1 if internet == "Non" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment == "Mailed check" else 0,
        }

        # Nouvelles features
        services = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
        data['charge_par_mois'] = monthly / (tenure + 1)
        data['nb_services'] = sum(data[s] for s in services)
        data['nouveau_client'] = 1 if tenure <= 6 else 0
        data['client_moyen'] = 1 if 6 < tenure <= 24 else 0
        data['client_fidele'] = 1 if tenure > 24 else 0
        data['charges_estimees'] = monthly * 12
        data['risque_depart'] = 1 if (
            data['nouveau_client'] == 1 and
            data['Contract_One year'] == 0 and
            data['Contract_Two year'] == 0
        ) else 0
        data['charge_par_service'] = monthly / (data['nb_services'] + 1)
        data['client_isole'] = 1 if (
            data['Partner'] == 0 and data['Dependents'] == 0
        ) else 0

        # Prédiction
        features = metadata['features']
        df_pred = pd.DataFrame([data])[features]
        proba = model.predict_proba(df_pred)[0][1]

        # Affichage du résultat
        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Probabilité de Churn", f"{proba:.1%}")

        with col2:
            if proba >= 0.7:
                st.error(" Risque HIGH")
            elif proba >= SEUIL:
                st.warning(" Risque MEDIUM")
            else:
                st.success(" Risque LOW")

        with col3:
            if proba >= 0.7:
                st.error("Lancer campagne rétention urgente !")
            elif proba >= SEUIL:
                st.warning("Proposer une offre de fidélisation")
            else:
                st.success("Client fidèle - surveillance normale")

        # Jauge de probabilité
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.barh(['Churn'], [proba], color='#e74c3c' if proba >= SEUIL else '#2ecc71')
        ax.barh(['Churn'], [1 - proba], left=[proba], color='#ecf0f1')
        ax.set_xlim(0, 1)
        ax.axvline(x=SEUIL, color='orange', linestyle='--', label=f'Seuil ({SEUIL})')
        ax.set_xlabel('Probabilité')
        ax.legend()
        ax.set_title(f'Probabilité de Churn : {proba:.1%}')
        st.pyplot(fig)

        # =============================================================
        # EXPLICATION SHAP
        # =============================================================
        st.markdown("---")
        st.subheader("🔍 Pourquoi cette prédiction ?")

        # Calculer les valeurs SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_pred)

        # Top 10 variables
        shap_df = pd.DataFrame({
            'Variable': features,
            'Impact SHAP': shap_values[0],
            'Valeur': df_pred.iloc[0].values
        }).sort_values('Impact SHAP', key=abs, ascending=False).head(10)

        # Graphique
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        colors = ['#e74c3c' if v > 0 else '#2ecc71' 
                  for v in shap_df['Impact SHAP']]
        ax2.barh(shap_df['Variable'], shap_df['Impact SHAP'], color=colors)
        ax2.set_xlabel('Impact (rouge = pousse vers churn, vert = protège)')
        ax2.set_title('Explication de la prédiction', fontsize=14)
        ax2.axvline(x=0, color='black', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig2)

        # Explication en texte
        st.markdown("** Facteurs principaux :**")
        for _, row in shap_df.head(5).iterrows():
            direction = "🔴 pousse vers churn" if row['Impact SHAP'] > 0 else "🟢 protège du churn"
            st.markdown(f"- **{row['Variable']}** = {row['Valeur']:.2f} → {direction}")

# =============================================================
# PAGE 3 : PERFORMANCE DU MODÈLE
# =============================================================
elif page == " Performance Modèle":
    st.title(" Performance du Modèle")
    st.markdown("---")

    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    perf = metadata['performances']

    col1.metric(" Recall", f"{perf['recall']:.1%}",
                " Objectif > 80%" if perf['recall'] >= 0.80 else "❌ Objectif > 80%")
    col2.metric(" Precision", f"{perf['precision']:.1%}")
    col3.metric(" F1-Score", f"{perf['f1_score']:.3f}")
    col4.metric(" AUC", f"{perf['auc']:.3f}",
                " Objectif > 0.80" if perf['auc'] >= 0.80 else "❌ Objectif > 0.80")

    st.markdown("---")
    st.subheader(" Interprétation des Métriques")
    st.markdown("""
    | Métrique | Valeur | Objectif | Interprétation |
    |----------|--------|----------|----------------|
    | **Recall** | 81.6% | > 80% |  On détecte 81.6% des churners |
    | **Precision** | 48.4% | > 70% |  48.4% des alertes sont correctes |
    | **F1-Score** | 0.608 | > 0.75 |  Limite du dataset |
    | **AUC** | 0.825 | > 0.80 |  Bonne discrimination |
    
    **Note :** La Precision est sacrifiée pour maximiser le Recall car 
    rater un churner coûte 10x plus cher qu'une fausse alerte.
    """)