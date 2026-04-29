# =============================================================
# MAIN.PY - API FastAPI pour la prédiction du Churn
# =============================================================

from fastapi import FastAPI, HTTPException
from schemas import ClientData, PredictionResponse
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime

# =============================================================
# INITIALISATION DE L'API
# =============================================================
app = FastAPI(
    title=" Churn Prediction API",
    description="API de prédiction du churn client pour une entreprise de télécoms",
    version="1.0.0"
)

# =============================================================
# CHARGEMENT DU MODÈLE AU DÉMARRAGE
# =============================================================
print(" Chargement du modèle...")

with open('../models/xgb_churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('../models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

SEUIL = metadata['seuil_optimal']
FEATURES = metadata['features']

print(f" Modèle chargé ! Seuil optimal : {SEUIL}")

# =============================================================
# FONCTION UTILITAIRE
# =============================================================
def calculer_features(data: ClientData) -> pd.DataFrame:
    """
    Calcule toutes les features à partir des données du client,
    y compris les nouvelles variables créées dans le feature engineering.
    """
    d = data.dict()

    # Renommer les colonnes One-Hot pour correspondre au modèle
    d['Contract_One year'] = d.pop('Contract_One_year')
    d['Contract_Two year'] = d.pop('Contract_Two_year')
    d['InternetService_Fiber optic'] = d.pop('InternetService_Fiber_optic')
    d['InternetService_No'] = d.pop('InternetService_No')
    d['PaymentMethod_Credit card (automatic)'] = d.pop('PaymentMethod_Credit_card')
    d['PaymentMethod_Electronic check'] = d.pop('PaymentMethod_Electronic_check')
    d['PaymentMethod_Mailed check'] = d.pop('PaymentMethod_Mailed_check')

    # Nouvelles variables (feature engineering)
    d['charge_par_mois'] = d['MonthlyCharges'] / (d['tenure'] + 1)

    services = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
    d['nb_services'] = sum(d[s] for s in services)

    d['nouveau_client'] = 1 if d['tenure'] <= 6 else 0
    d['client_moyen'] = 1 if 6 < d['tenure'] <= 24 else 0
    d['client_fidele'] = 1 if d['tenure'] > 24 else 0
    d['charges_estimees'] = d['MonthlyCharges'] * 12

    d['risque_depart'] = 1 if (
        d['nouveau_client'] == 1 and
        d['Contract_One year'] == 0 and
        d['Contract_Two year'] == 0
    ) else 0

    d['charge_par_service'] = d['MonthlyCharges'] / (d['nb_services'] + 1)
    d['client_isole'] = 1 if (d['Partner'] == 0 and d['Dependents'] == 0) else 0

    # Créer le DataFrame dans le bon ordre
    df = pd.DataFrame([d])[FEATURES]
    return df

# =============================================================
# ENDPOINTS
# =============================================================

@app.get("/")
def home():
    """Page d'accueil de l'API."""
    return {
        "message": " Churn Prediction API",
        "version": "1.0.0",
        "status": " En ligne",
        "endpoints": {
            "/predict": "POST - Prédire le churn d'un client",
            "/health": "GET - Vérifier l'état de l'API",
            "/docs": "GET - Documentation interactive"
        }
    }

@app.get("/health")
def health():
    """Vérifier l'état de l'API et du modèle."""
    return {
        "status": " healthy",
        "model": "XGBoost",
        "seuil": SEUIL,
        "performances": metadata['performances'],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ClientData):
    """
    Prédire la probabilité de churn d'un client.
    Retourne la probabilité, le niveau de risque et une recommandation.
    """
    try:
        # Préparer les features
        df = calculer_features(data)

        # Prédiction
        proba = model.predict_proba(df)[0][1]

        # Niveau de risque
        if proba >= 0.7:
            risk_level = "HIGH"
            recommendation = " Lancer campagne rétention urgente"
        elif proba >= SEUIL:
            risk_level = "MEDIUM"
            recommendation = " Proposer une offre de fidélisation"
        else:
            risk_level = "LOW"
            recommendation = " Client fidèle - surveillance normale"

        return PredictionResponse(
            churn_probability=round(float(proba), 3),
            risk_level=risk_level,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))