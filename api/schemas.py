# =============================================================
# SCHEMAS.PY - Définition des structures de données
# =============================================================
# Pydantic valide automatiquement que les données reçues
# sont du bon type (int, float, str...)

from pydantic import BaseModel
from typing import Optional

class ClientData(BaseModel):
    """
    Structure des données d'un client pour la prédiction.
    Toutes les variables utilisées par le modèle.
    """
    # Démographie
    gender: int                    # 0=Female, 1=Male
    SeniorCitizen: int             # 0=Non, 1=Oui
    Partner: int                   # 0=Non, 1=Oui
    Dependents: int                # 0=Non, 1=Oui

    # Contrat
    tenure: int                    # Ancienneté en mois
    PhoneService: int              # 0=Non, 1=Oui
    MultipleLines: int             # 0=Non, 1=Oui
    OnlineSecurity: int            # 0=Non, 1=Oui
    OnlineBackup: int              # 0=Non, 1=Oui
    DeviceProtection: int          # 0=Non, 1=Oui
    TechSupport: int               # 0=Non, 1=Oui
    StreamingTV: int               # 0=Non, 1=Oui
    StreamingMovies: int           # 0=Non, 1=Oui
    PaperlessBilling: int          # 0=Non, 1=Oui

    # Charges
    MonthlyCharges: float          # Charges mensuelles
    TotalCharges: float            # Charges totales

    # Type de contrat (One-Hot)
    Contract_One_year: int         # 0=Non, 1=Oui
    Contract_Two_year: int         # 0=Non, 1=Oui

    # Internet (One-Hot)
    InternetService_Fiber_optic: int  # 0=Non, 1=Oui
    InternetService_No: int           # 0=Non, 1=Oui

    # Paiement (One-Hot)
    PaymentMethod_Credit_card: int    # 0=Non, 1=Oui
    PaymentMethod_Electronic_check: int  # 0=Non, 1=Oui
    PaymentMethod_Mailed_check: int   # 0=Non, 1=Oui


class PredictionResponse(BaseModel):
    """
    Structure de la réponse retournée par l'API.
    """
    churn_probability: float       # Probabilité de churn (0-1)
    risk_level: str                # LOW, MEDIUM, HIGH
    recommendation: str            # Action recommandée