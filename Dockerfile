# =============================================================
# DOCKERFILE - API Churn Prediction
# =============================================================

# Image de base : Python 3.11 léger
FROM python:3.11-slim

# Définir le dossier de travail dans la boîte
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API
COPY api/ ./api/
COPY models/ ./models/

# Exposer le port 8000
EXPOSE 8000

# Commande pour lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]