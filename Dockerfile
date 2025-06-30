# Image de base Python 3.9 slim pour optimiser la taille
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier des dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p data logs

# Exposer le port (optionnel, pour monitoring futur)
EXPOSE 8000

# Variables d'environnement par défaut
ENV PYTHONPATH=/app
ENV TZ=UTC

# Commande par défaut
CMD ["python", "main.py"]