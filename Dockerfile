# 1. On part d'une base officielle Python 3.12 légère (slim)
FROM python:3.12-slim

# 2. On définit le dossier de travail à l'intérieur du conteneur
WORKDIR /app

# 3. On installe les outils de base (parfois requis par FAISS ou des librairies C++)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. On installe Poetry
RUN pip install --no-cache-dir poetry

# 5. Astuce de pro : on dit à Poetry de ne pas créer d'environnement virtuel
# (Le conteneur Docker EST déjà un environnement isolé en soi)
RUN poetry config virtualenvs.create false

# 6. On copie d'abord SEULEMENT les fichiers de dépendances
# (Ça permet à Docker de mettre en cache cette étape longue et de ne pas tout
# retélécharger à chaque fois que tu modifies juste une ligne de code Python)
COPY pyproject.toml poetry.lock ./

# 7. On installe les librairies (sans la racine du projet pour l'instant)
RUN poetry install --no-interaction --no-ansi --no-root

# 8. On copie le reste de notre code source
COPY src/ ./src/

# 9. On prévient Docker que notre API écoutera sur le port 8000
EXPOSE 8000

# 10. La commande de lancement ! 
# Attention : le host "0.0.0.0" est OBLIGATOIRE dans Docker pour que l'API soit accessible de l'extérieur.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
