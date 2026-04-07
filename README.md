📅 Chatbot RAG - Événements Culturels en Alsace
📝 Description du Projet

Ce projet consiste en la création d'un système de Génération Augmentée par la Recherche (RAG) visant à interroger une base de données d'événements culturels en Alsace (Strasbourg, Mulhouse, Colmar, Sélestat, Haguenau). Le système est encapsulé dans une API REST robuste et conteneurisé via Docker.

🏗️ 1. Architecture Globale

L'architecture se divise en deux flux principaux :

    Pipeline d'Ingestion (Route /rebuild) :

        Récupération des données brutes au format JSON via l'API OpenAgenda.

        Nettoyage et structuration des données via un script Python (Pandas).

        Vectorisation des textes via le modèle d'embedding, puis stockage local dans une base vectorielle.

    Pipeline d'Interrogation (Route /ask) :

        L'utilisateur pose une question en langage naturel.

        La question est vectorisée et comparée à la base vectorielle pour extraire le contexte le plus pertinent (Top 3 des événements).

        Un modèle LLM génère la réponse finale en se basant strictement sur ce contexte.

🛠️ 2. Choix Technologiques & Modèles Utilisés

    Backend & API : FastAPI

        Pourquoi ? Pour sa rapidité, la validation automatique des types avec Pydantic, et la génération native de la documentation (Swagger UI).

    Base de données Vectorielle : FAISS

        Pourquoi ? Solution locale, légère et extrêmement rapide pour la recherche de similarité, parfaitement adaptée à un volume de données modéré (environ 1500 chunks).

    Modèles d'Intelligence Artificielle : Mistral AI

        Embedding : mistral-embed (Pour transformer les descriptions d'événements en vecteurs).

        LLM : mistral-small (Choisi pour son excellent ratio rapidité/performance. Il est bridé par un prompt système strict et une température basse pour éviter les hallucinations).

    Déploiement : Docker

        Pourquoi ? Pour garantir que l'application tourne de manière identique sur n'importe quel environnement, en isolant les dépendances.

✂️ 3. Stratégie de Découpage (Chunking)

Une attention particulière a été portée à la préparation des données pour optimiser la recherche sémantique :

    Injection des métadonnées : Le titre, la date et le lieu sont concaténés directement dans la description de l'événement avant vectorisation.

    Logique Métier (1 événement = 1 Document) : Les données d'agenda formant des blocs de sens cohérents, nous avons privilégié une approche sémantique naturelle plutôt qu'un découpage arbitraire.

    Sécurité pour les textes longs : Utilisation du RecursiveCharacterTextSplitter de LangChain avec une limite de 1000 caractères et un chevauchement (overlap) de 100 caractères, appliqué uniquement aux descriptions dépassant la limite.

📊 4. Tests et Résultats Observés

La robustesse du système est assurée par plusieurs couches de tests :

    Tests Unitaires (pytest) : Validation du script de nettoyage Pandas (gestion des valeurs manquantes, absence de .fr dans les clés API).

    Tests d'Intégration : Mise en place d'une route /health et tests automatisés des codes HTTP (200, 400, 503).

    Évaluation RAG (Ragas) : Utilisation du framework Ragas (LLM-as-a-judge) pour comparer les réponses du bot avec un jeu de données annoté (réponses attendues).

    Sécurité & Jailbreak : Le chatbot a été soumis à des tests de contournement (ex: demande de recette de cocktail Molotov). Résultat : Le bot refuse de répondre, prouvant le bon respect du prompt système restrictif.

🚀 5. Pistes d'Amélioration

Bien que fonctionnel, le système pourrait être amélioré via les axes suivants :

    Reformulation de requête (Query Rewriting) : Si la question de l'utilisateur est trop vague ou mal formulée, utiliser le LLM pour la reformuler ou l'enrichir avant d'interroger la base vectorielle.

    Filtrage par Métadonnées (Self-Querying) : Migrer de FAISS vers une solution comme ChromaDB pour permettre au LLM de construire lui-même des requêtes avec des filtres stricts (ex: WHERE date > '2026-06-01').

⚙️ 6. Instructions de lancement
Bash

# 1. Construire l'image Docker
docker build -t chatbot-alsace .

# 2. Lancer le conteneur (avec persistance de la BDD via volume)
docker run -p 8000:8000 -v $(pwd)/faiss_index:/app/faiss_index --env-file .env chatbot-alsace

# L'API est disponible sur http://localhost:8000/docs
