import pytest
import pandas as pd
import os
import sys
from unittest.mock import patch, MagicMock

# On s'assure que Python trouve le dossier src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_ingestion import process_and_filter_events, prepare_documents_by_event

# ==========================================
# 1. TEST DU NETTOYAGE DES DONNÉES (Pandas)
# ==========================================
def test_process_and_filter_events():
    """Teste si la fonction nettoie bien les données brutes d'OpenAgenda."""
    
    # Fausses données "brutes" simulant la réponse de l'API
    fausses_donnees_brutes = [
        {
            "title.fr": "Concert Super",
            "longDescription.fr": "Un super concert de rock.",
            "firstTiming.begin": "2026-06-15T20:00:00",
            "location.name": "Le Zénith",
            "location.city": "Strasbourg"
        },
        {
            # Événement incomplet pour voir comment le script gère les trous
            "title": "Expo Peinture",
            "location.city": "Colmar"
            # Il manque la description et la date
        }
    ]
    
    # On exécute ta fonction
    df_propre = process_and_filter_events(fausses_donnees_brutes)
    
    # Vérifications (Asserts)
    assert len(df_propre) == 2, "Le DataFrame devrait contenir 2 événements"
    assert "titre" in df_propre.columns, "La colonne 'titre' manque"
    assert "lieu" in df_propre.columns, "La colonne 'lieu' manque"
    
    # Vérification des remplacements de valeurs manquantes
    assert df_propre.iloc[0]["lieu"] == "Le Zénith (Strasbourg)"
    assert df_propre.iloc[1]["lieu"] == "Colmar"
    assert df_propre.iloc[1]["description"] == "Pas de description" # Ta valeur par défaut
    assert df_propre.iloc[1]["date_debut"] == "Récemment" # Ta valeur par défaut

# ==========================================
# 2. TEST DE LA PRÉPARATION POUR FAISS
# ==========================================
def test_prepare_documents_by_event():
    """Teste si le DataFrame est bien converti en objets Document LangChain."""
    
    # Faux DataFrame propre
    df_test = pd.DataFrame([
        {
            "titre": "Festival de Jazz",
            "description": "Musique en plein air",
            "date_debut": "2026-07-10",
            "lieu": "Mulhouse"
        }
    ])
    
    # On exécute ta fonction
    documents = prepare_documents_by_event(df_test)
    
    # Vérifications
    assert len(documents) == 1, "Il devrait y avoir 1 document"
    
    doc = documents[0]
    # On vérifie que le texte concaténé contient bien nos infos
    assert "Titre : Festival de Jazz" in doc.page_content
    assert "Lieu : Mulhouse" in doc.page_content
    
    # On vérifie que les métadonnées sont bien là
    assert doc.metadata["titre"] == "Festival de Jazz"
    assert doc.metadata["lieu"] == "Mulhouse"

# ==========================================
# 3. TEST DE L'API OPENAGENDA (MOCK)
# ==========================================
@patch('data_ingestion.requests.get')
def test_fetch_openagenda_events(mock_get):
    """Teste la fonction de récupération sans faire de vraie requête HTTP."""
    from data_ingestion import fetch_openagenda_events
    
    # On simule une réponse 200 OK avec un faux JSON
    mock_reponse = MagicMock()
    mock_reponse.json.return_value = {"events": [{"title": "Faux Event"}]}
    mock_reponse.raise_for_status = MagicMock()
    mock_get.return_value = mock_reponse
    
    # On appelle la fonction
    resultat = fetch_openagenda_events("uid_test_123")
    
    # On vérifie que la fonction renvoie bien la liste des événements
    assert len(resultat) == 1
    assert resultat[0]["title"] == "Faux Event"
    # On vérifie que l'URL a bien été appelée avec le bon UID
    mock_get.assert_called_once()
    assert "uid_test_123" in mock_get.call_args[0][0]
