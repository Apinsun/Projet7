import requests
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
from dotenv import load_dotenv
from mistralai.client import Mistral
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

# Charger les variables d'environnement
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)
open_agenda_api_key= os.getenv("OPEN_AGENDA_API_KEY")
def create_and_save_faiss_index(documents):
    """Vectorise les documents par petits paquets pour éviter l'erreur 400."""
    print(f"🧠 4. Lancement de la vectorisation par paquets...")
    
    api_key = os.getenv("MISTRAL_API_KEY")
    embeddings_model = MistralAIEmbeddings(
        mistral_api_key=api_key, 
        model="mistral-embed"
    )
    
    # On définit une taille de paquet (batch)
    # 50 est un bon chiffre pour ne pas saturer l'API gratuite
    batch_size = 50
    vectorstore = None

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        print(f"   ✨ Vectorisation du paquet {i//batch_size + 1} ({len(batch)} documents)...")
        
        try:
            if vectorstore is None:
                # Création du premier index avec le premier paquet
                vectorstore = FAISS.from_documents(batch, embeddings_model)
            else:
                # Ajout des paquets suivants à l'index existant
                vectorstore.add_documents(batch)
        except Exception as e:
            print(f"   ❌ Erreur sur ce paquet : {e}")
            continue # On passe au suivant si l'un plante

    # Sauvegarde finale
    if vectorstore:
        dossier_sauvegarde = "./faiss_index"
        vectorstore.save_local(dossier_sauvegarde)
        print(f"💾 ✅ Index FAISS sauvegardé dans '{dossier_sauvegarde}' !")

def fetch_openagenda_events(agenda_uid, search_term="Strasbourg"):
    """Récupère les événements d'un agenda spécifique sur OpenAgenda."""
    
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    
    # On ajoute la clé dans les paramètres de l'URL
    params = {
        "search": search_term,
        "size": 100,
        "key": open_agenda_api_key 
    }
    
    response = requests.get(url, params=params)
    
    # Si la requête échoue, ça lèvera une erreur proprement au lieu de planter silencieusement
    response.raise_for_status() 
    
    data = response.json()
    return data.get("events", [])

def process_and_filter_events(all_raw_data):
    import pandas as pd
    df = pd.json_normalize(all_raw_data)
    
    print("\n--- 🕵️ DÉBUT DU DEBUG PANDAS ---")
    print(f"Total initial d'événements : {len(df)}")
    
    # 1. On regarde ce que l'API nous a VRAIMENT envoyé
    colonnes_recues = df.columns.tolist()
    print(f"\nColonnes disponibles envoyées par l'API :")
    # On affiche juste les 15 premières pour ne pas spammer, mais ça donnera un gros indice
    print(colonnes_recues[:15], "...\n") 

    if len(all_raw_data) > 0:
        print("Aperçu des clés du TOUT PREMIER événement brut :")
        print(list(all_raw_data[0].keys()))
        print("-" * 30)

    # 2. GESTION DES TEXTES (Sécurisée)
    titre_col = 'title.fr' if 'title.fr' in df.columns else 'title'
    desc_col = 'description.fr' if 'description.fr' in df.columns else 'description'
    long_desc_col = 'longDescription.fr' if 'longDescription.fr' in df.columns else 'longDescription'
    
    if long_desc_col not in df.columns:
        df[long_desc_col] = None

    # Utilisation de .get pour éviter tout plantage si la colonne n'existe vraiment pas
    df['titre'] = df.get(titre_col, 'Titre inconnu').fillna('Titre inconnu')
    df['description'] = df.get(long_desc_col, df.get(desc_col, 'Pas de description')).fillna('Pas de description')

    # 3. GESTION DES DATES (Avec rapport de situation)
    if 'firstTiming' in df.columns:
        df['date_debut'] = pd.to_datetime(df['firstTiming'], errors='coerce')
        print("Info : Colonne 'firstTiming' trouvée et utilisée pour les dates.")
    elif 'dateRange.fr' in df.columns:
        # Une autre possibilité de l'API OpenAgenda
        print("Info : Utilisation de 'dateRange.fr' au lieu de firstTiming.")
        df['date_debut'] = df['dateRange.fr'] 
    elif 'timings' in df.columns:
        df['date_debut'] = df['timings'].apply(lambda x: x[0]['begin'] if isinstance(x, list) and len(x) > 0 else None)
        df['date_debut'] = pd.to_datetime(df['date_debut'], errors='coerce')
        print("Info : Colonne 'timings' (ancien format) trouvée et utilisée.")
    else:
        df['date_debut'] = None
        print("⚠️ AVERTISSEMENT : AUCUNE colonne de date connue n'a été trouvée dans les données !")

    # 4. LE FILTRAGE (On compte les morts 😅)
    nb_avant_titre = len(df)
    df = df[df['titre'] != 'Titre inconnu']
    nb_apres_titre = len(df)
    print(f"🗑️ Filtrage des titres : {nb_avant_titre - nb_apres_titre} événements supprimés car 'Titre inconnu'.")

    nb_avant_date = len(df)
    df = df.dropna(subset=['date_debut'])
    nb_apres_date = len(df)
    print(f"🗑️ Filtrage des dates : {nb_avant_date - nb_apres_date} événements supprimés car aucune date valide n'a été lue.")

    print(f"✅ Reste final : {len(df)} événements conservés pour vectorisation.")
    print("--- 🏁 FIN DU DEBUG ---\n")
    
    return df

def get_mistral_embeddings(textes):
    """Génère des embeddings pour une liste de textes via l'API Mistral (v2.x)."""
    # ... ton code précédent (client = Mistral(...)) ...
    
    # Correction de l'appel pour la version 2.1.3
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=textes
    )
    
    # Dans la v2, la réponse est un objet, on accède aux données comme ceci :
    return [item.embedding for item in response.data]

def prepare_documents_by_event(df) -> list[Document]:
    print("🔄 Conversion et filtrage de sécurité pour Mistral...")
    documents = []
    
    for index, row in df.iterrows():
        titre = str(row.get('titre', '')).strip()
        description = str(row.get('description', '')).strip()
        date_str = str(row.get('date', 'Inconnue')).strip()

        # Construction du texte
        texte_complet = f"Titre : {titre}\nDate : {date_str}\nDescription : {description}".strip()

        # --- SÉCURITÉS ANTI-ERREUR 400 ---
        
        # 1. On ignore si le texte est vide ou trop court (moins de 10 caractères)
        if len(texte_complet) < 10:
            continue 

        # 2. On tronque si c'est trop long (Mistral limite à environ 16k tokens, 
        # on va limiter à 10 000 caractères par sécurité, ce qui est déjà énorme)
        if len(texte_complet) > 10000:
            texte_complet = texte_complet[:10000] + "..."

        metadata = {"titre": titre, "date": date_str}
        documents.append(Document(page_content=texte_complet, metadata=metadata))
        
    print(f"✅ Filtrage terminé : {len(documents)} documents valides prêts pour Mistral.")
    return documents

if __name__ == "__main__":
    # Liste de plusieurs UIDs d'agendas de Strasbourg/Grand Est
    # (Tu peux en ajouter autant que tu veux !)
    AGENDA_UIDS = ["35291330", "65745437", "82229342", "73123154", "10748715", "97272582"] 
    
    print("📥 Récupération des données multi-agendas...")
    all_raw_data = []
    
    # On boucle sur chaque UID pour accumuler les événements
    for uid in AGENDA_UIDS:
        print(f"   -> Interrogation de l'agenda {uid}...")
        try:
            events = fetch_openagenda_events(uid, "Strasbourg")
            all_raw_data.extend(events)
        except Exception as e:
            print(f"   ⚠️ Erreur sur l'agenda {uid} : {e}")
            
    print(f"✅ {len(all_raw_data)} événements bruts récupérés au total.")
    
    print("🧹 Traitement avec Pandas...")
    df_clean = process_and_filter_events(all_raw_data)
    print(f"✅ {len(df_clean)} événements conservés après filtrage.")
    
    print("📦 3. Création des Documents LangChain (1 événement = 1 Document)...")
    documents = prepare_documents_by_event(df_clean)
    
    if documents:
            # On saute le test manuel et on lance la création de la BDD
            print("🚀 4. Création de la base de données FAISS (cela peut prendre 1-2 min)...")
            create_and_save_faiss_index(documents)
