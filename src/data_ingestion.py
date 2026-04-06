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
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        # 1. On récupère le chemin absolu du dossier "src" où se trouve ce script
        dossier_src = os.path.dirname(os.path.abspath(__file__))
        
        # 2. On remonte d'un cran pour atteindre la racine "Projet7"
        racine_projet = os.path.dirname(dossier_src)
        
        # 3. On crée le chemin final, blindé et absolu
        dossier_sauvegarde = os.path.join(racine_projet, "faiss_index")
        
        vectorstore.save_local(dossier_sauvegarde)
        print(f"💾 ✅ Index FAISS sauvegardé dans '{dossier_sauvegarde}' !")

def get_top_agendas_by_location(location, limit=10):
    """Récupère dynamiquement les UIDs des meilleurs agendas pour une ville donnée."""
    url = "https://api.openagenda.com/v2/agendas"
    params = {
        "search": location,
        "size": limit,
        "key": open_agenda_api_key
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return [agenda["uid"] for agenda in data.get("agendas", [])]

def fetch_openagenda_events(agenda_uid, search_term=None):
    url = f"https://api.openagenda.com/v2/agendas/{agenda_uid}/events"
    date_limite = (datetime.now() - relativedelta(years=1)).strftime("%Y-%m-%d")
    
    params = {
        "size": 100, # On prend le maximum par agenda
        "key": open_agenda_api_key,
        "timings[gte]": date_limite 
    }
    
    # On n'ajoute le filtre de recherche que s'il est précisé
    if search_term:
        params["search"] = search_term
        
    response = requests.get(url, params=params)
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
    # --- Pour le titre ---
    df['titre'] = 'Titre inconnu' # Valeur par défaut pour tout le monde
    
    if 'title' in df.columns:
        df['titre'] = df['title'].fillna('Titre inconnu')
        
    if 'title.fr' in df.columns:
        # On remplace par title.fr là où il y en a, sinon on garde ce qu'on avait trouvé dans 'title'
        df['titre'] = df['title.fr'].fillna(df['titre'])

    # --- Pour la description ---
    df['description'] = 'Pas de description'
    
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('Pas de description')
    if 'description.fr' in df.columns:
        df['description'] = df['description.fr'].fillna(df['description'])
        
    if 'longDescription' in df.columns:
        df['description'] = df['longDescription'].fillna(df['description'])
    if 'longDescription.fr' in df.columns:
        df['description'] = df['longDescription.fr'].fillna(df['description'])



# 2. GESTION DES TEXTES (Sécurisée)
    # # --- Pour le titre ---
    # if 'title.fr' in df.columns:
    #     df['titre'] = df['title.fr'].fillna('Titre inconnu')
    # elif 'title' in df.columns:
    #     df['titre'] = df['title'].fillna('Titre inconnu')
    # else:
    #     df['titre'] = 'Titre inconnu'

    # # --- Pour la description ---
    # # On commence par la description longue
    # if 'longDescription.fr' in df.columns:
    #     df['description'] = df['longDescription.fr']
    # elif 'longDescription' in df.columns:
    #     df['description'] = df['longDescription']
    # else:
    #     df['description'] = None # On crée la colonne vide
        
    # # On bouche les trous avec la description courte si elle existe
    # if 'description.fr' in df.columns:
    #     df['description'] = df['description'].fillna(df['description.fr'])
    # elif 'description' in df.columns:
    #     df['description'] = df['description'].fillna(df['description'])
        
    # # S'il y a toujours des trous, on met notre texte par défaut
    # df['description'] = df['description'].fillna('Pas de description')


# 3. GESTION DES DATES (L'API a déjà fait le tri < 1 an, on extrait juste le texte pour Mistral)
    # if 'firstTiming.begin' in df.columns:
    #     # Si Pandas a aplati le dictionnaire
    #     df['date_debut'] = df['firstTiming.begin']
    # elif 'firstTiming' in df.columns:
    #     # Si c'est resté un dictionnaire
    #     df['date_debut'] = df['firstTiming'].apply(lambda x: x.get('begin') if isinstance(x, dict) else x)
    # elif 'dateRange' in df.columns:
    #     df['date_debut'] = df['dateRange']
    # else:
    #     df['date_debut'] = 'Récemment'

    # 3. GESTION DES DATES (Sécurisée)
    df['date_debut'] = 'Récemment' # On met la valeur par défaut pour tout le monde
    
    if 'dateRange' in df.columns:
        df['date_debut'] = df['dateRange'].fillna(df['date_debut'])
        
    if 'firstTiming' in df.columns:
        # Si c'est resté un dictionnaire, on extrait prudemment
        temp_date = df['firstTiming'].apply(lambda x: x.get('begin') if isinstance(x, dict) else x)
        df['date_debut'] = temp_date.fillna(df['date_debut'])
        
    if 'firstTiming.begin' in df.columns:
        # Si Pandas a aplati le dictionnaire
        df['date_debut'] = df['firstTiming.begin'].fillna(df['date_debut'])

    # 4. LE FILTRAGE SIMPLE
    nb_avant = len(df)
    
    # On supprime juste les erreurs de saisie (sans titre)
    df = df[df['titre'] != 'Titre inconnu']
    
    nb_apres = len(df)
    print(f"🗑️ Filtrage de qualité : {nb_avant - nb_apres} événements supprimés.")
    print(f"✅ Reste final : {len(df)} événements récents conservés.")
    
    # # 5. GESTION DU LIEU (Nom du lieu + Ville)
    # if 'location.name' in df.columns and 'location.city' in df.columns:
    #     # On combine le nom de la salle et la ville : "Le Zénith (Strasbourg)"
    #     df['lieu'] = df['location.name'].fillna('') + " (" + df['location.city'].fillna('') + ")"
    # elif 'location.city' in df.columns:
    #     df['lieu'] = df['location.city']
    # else:
    #     df['lieu'] = "Alsace"


    # 5. GESTION DU LIEU (Nom du lieu + Ville)
    if 'location.name' in df.columns and 'location.city' in df.columns:
        # On utilise 'apply' pour regarder chaque ligne individuellement
        df['lieu'] = df.apply(
            lambda row: f"{row['location.name']} ({row['location.city']})" 
            if pd.notna(row['location.name']) and row['location.name'] != '' 
            else row['location.city'], 
            axis=1
        )
    elif 'location.city' in df.columns:
        df['lieu'] = df['location.city']
    else:
        df['lieu'] = "Alsace"

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
        date_str = str(row.get('date_debut', 'Inconnue')).strip()
        # --- ON RÉCUPÈRE LE LIEU ICI ---
        lieu = str(row.get('lieu', 'Lieu non précisé')).strip()

        # On ajoute le Lieu dans le texte complet
        texte_complet = f"Titre : {titre}\nDate : {date_str}\nLieu : {lieu}\nDescription : {description}".strip()

        if len(texte_complet) < 10:
            continue 

        if len(texte_complet) > 10000:
            texte_complet = texte_complet[:10000] + "..."

        # On l'ajoute aussi dans les métadonnées pour être propre
        metadata = {"titre": titre, "date": date_str, "lieu": lieu}
        documents.append(Document(page_content=texte_complet, metadata=metadata))
        
    print(f"✅ Filtrage terminé : {len(documents)} documents valides prêts pour Mistral.")
    return documents

    
if __name__ == "__main__":
    # Liste des villes pour couvrir l'Alsace
    VILLES_ALSACE = ["Strasbourg", "Colmar", "Mulhouse", "Sélestat", "Haguenau"]
    all_raw_data = []
    all_agenda_uids = set()

    print(f"🔍 Recherche des agendas les plus actifs...")
    for ville in VILLES_ALSACE:
        # On demande les 10 meilleurs agendas par ville
        uids = get_top_agendas_by_location(ville, limit=15) 
        all_agenda_uids.update(uids)

    print(f"✅ {len(all_agenda_uids)} agendas trouvés. Extraction des événements...")

    for uid in all_agenda_uids:
        try:
            # IMPORTANT : on passe search_term=None pour ne pas filtrer par mot-clé
            # On veut TOUT ce que l'agenda propose de récent
            events = fetch_openagenda_events(uid, search_term=None) 
            all_raw_data.extend(events)
            print(f"   -> Agenda {uid} : {len(events)} événements ajoutés.")
        except Exception as e:
            continue
            
    print(f"✅ {len(all_raw_data)} événements bruts récupérés au total.")
    
    print("🧹 Traitement avec Pandas...")
    df_clean = process_and_filter_events(all_raw_data)
    print(f"✅ {len(df_clean)} événements conservés après filtrage.")
    
    print("📦 3. Création des Documents LangChain (1 événement = 1 Document)...")
    documents_initiaux = prepare_documents_by_event(df_clean)
    
    if documents_initiaux:
        print("✂️ 3.5 Découpage (Chunking) des textes longs...")
        # On configure le découpeur : morceaux de 1000 caractères, avec 100 caractères 
        # de chevauchement (overlap) pour ne pas couper un mot ou une idée en deux.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # On applique le découpage à nos documents
        documents_chunkes = text_splitter.split_documents(documents_initiaux)
        print(f"✅ Nous sommes passés de {len(documents_initiaux)} événements à {len(documents_chunkes)} chunks.")

        print("🚀 4. Création de la base de données FAISS (cela peut prendre 1-2 min)...")
        # Attention à bien envoyer les documents "chunkés" à FAISS !
        create_and_save_faiss_index(documents_chunkes)