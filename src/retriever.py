import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

# On charge la clé API
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

def load_faiss_index():
    """Charge l'index FAISS depuis le disque dur."""
    print("📂 Chargement de la base de données FAISS...")
    
    # On doit recréer l'objet d'embedding pour que FAISS sache comment lire la question
    embeddings_model = MistralAIEmbeddings(
        mistral_api_key=api_key, 
        model="mistral-embed"
    )
    
    # On retrouve le chemin exact de notre dossier faiss_index à la racine
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    racine_projet = os.path.dirname(dossier_src)
    dossier_sauvegarde = os.path.join(racine_projet, "faiss_index")
    
    # On charge la base en mémoire
    # Note : allow_dangerous_deserialization=True est une sécurité récente de LangChain. 
    # C'est obligatoire pour lire des fichiers locaux .pkl que l'on a créés nous-mêmes.
    vectorstore = FAISS.load_local(
        dossier_sauvegarde, 
        embeddings_model,
        allow_dangerous_deserialization=True 
    )
    
    print("✅ Base chargée avec succès !")
    return vectorstore

def search_events(vectorstore, query, k=3):
    """Cherche les 'k' événements les plus pertinents pour une question."""
    print(f"\n🔍 Recherche en cours pour : '{query}'\n")
    
    # C'est ici que la magie opère : FAISS trouve les plus proches voisins mathématiques
    resultats = vectorstore.similarity_search(query, k=k)
    
    for i, doc in enumerate(resultats):
        print(f"--- Résultat {i+1} ---")
        print(f"Titre : {doc.metadata.get('titre')}")
        print(f"Date : {doc.metadata.get('date')}")
        # On affiche juste les 150 premiers caractères pour vérifier
        print(f"Extrait : {doc.page_content[:150]}...\n")

if __name__ == "__main__":
    # 1. On charge la base (C'est instantané et on ne repaie pas l'API pour les 449 events !)
    db = load_faiss_index()
    
    # 2. On pose une question en langage naturel
    question = "Quels sont les événements musicaux ou les concerts prévus ?"
    
    # 3. On demande à l'IA de nous sortir le top 3
    search_events(db, question, k=3)