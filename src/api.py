import os
import subprocess
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# Import de la fonction setup_chatbot depuis ton fichier chatbot.py
from src.chatbot import setup_chatbot

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API RAG Événements Alsace",
    description="Une API REST pour interroger notre base vectorielle d'événements culturels.",
    version="1.0"
)

# --- CHARGEMENT TOLÉRANT AU DÉMARRAGE ---
bot = None
try:
    print("⏳ Chargement du modèle Mistral et de l'index FAISS...")
    bot, _, _ = setup_chatbot()
    print("✅ Bot prêt à recevoir des requêtes HTTP !")
except Exception as e:
    print(f"⚠️ AVERTISSEMENT : Impossible de charger la base FAISS ({e}).")
    print("👉 L'API est démarrée, mais vous devez appeler /rebuild pour initialiser le bot.")


# --- SÉCURITÉ ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "admin123")

def verify_admin_key(api_key: str = Security(api_key_header)):
    if api_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Accès refusé.")
    return api_key

class QuestionRequest(BaseModel):
    question: str

# --- ENDPOINT 1 : POSER UNE QUESTION ---
@app.post("/ask", summary="Poser une question au chatbot")
async def ask_question(request: QuestionRequest):
    # On vérifie d'abord si le bot a bien été chargé !
    if bot is None:
         raise HTTPException(
             status_code=503, 
             detail="Le chatbot n'est pas initialisé (Base FAISS manquante). Lancez d'abord un /rebuild."
         )

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
    
    try:
        reponse = bot.invoke(request.question)
        return {"question": request.question, "answer": reponse}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne du RAG : {str(e)}")


# --- ENDPOINT 2 : RECONSTRUIRE LA BASE (Privé/Sécurisé) ---
@app.post("/rebuild", dependencies=[Depends(verify_admin_key)], summary="Reconstruire la base FAISS")
async def rebuild_database():
    """Relance le script d'ingestion et recharge le bot en mémoire."""
    try:
        print("🔄 Lancement de la mise à jour de la base de données...")
        # Chemin absolu vers data_ingestion.py
        script_path = os.path.join(os.path.dirname(__file__), "data_ingestion.py")
        
        # Exécution du script comme dans le terminal
        subprocess.run(["poetry", "run", "python", script_path], check=True)
        
        # On recharge le bot avec le nouvel index FAISS
        global bot
        bot, _, _ = setup_chatbot()
        
        return {"status": "success", "message": "Base de données mise à jour et rechargée !"}
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la mise à jour : {str(e)}")