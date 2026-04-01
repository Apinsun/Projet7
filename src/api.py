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

# Chargement du bot au démarrage du serveur
print("⏳ Chargement du modèle Mistral et de l'index FAISS...")
# On utilise _, _ pour ignorer le retriever et le prompt dont l'API n'a pas besoin
bot, _, _ = setup_chatbot() 
print("✅ Bot prêt à recevoir des requêtes HTTP !")

# --- SÉCURITÉ POUR LA ROUTE /REBUILD ---
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
# On récupère le mot de passe depuis le .env, ou on met une valeur par défaut
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "admin123")

def verify_admin_key(api_key: str = Security(api_key_header)):
    """Vérifie que la clé fournie dans le header est correcte."""
    if api_key != ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Accès refusé. Clé administrateur invalide.")
    return api_key


# --- MODÈLE DE DONNÉES (Pydantic) ---
class QuestionRequest(BaseModel):
    question: str


# --- ENDPOINT 1 : POSER UNE QUESTION (Public) ---
@app.post("/ask", summary="Poser une question au chatbot")
async def ask_question(request: QuestionRequest):
    """Reçoit une question et renvoie la réponse générée par le système RAG."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="La question ne peut pas être vide.")
    
    try:
        # On passe la question au bot LangChain (LCEL)
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