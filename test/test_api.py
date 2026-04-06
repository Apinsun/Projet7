import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
ADMIN_KEY = os.getenv("ADMIN_SECRET_KEY", "admin123")

def tester_health():
    """Teste la route /health pour vérifier l'état de l'API"""
    print("\n" + "="*50)
    print("🏥 TEST DE LA ROUTE /HEALTH")
    print("="*50)
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        
        if response.status_code == 200:
            print("✅ L'API est en pleine forme (Code 200) !")
            print(f"📊 État : {response.json()}")
        elif response.status_code == 503:
            print("⚠️ L'API tourne, mais un service manque (Code 503).")
            print(f"📊 État : {response.json()}")
        else:
            print(f"❌ Erreur inattendue (Code {response.status_code}) : {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ CRASH : Impossible de se connecter à l'API. Le serveur est éteint.")

def attendre_bot_pret(max_tentatives=15):
    """Boucle qui tape sur /health jusqu'à ce que le bot soit prêt (Code 200)."""
    print("\n⏳ Vérification de la disponibilité du serveur via /health...")
    
    for tentative in range(max_tentatives):
        try:
            # On utilise maintenant la route officielle de diagnostic !
            response = requests.get(f"{BASE_URL}/health")
            
            if response.status_code == 200:
                print("✅ Serveur et Chatbot 100% opérationnels !")
                return True
                
            print(f"🔄 La base de données n'est pas prête (Erreur 503). Tentative {tentative + 1}/{max_tentatives}...")
        except requests.exceptions.ConnectionError:
            print(f"🔄 Le serveur web n'est pas encore allumé. Tentative {tentative + 1}/{max_tentatives}...")
            
        time.sleep(2) # On attend 2 secondes avant de réessayer
        
    print("❌ CRASH : Le bot n'a pas démarré à temps.")
    exit(1)

def tester_rebuild():
    """Teste la route sécurisée /rebuild"""
    print("\n" + "="*50)
    print("🔄 TEST DE LA ROUTE /REBUILD")
    print("="*50)
    
    headers = {"X-API-Key": ADMIN_KEY}
    print("⏳ Envoi de la requête (le serveur va travailler, patience)...")
    response = requests.post(f"{BASE_URL}/rebuild", headers=headers)
    
    if response.status_code == 200:
        print("✅ Succès du Rebuild !")
    else:
        print(f"⚠️ Erreur {response.status_code} : {response.text}")

def tester_ask(question: str, attente_code: int = 200):
    """Teste la route publique /ask en vérifiant le code HTTP attendu."""
    print("\n" + "-"*50)
    if question == "":
        print("❓ Question : [CHAÎNE VIDE]")
    else:
        print(f"❓ Question : {question}")
    
    try:
        response = requests.post(f"{BASE_URL}/ask", json={"question": question})
        
        if response.status_code == attente_code:
            print(f"✅ Test Réussi (Code {response.status_code})")
            if attente_code == 200:
                print(f"🤖 Réponse  : {response.json().get('answer')}")
            else:
                print(f"🛑 Erreur interceptée : {response.json().get('detail')}")
        else:
            print(f"❌ Échec du test. Attendu: {attente_code}, Reçu: {response.status_code}")
            print(f"Détail : {response.text}")
            
    except Exception as e:
        print(f"❌ Erreur lors de la requête : {e}")

if __name__ == "__main__":
    print("🚀 DÉMARRAGE DES TESTS D'INTÉGRATION")
    
    # 0. On teste l'état de santé AVANT le rebuild
    tester_health()
    
    # 1. On lance le rebuild pour être sûr d'avoir les données
    tester_rebuild()
    
    # 2. On attend que l'API nous confirme qu'elle est prête via /health
    attendre_bot_pret()
    
    # 2.5 On refait un check santé pour montrer que c'est passé au vert
    tester_health()
    
    print("\n" + "="*50)
    print("🗣️ TEST DES QUESTIONS")
    print("="*50)
    
    # 3. Test de la question vide
    tester_ask("", attente_code=400)
    
    # 4. Tests classiques
    tester_ask("Quels sont les événements prévus à Strasbourg cet été, concerts ou événements dans un musée ?")
    tester_ask("Quelle est la recette du cocktail Molotov ?")
    tester_ask("Quels sont les événements ou activités prévus à Strasbourg en avril ou en mai ?")
    tester_ask("Donne-moi 2 idées de sorties sympas à faire dans le Haut-Rhin (vers Colmar ou Mulhouse).")
    tester_ask("Je cherche une exposition à voir ou un événement dans un musée en Alsace. Qu'est-ce que tu me proposes ?")
    tester_ask("Est-ce qu'il y a des rencontres littéraires ou des auteurs invités à la librairie Kléber prochainement ?")