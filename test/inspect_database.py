import os
import sys
from dotenv import load_dotenv

# On ajoute le dossier src au chemin pour pouvoir importer ton chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from chatbot import setup_chatbot

load_dotenv()

def inspect_context_for_question():
    print("⏳ Chargement du moteur de recherche vectoriel (FAISS)...")
    # On récupère le retriever directement depuis ton fichier chatbot
    _, retriever, _ = setup_chatbot()
    print("✅ Moteur prêt !\n")
    
    while True:
        question = input("❓ Pose une question test (ou 'quitter' pour arrêter) : ")
        if question.lower() in ['quitter', 'exit', 'q']:
            break
            
        print("\n🔍 Recherche des documents dans FAISS...")
        # On interroge FAISS exactement comme le fera Mistral
        docs = retriever.invoke(question)
        
        if not docs:
            print("❌ Aucun document trouvé (le contexte envoyé au LLM sera VIDE).")
            print("👉 Si tu utilises cette question pour tes tests, la 'ground_truth' doit être la phrase de refus de ton prompt !")
        else:
            print(f"✅ {len(docs)} document(s) trouvé(s) ! Voici ce que Mistral va lire :\n")
            for i, doc in enumerate(docs):
                print(f"--- CHUNK {i+1} ---")
                print(doc.page_content)
                print("-----------------\n")

if __name__ == "__main__":
    inspect_context_for_question()