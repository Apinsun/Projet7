import os
import argparse
import logging
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configuration
load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")
SCORE_SIMILARITE = 0.4

def format_docs(docs):
    """Combine les contenus des documents trouvés en une seule chaîne."""
    return "\n\n".join(doc.page_content for doc in docs)

def setup_chatbot():
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key, model="mistral-embed")
    
    # Chemin vers ton index FAISS
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    racine_projet = os.path.dirname(dossier_src)
    dossier_faiss = os.path.join(racine_projet, "faiss_index")
    
    vectorstore = FAISS.load_local(
        dossier_faiss, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    llm = ChatMistralAI(model="mistral-small", mistral_api_key=api_key)

    # 2. PROMPT SYSTÈME
    template = """Tu es un assistant expert en événements à Strasbourg.
    Utilise UNIQUEMENT le contexte suivant pour répondre. Si tu ne sais pas, dis que tu ne trouves rien.
    
    CONTEXTE : {context}
    
    QUESTION : {question}
    
    RÉPONSE :"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # 3. LE RETRIEVER (avec ton seuil de similarité)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": SCORE_SIMILARITE}
    )

    # 4. CONSTRUCTION DE LA CHAÎNE (LCEL)
    # C'est ici qu'on assemble les pièces comme des Lego
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retourne aussi le retriever et le prompt pour permettre l'affichage
    # du payload envoyé au LLM en mode debug
    return rag_chain, retriever, prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debugg", action="store_true", help="Activer le mode debug (affiche le payload envoyé au LLM)")
    parser.add_argument("--debug", action="store_true", help="Alias pour --debugg")
    args = parser.parse_args()

    if args.debugg or args.debug:
        logging.basicConfig(level=logging.DEBUG)

    bot, retriever, prompt = setup_chatbot()
    print("🤖 Chatbot LCEL prêt !")

    while True:
        user_input = input("\nVous : ")
        if user_input.lower() in ["quitter", "exit"]:
            break

        print("\n✨ Mistral réfléchit...")

        # Si on est en mode debug, récupère le contexte et affiche le payload
        if args.debugg or args.debug:
            try:
                docs = retriever.get_relevant_documents(user_input)
                context = format_docs(docs)
            except Exception:
                # Si la méthode spécifique au retriever diffère, on ignore proprement
                context = "<impossible de récupérer les docs (méthode get_relevant_documents introuvable)>"

            debug_payload = {"context": context, "question": user_input}
            print("\n[DEBUG] Payload envoyé au LLM :")
            print(debug_payload)

        # On utilise invoke directement sur la chaîne
        try:
            response = bot.invoke(user_input)
            print(f"\nAssistant : {response}")
        except Exception as e:
            print(f"\n⚠️ Erreur : {e}")
            print("Conseil : Si l'erreur mentionne le score, ton seuil est peut-être trop strict.")
