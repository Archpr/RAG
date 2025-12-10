import sys
import os

# Fix import path so "src" works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.retriever.chroma_retriever import get_chroma_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import LLM_MODEL

# Direct Reranker Import
from sentence_transformers import CrossEncoder

# Global cache for the reranker model to avoid reloading
_RERANKER = None

def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        # Load the model once
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _RERANKER


def safe_retrieve(retriever, query):
    """
    Works for all LangChain versions:
    - New versions use get_relevant_documents() (or invoke in very new ones)
    - Older versions use _get_relevant_documents()
    """
    try:
        # For very new langchain, check invoke first if available, else get_relevant_documents
        if hasattr(retriever, "invoke"):
            return retriever.invoke(query)
        return retriever.get_relevant_documents(query)
    except Exception:
        return retriever._get_relevant_documents(query, run_manager=None)


def run_chain(query: str):
    # 1. Initialize Base Retriever (High Recall)
    # Fetch top 10 documents from Chroma
    base_retriever = get_chroma_retriever(k=10)
    docs = safe_retrieve(base_retriever, query)

    if not docs:
        return "I couldn't find any relevant information to answer your question."

    # 2. Rerank with CrossEncoder
    try:
        reranker = get_reranker()
        
        # Prepare pairs: [ [query, doc_text], ... ]
        pairs = [[query, doc.page_content] for doc in docs]
        
        # Predict scores
        scores = reranker.predict(pairs)
        
        # Combine docs with scores
        scored_docs = list(zip(docs, scores))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 3
        top_docs = [doc for doc, score in scored_docs[:3]]
        
    except Exception as e:
        # Fallback if reranker fails (e.g. model load error)
        print(f"Reranking failed: {e}. Using top 3 from raw retrieval.")
        top_docs = docs[:3]

    # Combine all context
    context = "\n\n".join(doc.page_content for doc in top_docs)

    # Load LLM from config
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)

    prompt = f"""
You are an expert RAG assistant. Use ONLY the given context to answer.

Context:
{context}

Question: {query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content  # return only text

