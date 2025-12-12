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


def format_docs_with_sources(docs):
    """Format documents with clear source labels for the LLM."""
    formatted_text = ""
    for i, doc in enumerate(docs, 1):
        # clean source path to just filename
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        content = doc.page_content.replace("\n", " ").strip()
        
        formatted_text += f"[Source {i}]: {source}{page_info}\n{content}\n\n"
    return formatted_text


def run_chain(query: str):
    # 1. Initialize Base Retriever (High Recall)
    # Fetch top 10 documents from Chroma
    base_retriever = get_chroma_retriever(k=10)
    docs = safe_retrieve(base_retriever, query)

    if not docs:
        return "I couldn't find any relevant documents to answer your question."

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
        
        # Take top 5 (increased from 3 for better context)
        top_docs = [doc for doc, score in scored_docs[:5]]
        
    except Exception as e:
        # Fallback if reranker fails
        print(f"Reranking failed: {e}. Using top 3 from raw retrieval.")
        top_docs = docs[:3]

    # Combine all context with sources
    context_text = format_docs_with_sources(top_docs)

    # Load LLM from config
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)

    # 3. Advanced Prompt Engineering
    system_prompt = f"""
You are an expert Admissions Counselor for Kerala University. Your goal is to provide accurate, helpful, and professional assistance to students.

Instructions:
1. **Analyze the Context**: Read the provided "Context Documents" carefully.
2. **Answer Based ONLY on Context**: Do not use outside knowledge. If the answer is not in the documents, say "I don't have that information in my current documents."
3. ** Cite Sources**: When stating facts, reference the source ID (e.g., [Source 1]).
4. **Be Clear and Structured**: Use bullet points for steps or lists. Keep the tone welcoming but precise.
5. **Reasoning**: Briefly think about which document best answers the specific question.

Context Documents:
{context_text}

User Question: {query}

Answer:
"""

    response = llm.invoke(system_prompt)
    return response.content  # return only text

