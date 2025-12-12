from langchain_chroma import Chroma
from src.config import CHROMA_PATH
from src.embeddings.gemini_embedding import embeddings  # embedding object

def store_in_chroma(chunks):
    """
    Store document chunks in ChromaDB using the Gemini embedding model.

    Args:
        chunks (list): List of Document objects

    Returns:
        Chroma: Vectorstore object
    """
    print("Storing embeddings in ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,  # embedding object used internally
        persist_directory=CHROMA_PATH
    )

    # Remove this line:
    # vectorstore.persist()

    print(f"ChromaDB persisted at: {CHROMA_PATH}")
    print(f"Total documents in Chroma: {vectorstore._collection.count()}")

    return vectorstore
