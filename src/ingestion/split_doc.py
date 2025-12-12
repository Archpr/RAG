# split.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


# Create a reusable splitter object
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)


def split_documents(documents):
    """Split documents into clean text chunks."""
    chunks = text_splitter.split_documents(documents)
    return chunks
