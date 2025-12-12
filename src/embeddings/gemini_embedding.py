import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import CHROMA_PATH, EMBEDDING_MODEL


load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env!")

# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


