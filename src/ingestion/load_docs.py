from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from src.config import INFO_DIR, PG_DIR, UG_DIR


# Load all .txt files from a folder
def load_text_files(path):
    loader = DirectoryLoader(
        path,
        glob="**/*.txt",
        loader_cls=lambda p: TextLoader(p, encoding="latin-1", autodetect_encoding=True)
    )
    return loader.load()


# Load all .pdf files from a folder
def load_pdf_files(path):
    loader = DirectoryLoader(
        path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# Load everything (simple version)
def load_all_documents():
    info_docs = load_text_files(INFO_DIR)
    pg_docs = load_pdf_files(PG_DIR)
    ug_docs = load_pdf_files(UG_DIR)
    all_documents = info_docs + pg_docs + ug_docs

    return all_documents

