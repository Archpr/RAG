from src.ingestion import load_all_documents
docs = load_all_documents()
print("Total documents loaded:", len(docs))