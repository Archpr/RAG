import sys
import os

# Fix path so "src" works
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

try:
    from src.ragchain.rag_chain import run_chain
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# Ensure stdout handles special characters (like Rupee symbol)
sys.stdout.reconfigure(encoding='utf-8')

def test_queries():
    queries = [
        "What is the application fee for SC/ST candidates?",
        "What documents are needed for admission?"
    ]

    print("Running RAG Verification...\n")
    
    for q in queries:
        print(f"Question: {q}")
        try:
            answer = run_chain(q)
            print("Answer:")
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_queries()
