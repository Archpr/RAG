import sys
import os

# Fix path so "src" works
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.ragchain.rag_chain import run_chain   # we already wrote this function


def main():
    print("\n🔍 RAG Query Tester\n")

    while True:
        query = input("Enter your question (or type 'exit'): ")

        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print("\n🧠 Retrieving and generating answer...\n")

        try:
            answer = run_chain(query)
            print("📘 Answer:\n")
            print(answer)bcom 
        except Exception as e:
            print(f"❌ Error: {e}")

        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()
