"""
main.py
---------------------------------
Entry point for preprocessing pipeline.
Loads documents, splits them, builds embeddings,
and persists a Chroma vector store.
"""

import os
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from build_vectorstore import initialize_vectorstore

# Environment/config setup
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PERSIST_DIR = Path(os.getenv("PERSIST_DIR", "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def main():
    print("Starting preprocessing pipeline...")
    print(f"Data directory: {DATA_DIR}")
    print(f"Persist directory: {PERSIST_DIR}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    vectordb = initialize_vectorstore()

    print("Preprocessing complete! Vector store saved to:", PERSIST_DIR)

    try:
        print("Vector count:", vectordb._collection.count())  # private but handy
    except Exception as e:
        print("Count check failed (non-fatal):", e)

    # Simple search
    query = "What are common adverse effects?"  # pick a question your data can answer
    docs = vectordb.similarity_search(query, k=4)
    for i, d in enumerate(docs, 1):
        print(f"\n[{i}] {d.metadata.get('filename','<no name>')} — {d.metadata.get('source','')}")
        print(d.page_content[:400], "...")

if __name__ == "__main__":
    main()
