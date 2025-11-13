"""
preprocessing/build_vectorstore.py

Recursively load text/markdown files, chunk them, embed with Ollama, and
persist a Chroma vector store.

Usage:
  python preprocessing/build_vectorstore.py \
    --data-dir data/drug_supplement \
    --persist-dir chroma_db \
    --chunk-size 800 \
    --chunk-overlap 200 \
    --rebuild false
"""

import os
import pathlib

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from load_documents import discover_paths, load_documents_from_paths
from split_text import recursive_split
from dotenv import load_dotenv

load_dotenv()
# Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "llama2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def initialize_vectorstore(
    data_dir="data/drug_supplement",
    persist_dir="chroma_db",
    allowed_ext=(".md", ".txt"),
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    reuse_if_exists: bool = False,):
    print("Initializing vector store...")
    print(EMBEDDING_MODEL)
    # Embeddings
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    # Reuse an exsisting DB
    if reuse_if_exists and pathlib.Path(persist_dir).exists():
        print(f"Reusing existing Chroma at: {persist_dir}")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Find and load all the documents in the directory
    paths = discover_paths(pathlib.Path(data_dir), allowed_ext=allowed_ext)
    docs = load_documents_from_paths(paths)

    # Split into chunks
    chunks = recursive_split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Split into {len(chunks)} chunks")

    # Create embeddings + ChromaDB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
        )
    vectordb.persist()
    print(f"Vector store persisted at: {persist_dir}")

    return vectordb

