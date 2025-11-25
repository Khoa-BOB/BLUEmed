#!/usr/bin/env python3
"""
Test script for RAG auto-build functionality.

This script demonstrates how the vector databases are automatically
built if they don't exist, using Google Gemini embeddings.
"""

from config.settings import settings

print("="*60)
print("Testing RAG Auto-Build with Google Gemini Embeddings")
print("="*60)
print()

# Show configuration
print("Configuration:")
print(f"  GOOGLE_API_KEY: {'✅ Set' if settings.GOOGLE_API_KEY else '❌ Not set'}")
print(f"  EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
print(f"  PERSIST_DIR: {settings.PERSIST_DIR}")
print(f"  USE_RETRIEVER: {settings.USE_RETRIEVER}")
print()

if not settings.GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not set in .env file")
    print("   Get your API key from: https://aistudio.google.com/app/apikey")
    exit(1)

# Import retriever (this will auto-build if needed)
print("Initializing retriever...")
print("(This will automatically build vector databases if they don't exist)")
print()

from app.rag.chroma_retriever import get_retriever

# Get retriever instance (auto_build=True by default)
retriever = get_retriever()

print()
print("="*60)
print("Testing Retrieval")
print("="*60)
print()

# Test query
test_query = "What is gonorrhea and how is it treated?"

print(f"Query: {test_query}")
print()

# Test Mayo Clinic retrieval (Expert A)
print("Expert A (Mayo Clinic) Results:")
print("-" * 60)
mayo_results = retriever.retrieve_mayo(test_query, k=2)
for i, doc in enumerate(mayo_results, 1):
    print(f"{i}. {doc.page_content[:200]}...")
    print()

# Test WebMD retrieval (Expert B)
print("Expert B (WebMD) Results:")
print("-" * 60)
webmd_results = retriever.retrieve_webmd(test_query, k=2)
for i, doc in enumerate(webmd_results, 1):
    print(f"{i}. {doc.page_content[:200]}...")
    print()

print("="*60)
print("✅ Test completed successfully!")
print("="*60)
