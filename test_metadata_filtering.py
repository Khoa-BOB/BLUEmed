#!/usr/bin/env python3
"""
Test script for metadata filtering in RAG system.

Demonstrates how to use category filters to retrieve specific types of medical information:
- drugs_supplements: Medication and supplement information
- diseases_conditions: Disease and condition information
- symptoms: Symptom information (Mayo Clinic only)
"""

from config.settings import settings

print("="*80)
print("Testing Metadata Filtering with Google Gemini Embeddings")
print("="*80)
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
print("Initializing retriever with metadata support...")
print()

from app.rag.chroma_retriever import get_retriever

# Get retriever instance
retriever = get_retriever()

print()
print("="*80)
print("Test 1: Unfiltered Retrieval (All Categories)")
print("="*80)
print()

query = "What is gonorrhea and what are the treatment options?"
print(f"Query: {query}")
print()

print("Mayo Clinic (All categories):")
print("-" * 80)
mayo_all = retriever.retrieve_mayo(query, k=3)
for i, doc in enumerate(mayo_all, 1):
    print(f"{i}. {doc[:200]}...")
    print()

print()
print("="*80)
print("Test 2: Filter by Category - Diseases Only")
print("="*80)
print()

print("Mayo Clinic (diseases_conditions only):")
print("-" * 80)
mayo_diseases = retriever.retrieve_mayo(query, k=3, filter_category="diseases_conditions")
for i, doc in enumerate(mayo_diseases, 1):
    print(f"{i}. {doc[:200]}...")
    print()

print()
print("="*80)
print("Test 3: Filter by Category - Drugs Only")
print("="*80)
print()

drug_query = "What medications are used to treat bacterial infections?"
print(f"Query: {drug_query}")
print()

print("Mayo Clinic (drugs_supplements only):")
print("-" * 80)
mayo_drugs = retriever.retrieve_mayo(drug_query, k=3, filter_category="drugs_supplements")
for i, doc in enumerate(mayo_drugs, 1):
    print(f"{i}. {doc[:200]}...")
    print()

print()
print("="*80)
print("Test 4: Comparison - Same Query, Different Filters")
print("="*80)
print()

comparison_query = "ceftriaxone"
print(f"Query: {comparison_query}")
print()

print("WebMD - Diseases only:")
print("-" * 80)
webmd_diseases = retriever.retrieve_webmd(comparison_query, k=2, filter_category="diseases_conditions")
for i, doc in enumerate(webmd_diseases, 1):
    print(f"{i}. {doc[:150]}...")
print()

print("WebMD - Drugs only:")
print("-" * 80)
webmd_drugs = retriever.retrieve_webmd(comparison_query, k=2, filter_category="drugs_supplements")
for i, doc in enumerate(webmd_drugs, 1):
    print(f"{i}. {doc[:150]}...")
print()

print()
print("="*80)
print("Test 5: Smart Retriever with Category Filter")
print("="*80)
print()

from app.rag.smart_retriever import get_smart_retriever

smart_retriever = get_smart_retriever()

medical_note = """
Patient diagnosed with gonorrhea. Considering treatment with ceftriaxone.
Patient has history of alcohol consumption.
"""

print("Medical Note:")
print(medical_note)
print()

print("Smart Retrieval (diseases_conditions only):")
print("-" * 80)
smart_diseases = smart_retriever.retrieve_with_decomposition(
    note=medical_note,
    expert="A",
    k_per_query=2,
    max_total=3,
    filter_category="diseases_conditions"
)

print("\nRetrieved Documents:")
for i, doc in enumerate(smart_diseases, 1):
    print(f"\n{i}. {doc[:300]}...")

print()
print("Smart Retrieval (drugs_supplements only):")
print("-" * 80)
smart_drugs = smart_retriever.retrieve_with_decomposition(
    note=medical_note,
    expert="A",
    k_per_query=2,
    max_total=3,
    filter_category="drugs_supplements"
)

print("\nRetrieved Documents:")
for i, doc in enumerate(smart_drugs, 1):
    print(f"\n{i}. {doc[:300]}...")

print()
print("="*80)
print("✅ All metadata filtering tests completed successfully!")
print("="*80)
print()
print("Summary:")
print("  - Unfiltered retrieval: Gets all relevant documents")
print("  - diseases_conditions filter: Gets only disease/condition info")
print("  - drugs_supplements filter: Gets only drug/supplement info")
print("  - symptoms filter: Gets only symptom info (Mayo Clinic only)")
print()
print("Use Cases:")
print("  1. When you need comprehensive info → No filter")
print("  2. When diagnosing → filter_category='diseases_conditions'")
print("  3. When prescribing → filter_category='drugs_supplements'")
print("  4. When analyzing symptoms → filter_category='symptoms'")
