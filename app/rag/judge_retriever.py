"""
Judge Retriever Module.

Provides optional access to WebMD and Mayo Clinic knowledge bases for the judge.
The judge can optionally use this to verify expert arguments with source knowledge.
"""
from app.rag.smart_hybrid_retriever import get_smart_hybrid_retriever
from app.rag.metadata_filter import determine_retrieval_strategy
from config.settings import settings
from typing import List, Tuple


def retrieve_judge_knowledge(
    medical_note: str,
    expert_a_args: List[str],
    expert_b_args: List[str],
    max_docs: int = 10
) -> Tuple[List, List]:
    """
    Retrieve knowledge from both Mayo Clinic and WebMD for the judge.

    This allows the judge to cross-reference expert claims with source knowledge.

    Args:
        medical_note: The medical note being analyzed
        expert_a_args: List of Expert A (Mayo Clinic) arguments
        expert_b_args: List of Expert B (WebMD) arguments
        max_docs: Maximum number of documents to retrieve per source

    Returns:
        Tuple of (mayo_clinic_docs, webmd_docs)
    """
    if not settings.USE_JUDGE_RETRIEVER:
        return [], []

    # Get smart hybrid retriever
    smart_hybrid_retriever = get_smart_hybrid_retriever()

    # Determine optimal retrieval strategy based on note content
    primary_cat, secondary_cat = determine_retrieval_strategy(medical_note)

    # Create enhanced query incorporating expert arguments
    judge_query = f"""Medical Note: {medical_note}

Expert A Claims: {' '.join(expert_a_args)}

Expert B Claims: {' '.join(expert_b_args)}

Verify these claims and provide authoritative medical knowledge."""

    print(f"\n[Judge Retriever] Fetching knowledge from both sources...")
    print(f"  Primary category: {primary_cat}, Secondary: {secondary_cat}")

    # Retrieve from Mayo Clinic (Expert A's source)
    mayo_docs = smart_hybrid_retriever.retrieve_with_decomposition(
        note=judge_query,
        expert="A",  # Mayo Clinic
        k_per_query=2,
        max_total=max_docs // 2,
        use_dense=True,
        use_sparse=True,
        use_online=False  # Use only local knowledge for judge
    )

    # Retrieve from WebMD (Expert B's source)
    webmd_docs = smart_hybrid_retriever.retrieve_with_decomposition(
        note=judge_query,
        expert="B",  # WebMD
        k_per_query=2,
        max_total=max_docs // 2,
        use_dense=True,
        use_sparse=True,
        use_online=False  # Use only local knowledge for judge
    )

    print(f"  Retrieved {len(mayo_docs)} Mayo Clinic docs, {len(webmd_docs)} WebMD docs")

    return mayo_docs, webmd_docs


def format_judge_knowledge(mayo_docs: List, webmd_docs: List) -> str:
    """
    Format retrieved knowledge for judge's context.

    Args:
        mayo_docs: Mayo Clinic documents
        webmd_docs: WebMD documents

    Returns:
        Formatted string with retrieved knowledge
    """
    if not mayo_docs and not webmd_docs:
        return ""

    parts = ["\n=== Retrieved Medical Knowledge for Verification ==="]

    if mayo_docs:
        parts.append("\n--- Mayo Clinic Sources ---")
        for i, doc in enumerate(mayo_docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            parts.append(f"\nMayo Source {i}:\n{content}\n")

    if webmd_docs:
        parts.append("\n--- WebMD Sources ---")
        for i, doc in enumerate(webmd_docs, 1):
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            parts.append(f"\nWebMD Source {i}:\n{content}\n")

    parts.append("\nUse this knowledge to verify expert claims and make your decision.\n")

    return "\n".join(parts)
