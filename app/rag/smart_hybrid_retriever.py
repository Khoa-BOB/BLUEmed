"""
Smart Hybrid Retriever combining query decomposition with hybrid search.

This combines the best of both:
1. Smart query decomposition from smart_retriever.py
2. Hybrid search (dense + sparse + online) from hybrid_retriever.py
"""
from typing import List, Dict, Set
from langchain_core.documents import Document
from app.rag.hybrid_retriever import HybridMedicalRetriever, get_hybrid_retriever
from app.rag.smart_retriever import SmartMedicalRetriever


class SmartHybridRetriever:
    """
    Combines smart query decomposition with hybrid retrieval.

    Flow:
    1. Decompose medical note into focused queries (smart retriever)
    2. For each query, perform hybrid search (dense + sparse + online)
    3. Combine and deduplicate results
    """

    def __init__(self, hybrid_retriever: HybridMedicalRetriever):
        """
        Initialize smart hybrid retriever.

        Args:
            hybrid_retriever: Base hybrid retriever
        """
        self.hybrid_retriever = hybrid_retriever

        # Reuse query decomposition logic from smart retriever
        from app.rag.smart_retriever import SmartMedicalRetriever
        dummy_retriever = None  # We won't use the base retriever
        self.smart_decomposer = SmartMedicalRetriever(dummy_retriever)

    def retrieve_with_decomposition(
        self,
        note: str,
        expert: str,
        k_per_query: int = 2,
        max_total: int = 5,
        filter_category: str = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_online: bool = True,
        max_online_queries: int = 1  # Limit online searches to first N queries
    ) -> List[Document]:
        """
        Retrieve documents using query decomposition + hybrid search.

        Args:
            note: Medical note
            expert: "A" for Mayo Clinic, "B" for WebMD
            k_per_query: Documents to retrieve per query
            max_total: Maximum total documents to return
            filter_category: Optional category filter
            use_dense: Whether to use dense vector search
            use_sparse: Whether to use sparse BM25 search
            use_online: Whether to use online web search
            max_online_queries: Maximum number of queries to search online (to reduce rate limits)

        Returns:
            List of relevant documents (deduplicated)
        """
        # Decompose into focused queries
        queries = self.smart_decomposer.decompose_query(note)

        expert_name = "Mayo Clinic" if expert == "A" else "WebMD"
        filter_msg = f" (filtered: {filter_category})" if filter_category else ""

        methods = []
        if use_dense:
            methods.append("dense")
        if use_sparse:
            methods.append("sparse")
        if use_online:
            methods.append("online")
        methods_str = "+".join(methods)

        print(f"\n[Smart Hybrid Retriever - {expert_name}]")
        print(f"  Decomposed into {len(queries)} queries")
        print(f"  Using: {methods_str} search{filter_msg}")

        for i, q in enumerate(queries[:5], 1):  # Show first 5
            print(f"    {i}. {q[:80]}...")

        # Retrieve for each query using hybrid search
        all_docs = []
        seen_contents: Set[str] = set()

        for idx, query in enumerate(queries):
            # Only use online search for first max_online_queries to reduce rate limits
            use_online_for_this_query = use_online and (idx < max_online_queries)

            docs = self.hybrid_retriever.hybrid_retrieve(
                query=query,
                expert=expert,
                k=k_per_query,
                filter_category=filter_category,
                use_dense=use_dense,
                use_sparse=use_sparse,
                use_online=use_online_for_this_query
            )

            # Deduplicate based on content
            for doc in docs:
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                fingerprint = content[:200].strip()

                if fingerprint not in seen_contents:
                    seen_contents.add(fingerprint)
                    all_docs.append(doc)

                    # Stop if we have enough
                    if len(all_docs) >= max_total:
                        break

            if len(all_docs) >= max_total:
                break

        print(f"[Smart Hybrid Retriever - {expert_name}] Retrieved {len(all_docs)} unique documents\n")
        return all_docs[:max_total]


# Global smart hybrid retriever instance
_smart_hybrid_retriever = None


def get_smart_hybrid_retriever(
    persist_dir: str = None,
    embedding_model: str = None,
    auto_build: bool = True,
    data_dir: str = "BlueMed_data",
    use_online: bool = True,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.3,
    online_weight: float = 0.2
) -> SmartHybridRetriever:
    """
    Get or create global smart hybrid retriever instance.

    Args:
        persist_dir: Directory containing vector databases
        embedding_model: Embedding model to use
        auto_build: If True, automatically build databases if they don't exist
        data_dir: Directory containing source documents
        use_online: Whether to enable online web search
        dense_weight: Weight for dense retrieval
        sparse_weight: Weight for sparse retrieval
        online_weight: Weight for online search

    Returns:
        SmartHybridRetriever instance
    """
    global _smart_hybrid_retriever
    if _smart_hybrid_retriever is None:
        # Get hybrid retriever
        hybrid_retriever = get_hybrid_retriever(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            auto_build=auto_build,
            data_dir=data_dir,
            use_online=use_online,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            online_weight=online_weight
        )

        # Create smart hybrid retriever
        _smart_hybrid_retriever = SmartHybridRetriever(hybrid_retriever)

    return _smart_hybrid_retriever
