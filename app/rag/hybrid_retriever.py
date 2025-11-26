"""
Hybrid RAG retriever combining dense, sparse, and online search.

This retriever implements a multi-stage retrieval strategy:
1. Dense retrieval: Vector similarity search (existing Chroma)
2. Sparse retrieval: BM25 keyword matching
3. Online retrieval: Web search from Mayo Clinic / WebMD
4. Fusion: Reciprocal Rank Fusion (RRF) to combine results
"""
from typing import List, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from app.rag.chroma_retriever import MedicalKnowledgeRetriever
from app.rag.online_search import OnlineMedicalSearch


class HybridMedicalRetriever:
    """
    Hybrid retriever combining dense vector search, sparse BM25, and online search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from different retrieval methods.
    """

    def __init__(
        self,
        dense_retriever: MedicalKnowledgeRetriever,
        use_online: bool = True,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.3,
        online_weight: float = 0.2,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.

        Args:
            dense_retriever: Chroma-based dense retriever
            use_online: Whether to include online web search
            dense_weight: Weight for dense retrieval scores
            sparse_weight: Weight for sparse BM25 scores
            online_weight: Weight for online search results
            rrf_k: RRF constant (typically 60)
        """
        self.dense_retriever = dense_retriever
        self.use_online = use_online
        self.online_search = OnlineMedicalSearch() if use_online else None

        # Weights for combining scores
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.online_weight = online_weight
        self.rrf_k = rrf_k

        # BM25 indices (built lazily per collection)
        self.mayo_bm25 = None
        self.webmd_bm25 = None
        self.mayo_docs_cache = []
        self.webmd_docs_cache = []

    def _build_bm25_index(self, expert: str):
        """
        Build BM25 index for a specific expert's collection.

        Args:
            expert: "A" for Mayo Clinic, "B" for WebMD
        """
        print(f"Building BM25 index for Expert {expert}...")

        if expert == "A":
            if self.mayo_bm25 is not None:
                return  # Already built

            db = self.dense_retriever.mayo_db
            if db is None:
                print("Mayo Clinic database not available for BM25 indexing")
                return

            # Get all documents from collection
            collection = db._collection
            results = collection.get(include=['documents', 'metadatas'])

            # Cache documents
            self.mayo_docs_cache = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]

            # Tokenize for BM25
            tokenized_docs = [doc.page_content.lower().split() for doc in self.mayo_docs_cache]
            self.mayo_bm25 = BM25Okapi(tokenized_docs)
            print(f"✅ Built Mayo Clinic BM25 index with {len(self.mayo_docs_cache)} documents")

        elif expert == "B":
            if self.webmd_bm25 is not None:
                return  # Already built

            db = self.dense_retriever.webmd_db
            if db is None:
                print("WebMD database not available for BM25 indexing")
                return

            # Get all documents from collection
            collection = db._collection
            results = collection.get(include=['documents', 'metadatas'])

            # Cache documents
            self.webmd_docs_cache = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(results['documents'], results['metadatas'])
            ]

            # Tokenize for BM25
            tokenized_docs = [doc.page_content.lower().split() for doc in self.webmd_docs_cache]
            self.webmd_bm25 = BM25Okapi(tokenized_docs)
            print(f"✅ Built WebMD BM25 index with {len(self.webmd_docs_cache)} documents")

    def _sparse_search(
        self,
        query: str,
        expert: str,
        k: int = 5,
        filter_category: str = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform BM25 sparse retrieval.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of documents to retrieve
            filter_category: Optional category filter

        Returns:
            List of (Document, score) tuples
        """
        # Ensure BM25 index is built
        self._build_bm25_index(expert)

        # Get BM25 index and doc cache
        if expert == "A":
            bm25 = self.mayo_bm25
            docs_cache = self.mayo_docs_cache
        else:
            bm25 = self.webmd_bm25
            docs_cache = self.webmd_docs_cache

        if bm25 is None or not docs_cache:
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = bm25.get_scores(tokenized_query)

        # Filter by category if specified
        if filter_category:
            filtered_indices = [
                i for i, doc in enumerate(docs_cache)
                if doc.metadata.get('category') == filter_category
            ]
            filtered_scores = [(i, scores[i]) for i in filtered_indices]
        else:
            filtered_scores = list(enumerate(scores))

        # Sort by score and get top k
        top_indices = sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:k]

        # Return documents with scores
        return [(docs_cache[idx], score) for idx, score in top_indices]

    def _online_search(
        self,
        query: str,
        expert: str,
        k: int = 3
    ) -> List[Tuple[Document, float]]:
        """
        Perform online web search using DuckDuckGo.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of results to fetch

        Returns:
            List of (Document, score) tuples
        """
        if not self.use_online or self.online_search is None:
            return []

        try:
            # Perform online search
            docs = self.online_search.search_for_expert(query, expert, k)

            # Assign scores based on ranking (higher rank = higher score)
            scored_docs = []
            for i, doc in enumerate(docs):
                # Score decreases with rank: 1st result gets 1.0, 2nd gets 0.8, etc.
                score = 1.0 / (i + 1)
                scored_docs.append((doc, score))

            return scored_docs

        except Exception as e:
            print(f"✗ Online search error: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        rankings: List[List[Tuple[Document, float]]],
        weights: List[float] = None
    ) -> List[Tuple[Document, float]]:
        """
        Combine multiple rankings using Reciprocal Rank Fusion (RRF).

        Args:
            rankings: List of ranked result lists from different methods
            weights: Optional weights for each ranking method

        Returns:
            Fused ranking as list of (Document, score) tuples
        """
        if weights is None:
            weights = [1.0] * len(rankings)

        # Track scores for each unique document
        doc_scores = {}
        doc_objects = {}

        for ranking, weight in zip(rankings, weights):
            for rank, (doc, _score) in enumerate(ranking):
                # Use content as key for deduplication
                content = doc.page_content[:200]  # First 200 chars as fingerprint

                # RRF formula: weight * 1 / (k + rank)
                rrf_score = weight / (self.rrf_k + rank + 1)

                if content in doc_scores:
                    doc_scores[content] += rrf_score
                else:
                    doc_scores[content] = rrf_score
                    doc_objects[content] = doc

        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return as list of (Document, score) tuples
        return [(doc_objects[content], score) for content, score in sorted_docs]

    def hybrid_retrieve(
        self,
        query: str,
        expert: str,
        k: int = 5,
        filter_category: str = None,
        use_dense: bool = True,
        use_sparse: bool = True,
        use_online: bool = True
    ) -> List[Document]:
        """
        Perform hybrid retrieval combining dense, sparse, and online search.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of final documents to return
            filter_category: Optional category filter
            use_dense: Whether to use dense vector search
            use_sparse: Whether to use sparse BM25 search
            use_online: Whether to use online web search

        Returns:
            List of Documents ranked by fused scores
        """
        expert_name = "Mayo Clinic" if expert == "A" else "WebMD"
        print(f"\n[Hybrid Retriever - {expert_name}] Query: {query[:100]}...")

        rankings = []
        weights = []

        # 1. Dense retrieval (vector similarity)
        if use_dense:
            dense_docs = self.dense_retriever.retrieve_for_expert(
                query=query,
                expert=expert,
                k=k * 2,  # Retrieve more for better fusion
                filter_category=filter_category
            )
            # Assign scores based on position (approximate similarity)
            dense_ranking = [(doc, 1.0 / (i + 1)) for i, doc in enumerate(dense_docs)]
            rankings.append(dense_ranking)
            weights.append(self.dense_weight)
            print(f"  ✓ Dense: {len(dense_docs)} docs")

        # 2. Sparse retrieval (BM25)
        if use_sparse:
            sparse_ranking = self._sparse_search(
                query=query,
                expert=expert,
                k=k * 2,
                filter_category=filter_category
            )
            rankings.append(sparse_ranking)
            weights.append(self.sparse_weight)
            print(f"  ✓ Sparse (BM25): {len(sparse_ranking)} docs")

        # 3. Online search
        if use_online and self.use_online:
            online_ranking = self._online_search(
                query=query,
                expert=expert,
                k=3
            )
            if online_ranking:
                rankings.append(online_ranking)
                weights.append(self.online_weight)
                print(f"  ✓ Online: {len(online_ranking)} docs")

        # Fuse rankings using RRF
        if not rankings:
            return []

        fused_results = self._reciprocal_rank_fusion(rankings, weights)

        # Return top k documents
        final_docs = [doc for doc, _score in fused_results[:k]]
        print(f"[Hybrid Retriever - {expert_name}] Returned {len(final_docs)} fused documents\n")

        return final_docs


# Global hybrid retriever instance
_hybrid_retriever = None


def get_hybrid_retriever(
    persist_dir: str = None,
    embedding_model: str = None,
    auto_build: bool = True,
    data_dir: str = "BlueMed_data",
    use_online: bool = True,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.3,
    online_weight: float = 0.2
) -> HybridMedicalRetriever:
    """
    Get or create global hybrid retriever instance.

    Args:
        persist_dir: Directory containing vector databases
        embedding_model: Embedding model to use
        auto_build: If True, automatically build databases if they don't exist
        data_dir: Directory containing source documents
        use_online: Whether to enable online web search
        dense_weight: Weight for dense retrieval (vector search)
        sparse_weight: Weight for sparse retrieval (BM25)
        online_weight: Weight for online search results

    Returns:
        HybridMedicalRetriever instance
    """
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from app.rag.chroma_retriever import get_retriever

        # Get base dense retriever
        dense_retriever = get_retriever(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            auto_build=auto_build,
            data_dir=data_dir
        )

        # Create hybrid retriever
        _hybrid_retriever = HybridMedicalRetriever(
            dense_retriever=dense_retriever,
            use_online=use_online,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            online_weight=online_weight
        )

        print("\n" + "="*60)
        print("🔥 Hybrid Retriever initialized:")
        print(f"   - Dense (Vector): {dense_weight * 100:.0f}%")
        print(f"   - Sparse (BM25): {sparse_weight * 100:.0f}%")
        print(f"   - Online (Web): {online_weight * 100:.0f}%" if use_online else "   - Online: Disabled")
        print("="*60 + "\n")

    return _hybrid_retriever
