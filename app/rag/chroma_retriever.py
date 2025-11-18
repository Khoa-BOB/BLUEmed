"""
Chroma-based retriever for asymmetric medical knowledge access.
"""
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict
import os


class MedicalKnowledgeRetriever:
    """
    Retriever for medical knowledge with asymmetric access.

    Expert A gets Mayo Clinic knowledge.
    Expert B gets WebMD knowledge.
    """

    def __init__(self, persist_dir: str = "vectordb", embedding_model: str = "models/text-embedding-004"):
        """
        Initialize the retriever.

        Args:
            persist_dir: Directory containing Chroma databases
            embedding_model: Google Gemini embedding model to use
        """
        self.persist_dir = persist_dir

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model)

        # Initialize collections
        self.mayo_db = None
        self.webmd_db = None

        self._load_collections()

    def _load_collections(self):
        """Load or create Chroma collections."""
        mayo_path = os.path.join(self.persist_dir, "mayo_clinic")
        webmd_path = os.path.join(self.persist_dir, "webmd")

        # Load Mayo Clinic collection
        if os.path.exists(mayo_path):
            self.mayo_db = Chroma(
                persist_directory=mayo_path,
                embedding_function=self.embeddings,
                collection_name="mayo_clinic"
            )
            print(f"✅ Loaded Mayo Clinic collection ({self.mayo_db._collection.count()} documents)")
        else:
            print(f"⚠️  Mayo Clinic collection not found at {mayo_path}")
            print("   Run: python scripts/build_medical_db.py")

        # Load WebMD collection
        if os.path.exists(webmd_path):
            self.webmd_db = Chroma(
                persist_directory=webmd_path,
                embedding_function=self.embeddings,
                collection_name="webmd"
            )
            print(f"✅ Loaded WebMD collection ({self.webmd_db._collection.count()} documents)")
        else:
            print(f"⚠️  WebMD collection not found at {webmd_path}")
            print("   Run: python scripts/build_medical_db.py")

    def retrieve_mayo(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve from Mayo Clinic knowledge base.

        Args:
            query: Search query (medical note or condition)
            k: Number of documents to retrieve

        Returns:
            List of relevant document texts
        """
        if self.mayo_db is None:
            return ["Mayo Clinic knowledge base not available."]

        docs = self.mayo_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def retrieve_webmd(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve from WebMD knowledge base.

        Args:
            query: Search query (medical note or condition)
            k: Number of documents to retrieve

        Returns:
            List of relevant document texts
        """
        if self.webmd_db is None:
            return ["WebMD knowledge base not available."]

        docs = self.webmd_db.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]

    def retrieve_for_expert(self, query: str, expert: str, k: int = 3) -> List[str]:
        """
        Retrieve knowledge for a specific expert.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of documents

        Returns:
            List of relevant documents
        """
        if expert == "A":
            return self.retrieve_mayo(query, k)
        elif expert == "B":
            return self.retrieve_webmd(query, k)
        else:
            raise ValueError(f"Unknown expert: {expert}")


# Global retriever instance
_retriever = None

def get_retriever(persist_dir: str = "vectordb") -> MedicalKnowledgeRetriever:
    """Get or create global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = MedicalKnowledgeRetriever(persist_dir)
    return _retriever
