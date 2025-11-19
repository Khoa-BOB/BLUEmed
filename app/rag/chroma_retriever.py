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

    def __init__(
        self,
        persist_dir: str = "vectordb/chroma_gemini",
        embedding_model: str = "gemini-embedding-001",
        auto_build: bool = True,
        data_dir: str = "BlueMed_data"
    ):
        """
        Initialize the retriever.

        Args:
            persist_dir: Directory containing Chroma databases
            embedding_model: Google Gemini embedding model to use
            auto_build: If True, automatically build databases if they don't exist
            data_dir: Directory containing source medical documents
        """
        self.persist_dir = persist_dir
        self.embedding_model = embedding_model
        self.auto_build = auto_build
        self.data_dir = data_dir

        # Get API key from settings
        from config.settings import settings

        # Initialize embeddings with explicit API key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=settings.GOOGLE_API_KEY_EMBED
        )

        # Initialize collections
        self.mayo_db = None
        self.webmd_db = None

        self._load_collections()

    def _load_collections(self):
        """Load or create Chroma collections."""
        mayo_path = os.path.join(self.persist_dir, "mayo_clinic")
        webmd_path = os.path.join(self.persist_dir, "webmd")

        # Check if databases need to be built
        mayo_exists = os.path.exists(mayo_path) and any(os.scandir(mayo_path))
        webmd_exists = os.path.exists(webmd_path) and any(os.scandir(webmd_path))

        # Auto-build if enabled and databases don't exist
        if self.auto_build and (not mayo_exists or not webmd_exists):
            print("\n" + "="*60)
            print("Vector databases not found. Building automatically...")
            print("="*60)

            from app.rag.vector_builder import build_medical_databases

            mayo_built, webmd_built = build_medical_databases(
                base_data_dir=self.data_dir,
                persist_base_dir=self.persist_dir,
                embedding_model=self.embedding_model,
                force_rebuild=False
            )

            # Update existence flags
            mayo_exists = mayo_built is not None
            webmd_exists = webmd_built is not None

        # Load Mayo Clinic collection
        if mayo_exists:
            try:
                self.mayo_db = Chroma(
                    persist_directory=mayo_path,
                    embedding_function=self.embeddings,
                    collection_name="mayo_clinic"
                )
                print(f"✅ Loaded Mayo Clinic collection ({self.mayo_db._collection.count()} documents)")
            except Exception as e:
                print(f"❌ Error loading Mayo Clinic collection: {e}")
                self.mayo_db = None
        else:
            print(f"⚠️  Mayo Clinic collection not found at {mayo_path}")
            if not self.auto_build:
                print("   Run: python app/rag/vector_builder.py")

        # Load WebMD collection
        if webmd_exists:
            try:
                self.webmd_db = Chroma(
                    persist_directory=webmd_path,
                    embedding_function=self.embeddings,
                    collection_name="webmd"
                )
                print(f"✅ Loaded WebMD collection ({self.webmd_db._collection.count()} documents)")
            except Exception as e:
                print(f"❌ Error loading WebMD collection: {e}")
                self.webmd_db = None
        else:
            print(f"⚠️  WebMD collection not found at {webmd_path}")
            if not self.auto_build:
                print("   Run: python app/rag/vector_builder.py")

    def retrieve_mayo(
        self,
        query: str,
        k: int = 3,
        filter_category: str = None
    ) -> List[str]:
        """
        Retrieve from Mayo Clinic knowledge base.

        Args:
            query: Search query (medical note or condition)
            k: Number of documents to retrieve
            filter_category: Optional category filter
                           ("drugs_supplements", "diseases_conditions", "symptoms")

        Returns:
            List of relevant document texts
        """
        if self.mayo_db is None:
            return ["Mayo Clinic knowledge base not available."]

        # Build metadata filter if specified
        filter_dict = None
        if filter_category:
            filter_dict = {"category": filter_category}

        docs = self.mayo_db.similarity_search(query, k=k, filter=filter_dict)
        return [doc.page_content for doc in docs]

    def retrieve_webmd(
        self,
        query: str,
        k: int = 3,
        filter_category: str = None
    ) -> List[str]:
        """
        Retrieve from WebMD knowledge base.

        Args:
            query: Search query (medical note or condition)
            k: Number of documents to retrieve
            filter_category: Optional category filter
                           ("drugs_supplements", "diseases_conditions")

        Returns:
            List of relevant document texts
        """
        if self.webmd_db is None:
            return ["WebMD knowledge base not available."]

        # Build metadata filter if specified
        filter_dict = None
        if filter_category:
            filter_dict = {"category": filter_category}

        docs = self.webmd_db.similarity_search(query, k=k, filter=filter_dict)
        return [doc.page_content for doc in docs]

    def retrieve_for_expert(
        self,
        query: str,
        expert: str,
        k: int = 3,
        filter_category: str = None
    ) -> List[str]:
        """
        Retrieve knowledge for a specific expert.

        Args:
            query: Search query
            expert: "A" for Mayo Clinic, "B" for WebMD
            k: Number of documents
            filter_category: Optional category filter
                           ("drugs_supplements", "diseases_conditions", "symptoms")

        Returns:
            List of relevant documents
        """
        if expert == "A":
            return self.retrieve_mayo(query, k, filter_category)
        elif expert == "B":
            return self.retrieve_webmd(query, k, filter_category)
        else:
            raise ValueError(f"Unknown expert: {expert}")


# Global retriever instance
_retriever = None

def get_retriever(
    persist_dir: str = None,
    embedding_model: str = None,
    auto_build: bool = True,
    data_dir: str = "BlueMed_data"
) -> MedicalKnowledgeRetriever:
    """
    Get or create global retriever instance.

    Args:
        persist_dir: Directory containing vector databases (default from settings.PERSIST_DIR)
        embedding_model: Embedding model to use (default from settings.EMBEDDING_MODEL)
        auto_build: If True, automatically build databases if they don't exist
        data_dir: Directory containing source documents

    Returns:
        MedicalKnowledgeRetriever instance
    """
    global _retriever
    if _retriever is None:
        # Get defaults from settings
        from config.settings import settings

        if persist_dir is None:
            persist_dir = settings.PERSIST_DIR
        if embedding_model is None:
            embedding_model = settings.EMBEDDING_MODEL

        _retriever = MedicalKnowledgeRetriever(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            auto_build=auto_build,
            data_dir=data_dir
        )
    return _retriever
