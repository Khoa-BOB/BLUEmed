"""
Vector database builder for medical knowledge sources.

Automatically builds Chroma vector databases from medical documents
using Google Gemini embeddings.
"""
import time
from pathlib import Path
from typing import Optional, List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader


class MedicalVectorBuilder:
    """
    Builds vector databases for medical knowledge sources.
    """

    def __init__(
        self,
        embedding_model: str = "gemini-embedding-001",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        google_api_key_embed: str = None,
        batch_size: int = 500,
        batch_delay: float = 1.0,
        requests_per_minute: int = 3000,
        tokens_per_minute: int = 1000000
    ):
        """
        Initialize the vector builder.

        Args:
            embedding_model: Google Gemini embedding model to use
            chunk_size: Characters per chunk
            chunk_overlap: Character overlap between chunks
            google_api_key_embed: Google API key (default from settings)
            batch_size: Number of chunks to process per batch (default 500 for paid tier)
            batch_delay: Delay between batches in seconds (default 1.0)
            requests_per_minute: API rate limit for requests (default 3000 for paid tier)
            tokens_per_minute: API rate limit for tokens (default 1000000 for paid tier)
        """
        # Get settings
        from config.settings import settings

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Use settings values if not explicitly provided
        self.batch_size = batch_size if batch_size != 500 else settings.EMBED_BATCH_SIZE
        self.batch_delay = batch_delay if batch_delay != 1.0 else settings.EMBED_BATCH_DELAY
        self.requests_per_minute = requests_per_minute if requests_per_minute != 3000 else settings.EMBED_RPM
        self.tokens_per_minute = tokens_per_minute if tokens_per_minute != 1000000 else settings.EMBED_TPM

        # Get API key from settings if not provided
        if google_api_key_embed is None:
            google_api_key_embed = settings.GOOGLE_API_KEY_EMBED

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=google_api_key_embed
        )

    def load_documents(self, data_dir: Path, source_name: str) -> List:
        """
        Load documents from a directory with metadata enrichment.

        Args:
            data_dir: Directory containing markdown/text files
            source_name: Name of the source (e.g., "mayo_clinic" or "webmd")

        Returns:
            List of loaded documents with enriched metadata
        """
        print(f"Loading documents from {data_dir}...")

        loader = DirectoryLoader(
            str(data_dir),
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True
        )

        try:
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")

            # Enrich metadata with category and source
            for doc in documents:
                # Extract category from file path
                doc_path = Path(doc.metadata.get("source", ""))
                category = self._extract_category(doc_path)

                # Add enriched metadata
                doc.metadata["category"] = category
                doc.metadata["source_name"] = source_name
                doc.metadata["file_path"] = str(doc_path)

            # Print category breakdown
            categories = {}
            for doc in documents:
                cat = doc.metadata.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            print(f"Category breakdown:")
            for cat, count in sorted(categories.items()):
                print(f"  - {cat}: {count} documents")

            return documents
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []

    def _extract_category(self, file_path: Path) -> str:
        """
        Extract category from file path.

        Args:
            file_path: Path to the document file

        Returns:
            Category name (drugs_supplements, diseases_conditions, symptoms, or unknown)
        """
        path_str = str(file_path).lower()

        if "drugs_supplements" in path_str or "drug_supplement" in path_str:
            return "drugs_supplements"
        elif "diseases_conditions" in path_str or "disease_condition" in path_str:
            return "diseases_conditions"
        elif "symptoms" in path_str or "symptom" in path_str:
            return "symptoms"
        else:
            return "unknown"

    def chunk_documents(self, documents: List) -> List:
        """
        Split documents into chunks.

        Args:
            documents: List of documents

        Returns:
            List of chunked documents
        """
        print(f"Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )

        chunks = splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        return chunks

    def build_vectorstore(
        self,
        data_dir: Path,
        persist_dir: Path,
        collection_name: str,
        source_name: str
    ) -> Optional[Chroma]:
        """
        Build a vector database from documents.

        Args:
            data_dir: Directory containing source documents
            persist_dir: Directory to persist the vector database
            collection_name: Name of the Chroma collection
            source_name: Name of the source for metadata (e.g., "mayo_clinic")

        Returns:
            Chroma vector store instance or None if failed
        """
        print(f"\n{'='*60}")
        print(f"Building {collection_name} vector database")
        print(f"{'='*60}")

        # Check if data directory exists
        if not data_dir.exists():
            print(f"Error: Data directory not found: {data_dir}")
            return None

        # Load documents with metadata
        documents = self.load_documents(data_dir, source_name)
        if not documents:
            print(f"No documents loaded from {data_dir}")
            return None

        # Chunk documents
        chunks = self.chunk_documents(documents)
        if not chunks:
            print("No chunks created")
            return None

        # Create persist directory
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Build vector database with rate limiting
        print(f"Building Chroma database at {persist_dir}...")
        print(f"Using embedding model: {self.embedding_model}")
        print(f"Rate limits: {self.requests_per_minute} RPM, {self.tokens_per_minute} TPM")
        print(f"Processing {len(chunks)} chunks in batches of {self.batch_size}...")

        # Estimate processing time
        total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
        est_time_min = (total_batches * self.batch_delay) / 60
        print(f"Estimated time: ~{est_time_min:.1f} minutes ({total_batches} batches)")

        try:
            vectordb = None
            start_time = time.time()

            for i in range(0, len(chunks), self.batch_size):
                batch_num = i // self.batch_size + 1
                batch = chunks[i:i + self.batch_size]

                print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")

                retry_count = 0
                max_retries = 3
                batch_start = time.time()

                while retry_count < max_retries:
                    try:
                        if vectordb is None:
                            # Create new database with first batch
                            vectordb = Chroma.from_documents(
                                documents=batch,
                                embedding=self.embeddings,
                                persist_directory=str(persist_dir),
                                collection_name=collection_name
                            )
                        else:
                            # Add to existing database
                            vectordb.add_documents(batch)

                        # Success
                        batch_time = time.time() - batch_start
                        print(f"✓ ({batch_time:.1f}s)")
                        break

                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e).lower()
                        if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                            wait_time = 60 * retry_count  # Exponential backoff
                            print(f"\n    ⚠️  Rate limit hit. Waiting {wait_time}s before retry {retry_count}/{max_retries}...")
                            time.sleep(wait_time)
                        else:
                            print(f"\n    ❌ Error: {e}")
                            raise e

                if retry_count >= max_retries:
                    print(f"    ❌ Failed after {max_retries} retries")
                    return None

                # Rate limiting delay between batches
                if i + self.batch_size < len(chunks):
                    time.sleep(self.batch_delay)

            total_time = time.time() - start_time
            print(f"✅ Successfully built {collection_name} database in {total_time/60:.1f} minutes")
            print(f"   Documents: {len(documents)}")
            print(f"   Chunks: {len(chunks)}")
            print(f"   Location: {persist_dir}")
            print()

            return vectordb

        except Exception as e:
            print(f"❌ Error building vector database: {e}")
            return None


def build_medical_databases(
    base_data_dir: str = "BlueMed_data",
    persist_base_dir: str = None,
    embedding_model: Optional[str] = None,
    force_rebuild: bool = None
) -> tuple[Optional[Path], Optional[Path]]:
    """
    Build both Mayo Clinic and WebMD vector databases.

    Args:
        base_data_dir: Base directory containing medical data
        persist_base_dir: Base directory for vector databases (default from settings)
        embedding_model: Embedding model to use (default from settings)
        force_rebuild: If True, rebuild even if databases exist (default from settings.FORCE_REBUILD)

    Returns:
        Tuple of (mayo_path, webmd_path) for the vector databases
    """
    # Import settings
    from config.settings import settings

    # Get defaults from settings if not specified
    if persist_base_dir is None:
        persist_base_dir = settings.PERSIST_DIR
    if embedding_model is None:
        embedding_model = settings.EMBEDDING_MODEL
    if force_rebuild is None:
        force_rebuild = settings.FORCE_REBUILD

    # Initialize builder
    builder = MedicalVectorBuilder(embedding_model=embedding_model)

    # Define paths
    base_data = Path(base_data_dir)
    persist_base = Path(persist_base_dir)

    mayo_data = base_data / "MayoClinic"
    webmd_data = base_data / "WebMD"

    mayo_persist = persist_base / "mayo_clinic"
    webmd_persist = persist_base / "webmd"

    mayo_built = False
    webmd_built = False

    # Build Mayo Clinic database
    if force_rebuild or not mayo_persist.exists() or not list(mayo_persist.glob("*")):
        print("\nMayo Clinic database not found or rebuild requested")
        mayo_db = builder.build_vectorstore(
            data_dir=mayo_data,
            persist_dir=mayo_persist,
            collection_name="mayo_clinic",
            source_name="mayo_clinic"
        )
        mayo_built = mayo_db is not None
    else:
        print(f"\n✅ Mayo Clinic database already exists at {mayo_persist}")
        mayo_built = True

    # Build WebMD database
    if force_rebuild or not webmd_persist.exists() or not list(webmd_persist.glob("*")):
        print("\nWebMD database not found or rebuild requested")
        webmd_db = builder.build_vectorstore(
            data_dir=webmd_data,
            persist_dir=webmd_persist,
            collection_name="webmd",
            source_name="webmd"
        )
        webmd_built = webmd_db is not None
    else:
        print(f"\n✅ WebMD database already exists at {webmd_persist}")
        webmd_built = True

    # Return paths if successful
    return (
        mayo_persist if mayo_built else None,
        webmd_persist if webmd_built else None
    )


if __name__ == "__main__":
    """
    Run as standalone script to build databases.
    """
    from config.settings import settings

    print("Building medical knowledge vector databases...")
    print(f"Using embedding model: {settings.EMBEDDING_MODEL}")
    print(f"Persist directory: {settings.PERSIST_DIR}")
    print(f"Force rebuild: {settings.FORCE_REBUILD}")
    print()

    mayo_path, webmd_path = build_medical_databases()

    if mayo_path and webmd_path:
        print("\n" + "="*60)
        print("✅ All databases built successfully!")
        print("="*60)
        print(f"Mayo Clinic: {mayo_path}")
        print(f"WebMD: {webmd_path}")
    else:
        print("\n" + "="*60)
        print("❌ Some databases failed to build")
        print("="*60)
