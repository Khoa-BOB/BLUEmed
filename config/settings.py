from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import pathlib
class Settings(BaseSettings):
    # Define your fields with optional defaults
    EXPERT_MODEL: str
    JUDGE_MODEL: str
    OLLAMA_URL: str
    PERSIST_DIR: str

    # Google Gemini API
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Embedding Model for RAG
    GOOGLE_API_KEY_EMBED: str = ""
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # RAG Retrieval Toggle
    USE_RETRIEVER: bool = False  # Set to True to enable RAG retrieval

    # Force rebuild vector databases (even if they already exist)
    FORCE_REBUILD: bool = False  # Set to True to force rebuild databases

    # Vector build performance settings (optimized for paid tier)
    EMBED_BATCH_SIZE: int = 500  # Chunks per batch (500 for paid, 100 for free tier)
    EMBED_BATCH_DELAY: float = 1.0  # Seconds between batches (1.0 for paid, 5.0 for free)
    EMBED_RPM: int = 3000  # Requests per minute (3000 paid, 15 free)
    EMBED_TPM: int = 1000000  # Tokens per minute (1M paid, 1500 free)

    # Use model_config instead of class Config
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        case_sensitive=True  # or False if you want case-insensitive matching
    )

# Debug: Check current working directory
# print(f"Current working directory: {os.getcwd()}")
# print(f"Settings.py location: {__file__}")
# print(f".env exists in cwd: {pathlib.Path('.env').exists()}")

settings = Settings()

print(settings.model_dump())