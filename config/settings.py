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

    # RAG Retrieval Toggle
    USE_RETRIEVER: bool = False  # Set to True to enable RAG retrieval

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