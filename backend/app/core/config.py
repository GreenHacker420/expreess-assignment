"""Application configuration and environment settings."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM provider settings
    llm_provider: str = Field(default="gemini", alias="LLM_PROVIDER")
    gemini_api_key: str | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    gemini_model: str = Field(default="gemini-3.8-flash", alias="GEMINI_MODEL")
    gemini_fast_model: str = Field(default="gemini-3.8-flash", alias="GEMINI_FAST_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    llm_model: str = Field(default="openai/gpt-oss-120b", alias="LLM_MODEL")
    fast_llm_model: str = Field(default="openai/gpt-oss-120b", alias="FAST_LLM_MODEL")
    llm_temperature: float = Field(default=0.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: float = Field(default=30.0, alias="LLM_TIMEOUT_SECONDS")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    web_search_enabled: bool = Field(default=False, alias="WEB_SEARCH_ENABLED")
    web_search_max_results: int = Field(default=5, alias="WEB_SEARCH_MAX_RESULTS")

    # Embeddings & Vector Store
    embedding_provider: str = Field(default="gemini", alias="EMBEDDING_PROVIDER")
    gemini_embedding_model: str = Field(
        default="gemini-embedding-2", alias="GEMINI_EMBEDDING_MODEL"
    )
    gemini_embedding_dimensions: int = Field(default=768, alias="GEMINI_EMBEDDING_DIMENSIONS")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )
    # Storage & Persistence
    chroma_dir: str = Field(default="backend/data/chroma", alias="CHROMA_DIR")
    chroma_collection: str = Field(default="fastapi_docs_gemini2_qa_v1", alias="CHROMA_COLLECTION")
    db_path: str = Field(default="backend/data/app.db", alias="DB_PATH")
    checkpoint_db_path: str = Field(
        default="backend/data/checkpoints.db", alias="CHECKPOINT_DB_PATH"
    )
    docs_dir: str = Field(default="backend/data/corpus", alias="DOCS_DIR")
    manifest_path: str = Field(default="backend/data/corpus_manifest.json", alias="MANIFEST_PATH")

    # RAG Workflow Parameters
    max_retries: int = Field(default=2, alias="MAX_RETRIES")
    top_k: int = Field(default=5, alias="TOP_K")
    grading_concurrency: int = Field(default=3, alias="GRADING_CONCURRENCY")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")

    # API Validation Limits
    max_question_length: int = Field(default=1000, alias="MAX_QUESTION_LENGTH")
    max_upload_size_bytes: int = Field(default=5 * 1024 * 1024, alias="MAX_UPLOAD_SIZE_BYTES")
    cors_origins: list[str] = Field(default_factory=list, alias="CORS_ORIGINS")

    model_config = SettingsConfigDict(
        env_file=[".env", "backend/.env"],
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Ensure all required local data directories exist."""
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.docs_dir).mkdir(parents=True, exist_ok=True)
        db_parent = Path(self.db_path).parent
        if db_parent:
            db_parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings
