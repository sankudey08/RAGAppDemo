"""Application configuration loaded from environment variables and .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, HttpUrl, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated runtime settings for the RAG application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ollama_base_url: HttpUrl = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="qwen3:8b", min_length=1)
    ollama_embedding_model: str = Field(default="nomic-embed-text", min_length=1)
    chroma_persist_directory: Path = Field(default=Path("./data/chroma"))
    chroma_collection_name: str = Field(
        default="rag_documents",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$",
    )
    cache_collection_name: str = Field(
        default="rag_answer_cache",
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{1,61}[A-Za-z0-9]$",
    )
    max_upload_size_mb: int = Field(default=25, ge=1, le=1024)
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=5000)
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    retrieval_min_score: float = Field(default=0.2, ge=0, le=1)
    cache_enabled: bool = False
    cache_similarity_threshold: float = Field(default=0.92, ge=0, le=1)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("ollama_chat_model", "ollama_embedding_model")
    @classmethod
    def model_name_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("model names cannot be blank")
        return value

    @field_validator("chroma_persist_directory")
    @classmethod
    def persistence_path_must_not_be_file(cls, value: Path) -> Path:
        if value.exists() and not value.is_dir():
            raise ValueError(f"ChromaDB persistence path is not a directory: {value}")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_must_be_smaller_than_chunk_size(cls, value: int, info: ValidationInfo) -> int:
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""

    return Settings()
