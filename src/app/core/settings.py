"""
Author: Patrik Kiseda
File: src/app/core/settings.py
Description: Runtime settings loaded from environment variables.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator

DEFAULT_CHUNK_SIZE_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 150

# Settings: typed config model loaded from .env.
class Settings(BaseSettings):
    # App runtime config (defaults considered safe for local development).
    app_name: str = "rag-thesis-app"
    app_env: str = "dev"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"

    # Qdrant config (critical fields are required and validated).
    qdrant_url: str = Field(..., description="Qdrant base URL, e.g. http://127.0.0.1:6333")
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 3.0
    qdrant_collection: str = Field(..., description="Qdrant collection used by the app")
    sqlite_path: str = "./data/app.db"
    storage_dir: str = "./data/uploads"
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS

    # Model config (critical fields are required and validated).
    litellm_model: str = Field(..., description="Generation model, e.g. openai/gpt-4o-mini")
    embedding_provider: str = Field(..., description="Embedding provider, e.g. openai")
    embedding_model: str = Field(..., description="Embedding model id")
    openai_api_key: str | None = None

    # model_config: control of how pydantic-settings reads env vars 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # qdrant_url validator: ensures URL is explicitly HTTP(S).
    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("QDRANT_URL cannot be empty.")
        if not (normalized.startswith("http://") or normalized.startswith("https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://.")
        return normalized

    # timeout validator: enforces a positive timeout value.
    @field_validator("qdrant_timeout_seconds")
    @classmethod
    def validate_qdrant_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("QDRANT_TIMEOUT_SECONDS must be greater than 0.")
        return value

    # chunk size validator: ensures chunk size is a positive integer.
    @field_validator("chunk_size_chars")
    @classmethod
    def validate_chunk_size_chars(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("CHUNK_SIZE_CHARS must be greater than 0.")
        return value

    # chunk overlap validator: ensures chunk overlap is non-negative and reasonable compared to size of chunk.
    @field_validator("chunk_overlap_chars")
    @classmethod
    def validate_chunk_overlap_chars_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHUNK_OVERLAP_CHARS must be greater than or equal to 0.")
        return value
    
    # critical-string validator: rejects blank critical values for required config keys.
    @field_validator(
        "qdrant_collection",
        "litellm_model",
        "embedding_provider",
        "embedding_model",
        "sqlite_path",
        "storage_dir",
    )
    # validate_non_empty_critical_strings: makes sure string config values are not empty or just whitespace.
    @classmethod
    def validate_non_empty_critical_strings(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Critical config value cannot be empty.")
        return normalized

    # model-level validator: enforces provider-dependent API key requirements.
    @model_validator(mode="after")
    def validate_provider_dependencies(self) -> "Settings":
        if self.embedding_provider.lower() == "openai":
            if not self.openai_api_key or not self.openai_api_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER is set to 'openai'."
                )
        # chunk size vs. overlap validator: makes sure there's logical consistency check between chunking parameters.
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS.")
        return self

# get_settings decorator: cache one Settings instance for repeated app access.  https://docs.python.org/3/library/functools.html#functools.lru_cache for details.
@lru_cache(maxsize=1)
# get_settings: shared accessor for settings between scripts/app
def get_settings() -> Settings:
    return Settings()
