"""
Author: Patrik Kiseda
File: src/app/core/settings.py
Description: Runtime settings loaded from environment variables.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic import Field, field_validator, model_validator

DEFAULT_CHUNK_SIZE_CHARS = 1000
DEFAULT_CHUNK_OVERLAP_CHARS = 150
# Default environment file path. 
DEFAULT_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"

class Settings(BaseSettings):
    """Typed config model loaded from `.env`."""

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
    qdrant_vector_size: int = Field(..., description="Dense vector size for Qdrant collection")
    sqlite_path: str = "./data/app.db"
    storage_dir: str = "./data/uploads"
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS

    # Model config (critical fields are required and validated).
    litellm_model: str = Field(..., description="Generation model, e.g. openai/gpt-4o-mini")
    embedding_provider: str = Field(..., description="Embedding provider, e.g. openai")
    embedding_model: str = Field(..., description="Embedding model id")
    embedding_api_enabled: bool = True
    openai_api_key: str | None = None

    # model_config: control of how pydantic-settings reads env vars 
    model_config = SettingsConfigDict(
        # Use the repo-local .env explicitly so sibling workspaces don't interfere with each other during development.
        # (If using multiple workspaces side by side for prototyping, this helps avoid configuration mismatches) 
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use this repo's `.env` over inherited shell env for prototype isolation.

        Args:
            settings_cls: Pydantic settings class being built.
            init_settings: Values passed directly during construction.
            env_settings: Values from the shell environment.
            dotenv_settings: Values from the repo-local `.env`.
            file_secret_settings: Values from secret files.

        Returns:
            Settings sources in the order they should be used.
        """
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    @field_validator("qdrant_url")
    @classmethod
    def validate_qdrant_url(cls, value: str) -> str:
        """Make sure Qdrant URL is explicitly HTTP(S).

        Args:
            value: Raw configured URL.

        Returns:
            The stripped URL.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("QDRANT_URL cannot be empty.")
        if not (normalized.startswith("http://") or normalized.startswith("https://")):
            raise ValueError("QDRANT_URL must start with http:// or https://.")
        return normalized

    @field_validator("qdrant_timeout_seconds")
    @classmethod
    def validate_qdrant_timeout(cls, value: float) -> float:
        """Make sure the Qdrant timeout is positive.

        Args:
            value: Timeout in seconds.

        Returns:
            The validated timeout.
        """
        if value <= 0:
            raise ValueError("QDRANT_TIMEOUT_SECONDS must be greater than 0.")
        return value

    @field_validator("qdrant_vector_size")
    @classmethod
    def validate_qdrant_vector_size(cls, value: int) -> int:
        """Make sure vector size can define a deterministic dense schema.

        Args:
            value: Configured vector size.

        Returns:
            The validated vector size.
        """
        if value <= 0:
            raise ValueError("QDRANT_VECTOR_SIZE must be greater than 0.")
        return value

    @field_validator("chunk_size_chars")
    @classmethod
    def validate_chunk_size_chars(cls, value: int) -> int:
        """Make sure chunk size is a positive integer.

        Args:
            value: Configured chunk size in chars.

        Returns:
            The validated chunk size.
        """
        if value <= 0:
            raise ValueError("CHUNK_SIZE_CHARS must be greater than 0.")
        return value

    @field_validator("chunk_overlap_chars")
    @classmethod
    def validate_chunk_overlap_chars_non_negative(cls, value: int) -> int:
        """Make sure chunk overlap is not negative.

        Args:
            value: Configured overlap in chars.

        Returns:
            The validated overlap.
        """
        if value < 0:
            raise ValueError("CHUNK_OVERLAP_CHARS must be greater than or equal to 0.")
        return value
    
    @field_validator(
        "qdrant_collection",
        "litellm_model",
        "embedding_provider",
        "embedding_model",
        "sqlite_path",
        "storage_dir",
    )
    @classmethod
    def validate_non_empty_critical_strings(cls, value: str) -> str:
        """Make sure critical string config values are not blank.

        Args:
            value: Raw string value.

        Returns:
            The stripped non-empty string.
        """
        normalized = value.strip()
        if not normalized:
            raise ValueError("Critical config value cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_provider_dependencies(self) -> "Settings":
        """Check settings that depend on each other.

        Returns:
            The validated settings object.
        """
        if self.embedding_api_enabled and self.embedding_provider.lower() == "openai":
            if not self.openai_api_key or not self.openai_api_key.strip():
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER is set to 'openai'."
                )
        # chunk size vs. overlap validator: makes sure there's logical consistency check between chunking parameters.
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS.")
        return self

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Shared cached accessor for settings between scripts and app.

    Returns:
        Loaded settings instance.
    """
    return Settings()
