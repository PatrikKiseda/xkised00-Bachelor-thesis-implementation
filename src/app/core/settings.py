"""
Author: Patrik Kiseda
File: src/app/core/settings.py
Description: Runtime settings loaded from environment variables.
"""

from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

# Settings: typed config model loaded from .env.
class Settings(BaseSettings):
    app_name: str = "rag-thesis-app"
    app_env: str = "dev"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: str | None = None
    qdrant_timeout_seconds: float = 3.0
    qdrant_collection: str = "documents"

    # model_config: control of how pydantic-settings reads env vars 
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# get_settings decorator: cache one Settings instance for repeated app access.  https://docs.python.org/3/library/functools.html#functools.lru_cache for details.
@lru_cache(maxsize=1)
# get_settings: shared accessor for settings between scripts/app
def get_settings() -> Settings:
    return Settings()