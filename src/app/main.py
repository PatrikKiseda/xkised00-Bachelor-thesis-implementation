"""
Author: Patrik Kiseda
File: src/app/main.py
Description: App startup and health endpoint with Qdrant connectivity reporting.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from fastapi import FastAPI

from app.api.documents import router as documents_router
from app.api.jobs import router as jobs_router
from app.api.query import router as query_router
from app.core.settings import Settings, get_settings
from app.embeddings.adapter import EmbeddingClient
from app.embeddings.providers import build_embedding_client
from app.storage.qdrant_store import QdrantStore
from app.storage.sqlite_schema import initialize_sqlite_schema

# StoreFactory: typed factory contract used for dependency injection in tests/startup.
StoreFactory = Callable[[Settings], QdrantStore]
EmbeddingClientFactory = Callable[[Settings], EmbeddingClient]


# create_app: builds the FastAPI app and wires Qdrant startup + health reporting.
def create_app(
    settings: Settings | None = None,
    store_factory: StoreFactory | None = None,
    embedding_client_factory: EmbeddingClientFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_store_factory = store_factory or QdrantStore.from_settings
    resolved_embedding_client_factory = embedding_client_factory or build_embedding_client

    # lifespan: initializes Qdrant store at startup and stores initial reachability state. see https://fastapi.tiangolo.com/advanced/events/#lifespan.
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        sqlite_db_path = initialize_sqlite_schema(resolved_settings.sqlite_path)
        storage_dir = Path(resolved_settings.storage_dir).expanduser()
        storage_dir.mkdir(parents=True, exist_ok=True)
        qdrant_store = resolved_store_factory(resolved_settings)
        embedding_client = resolved_embedding_client_factory(resolved_settings)
        startup_status = qdrant_store.check_connection()

        app.state.settings = resolved_settings
        app.state.sqlite_db_path = str(sqlite_db_path)
        app.state.storage_dir = str(storage_dir)
        app.state.qdrant_store = qdrant_store
        app.state.embedding_client = embedding_client
        app.state.qdrant_reachable_on_startup = startup_status.reachable
        app.state.qdrant_startup_error = startup_status.error
        yield

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(query_router)


    # health endpoint decorator: exposing runtime health snapshot for app + Qdrant.
    @app.get("/api/health")
    # health: returns current Qdrant reachability and startup-time reachability details.
    def health() -> dict[str, object]:
        current_status = app.state.qdrant_store.check_connection()
        status = "ok" if current_status.reachable else "degraded"

        return {
            "status": status,
            "qdrant": {
                "url": app.state.settings.qdrant_url,
                "reachable": current_status.reachable,
                "reachable_on_startup": app.state.qdrant_reachable_on_startup,
                "startup_error": app.state.qdrant_startup_error,
                "last_error": current_status.error,
            },
            "sqlite": {
                "path": app.state.sqlite_db_path,
                "schema_initialized": True,
            },
        }


    return app


# app instance: application used by uvicorn.
app = create_app()
