"""
Author: Patrik Kiseda
File: src/app/main.py
Description: App startup and health endpoint with Qdrant connectivity reporting.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI

from app.core.settings import Settings, get_settings
from app.storage.qdrant_store import QdrantStore

# StoreFactory: typed factory contract used for dependency injection in tests/startup.
StoreFactory = Callable[[Settings], QdrantStore]


# create_app: builds the FastAPI app and wires Qdrant startup + health reporting.
def create_app(
    settings: Settings | None = None,
    store_factory: StoreFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    resolved_store_factory = store_factory or QdrantStore.from_settings

    # lifespan: initializes Qdrant store at startup and stores initial reachability state. see https://fastapi.tiangolo.com/advanced/events/#lifespan .
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        qdrant_store = resolved_store_factory(resolved_settings)
        startup_status = qdrant_store.check_connection()

        app.state.settings = resolved_settings
        app.state.qdrant_store = qdrant_store
        app.state.qdrant_reachable_on_startup = startup_status.reachable
        app.state.qdrant_startup_error = startup_status.error
        yield

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)

    return app


# app instance: application used by uvicorn.
app = create_app()
