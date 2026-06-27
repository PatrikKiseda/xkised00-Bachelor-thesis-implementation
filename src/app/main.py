"""
Author: Patrik Kiseda
File: src/app/main.py
Description: App startup and health endpoint with Qdrant connectivity reporting.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse

from app.api.documents import router as documents_router
from app.api.jobs import router as jobs_router
from app.api.query import router as query_router
from app.api.tracking import router as tracking_router
from app.core.settings import Settings, get_settings
from app.embeddings.adapter import EmbeddingClient
from app.embeddings.providers import build_embedding_client
from app.generation.adapter import GenerationClient
from app.generation.providers import build_generation_client
from app.storage.qdrant_store import QdrantStore
from app.storage.sqlite_schema import initialize_sqlite_schema
from app.tracking.discord import DiscordUsageTracker

# StoreFactory: typed factory contract used for dependency injection in tests/startup.
StoreFactory = Callable[[Settings], QdrantStore]
EmbeddingClientFactory = Callable[[Settings], EmbeddingClient]
GenerationClientFactory = Callable[[Settings], GenerationClient]


def create_app(
    settings: Settings | None = None,
    store_factory: StoreFactory | None = None,
    embedding_client_factory: EmbeddingClientFactory | None = None,
    generation_client_factory: GenerationClientFactory | None = None,
) -> FastAPI:
    """Build the FastAPI app and wire startup dependencies.

    Args:
        settings: Optional settings override, mostly useful in tests.
        store_factory: Optional Qdrant store factory.
        embedding_client_factory: Optional embedding client factory.
        generation_client_factory: Optional generation client factory.

    Returns:
        Configured FastAPI app.
    """
    resolved_settings = settings or get_settings()
    resolved_store_factory = store_factory or QdrantStore.from_settings
    resolved_embedding_client_factory = embedding_client_factory or build_embedding_client
    resolved_generation_client_factory = generation_client_factory or build_generation_client

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize app state for SQLite, storage, Qdrant, and model clients.

        Args:
            app: FastAPI app being started.
        """
        sqlite_db_path = initialize_sqlite_schema(resolved_settings.sqlite_path)
        storage_dir = Path(resolved_settings.storage_dir).expanduser()
        storage_dir.mkdir(parents=True, exist_ok=True)
        qdrant_store = resolved_store_factory(resolved_settings)
        embedding_client = resolved_embedding_client_factory(resolved_settings)
        generation_client = resolved_generation_client_factory(resolved_settings)
        startup_status = qdrant_store.check_connection()

        app.state.settings = resolved_settings
        app.state.sqlite_db_path = str(sqlite_db_path)
        app.state.storage_dir = str(storage_dir)
        app.state.qdrant_store = qdrant_store
        app.state.embedding_client = embedding_client
        app.state.generation_client = generation_client
        app.state.usage_tracker = DiscordUsageTracker(
            webhook_url=resolved_settings.thesis_tracking_discord_webhook_url,
            enabled=resolved_settings.thesis_tracking_enabled,
        )
        app.state.qdrant_reachable_on_startup = startup_status.reachable
        app.state.qdrant_startup_error = startup_status.error
        yield

    app = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    app.include_router(documents_router)
    app.include_router(jobs_router)
    app.include_router(query_router)
    app.include_router(tracking_router)

    @app.get("/", response_class=HTMLResponse)
    def localhost_ui(request: Request, background_tasks: BackgroundTasks) -> HTMLResponse:
        """Serve the simple local HTML UI.

        Returns:
            HTML response with the local UI file.
        """
        request.app.state.usage_tracker.track(
            background_tasks=background_tasks,
            request=request,
            event="page_view",
            details={"page": "showcase"},
        )
        ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
        return HTMLResponse(content=ui_path.read_text(encoding="utf-8"))

    @app.get("/assets/projekt.pdf")
    def thesis_pdf(request: Request, background_tasks: BackgroundTasks) -> FileResponse:
        """Serve the thesis PDF used by the reviewer showcase UI.

        Returns:
            PDF file response.
        """
        request.app.state.usage_tracker.track(
            background_tasks=background_tasks,
            request=request,
            event="pdf_opened",
            details={"file": "projekt.pdf"},
        )
        pdf_path = Path(__file__).resolve().parent / "ui" / "assets" / "projekt.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Thesis PDF not found.")
        return FileResponse(
            path=pdf_path,
            media_type="application/pdf",
            filename="projekt.pdf",
            content_disposition_type="inline",
        )

    @app.get("/api/health")
    def health() -> dict[str, object]:
        """Return current Qdrant and SQLite health snapshot.

        Returns:
            Health response dict for the API.
        """
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
