"""
Shared test helpers for the local unittest suite.

The tests run without real Qdrant or model APIs. This file keeps that setup in one
place: relative src imports, default env values, fake stores/clients, common
Settings construction, and small SQLite seed helpers.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from pathlib import Path
from typing import Iterator


SRC_PATH = Path("src")
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Minimum env needed by Settings when tests import app modules.
# These are defaults only; tests that check env behavior still patch os.environ.
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "documents")
os.environ.setdefault("QDRANT_VECTOR_SIZE", "8")
os.environ.setdefault("LITELLM_MODEL", "openai/gpt-4o-mini")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("EMBEDDING_API_ENABLED", "false")

from app.core.settings import Settings
from app.retrieval.service import RetrievedChunk
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import ChunkUpsert, replace_document_chunks
from app.storage.qdrant_store import ChunkVector, DenseSearchHit, QdrantConnectionStatus


def build_settings(
    sqlite_path: str = ":memory:",
    storage_dir: str = "./data/uploads",
    **overrides: object,
) -> Settings:
    """Build validated test settings with local providers and small chunks.

    Args:
        sqlite_path: SQLite path used by the app under test.
        storage_dir: Upload storage path used by document API tests.
        **overrides: Any Settings fields that a test wants to change.

    Returns:
        Settings instance with deterministic local defaults.
    """
    payload = {
        # The app name is not behavior-critical, but using a test-specific value makes
        # accidental dependency on production/default config easier to spot.
        "app_name": "test-app",
        # Qdrant URL is syntactically valid but intentionally fake; store fakes replace
        # the real client in API tests, so no network connection should happen.
        "qdrant_url": "http://test-qdrant:6333",
        # Collection name is the runtime default concept used through retrieval/indexing.
        "qdrant_collection": "documents",
        # Eight dimensions keeps deterministic embedding vectors tiny while still
        # exercising vector-size validation and dense-search plumbing.
        "qdrant_vector_size": 8,
        # Most tests pass temp paths so each test gets an isolated database.
        "sqlite_path": sqlite_path,
        # Upload tests pass a temp directory here; unit tests can keep this harmless default.
        "storage_dir": storage_dir,
        # Small chunk values make uploads split into chunks without huge fixture content.
        "chunk_size_chars": 120,
        # Non-zero overlap keeps tests aligned with the actual indexing behavior.
        "chunk_overlap_chars": 20,
        # Generation calls are faked/patchable, but this still must be a plausible model id.
        "litellm_model": "openai/gpt-4o-mini",
        # Local embedding provider plus API disabled avoids requiring real API keys.
        "embedding_provider": "local",
        # Model name is preserved in metadata assertions, even when embeddings are fake.
        "embedding_model": "text-embedding-3-small",
        # Default tests should never call external embedding APIs.
        "embedding_api_enabled": False,
    }
    payload.update(overrides)
    return Settings(**payload)


class HealthyStore:
    """Fake Qdrant store that reports healthy and accepts indexing calls."""

    def check_connection(self) -> QdrantConnectionStatus:
        """Return a reachable status for startup and health endpoint checks."""
        return QdrantConnectionStatus(reachable=True)

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        """Pretend the collection already exists with the expected schema."""
        return None

    def upsert_chunk_vectors(self, *, collection_name: str, vectors: list[object]) -> None:
        """Accept vectors without storing them; enough for upload lifecycle tests."""
        return None


class UnhealthyStore(HealthyStore):
    """Fake Qdrant store that reports an unreachable startup/current state."""

    def check_connection(self) -> QdrantConnectionStatus:
        """Return an unreachable status with a stable error string."""
        return QdrantConnectionStatus(reachable=False, error="qdrant unreachable")


class InMemoryDenseStore(HealthyStore):
    """Fake dense store with deterministic in-memory vector search results."""

    def __init__(self) -> None:
        """Create empty vector storage and optional search override."""
        self._vectors: list[ChunkVector] = []
        self.dense_hits_override: list[DenseSearchHit] | None = None

    def upsert_chunk_vectors(self, *, collection_name: str, vectors: list[ChunkVector]) -> None:
        """Store latest vectors so dense API tests can query them later."""
        self._vectors = vectors

    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
        """Return deterministic dense hits from override or stored vectors."""
        if self.dense_hits_override is not None:
            return self.dense_hits_override[:limit]

        return [
            DenseSearchHit(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                chunk_index=item.chunk_index,
                score=1.0 - (index * 0.01),
            )
            for index, item in enumerate(self._vectors[:limit])
        ]


class RecordingGenerationClient:
    """Fake generation client that records prompts and returns fixed text."""

    model = "openai/gpt-5.4-mini"

    def __init__(self, response_text: str = "Grounded answer [S1]") -> None:
        """Store response text and initialize recorded calls."""
        self.response_text = response_text
        self.calls: list[tuple[str, float]] = []

    def generate_text(self, *, prompt: str, temperature: float) -> str:
        """Record prompt/temperature and return the configured fake answer."""
        self.calls.append((prompt, temperature))
        return self.response_text


def sample_sources() -> list[RetrievedChunk]:
    """Return two stable retrieved chunks for prompt/generation tests."""
    return [
        RetrievedChunk(
            source_id="S1",
            chunk_id="doc-1:000000",
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=0,
            score=0.91,
            content="alpha beta gamma",
        ),
        RetrievedChunk(
            source_id="S2",
            chunk_id="doc-1:000001",
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=1,
            score=0.87,
            content="delta epsilon zeta",
        ),
    ]


def insert_test_document(db_path: str, *, document_id: str = "doc-1", filename: str = "notes.txt") -> None:
    """Insert a minimal document row for retrieval/storage tests."""
    insert_document(
        db_path,
        document_id=document_id,
        filename=filename,
        source_type="txt",
        source_path=f"tmp/{filename}",
        size_bytes=123,
        checksum=f"checksum-{document_id}",
        status="success",
    )


def seed_document_chunks(
    db_path: str,
    *,
    document_id: str = "doc-1",
    filename: str = "notes.txt",
    chunks: list[ChunkUpsert],
) -> None:
    """Insert a document and replace its chunks for query/retrieval tests."""
    insert_test_document(db_path, document_id=document_id, filename=filename)
    replace_document_chunks(db_path, document_id=document_id, chunks=chunks)


def two_ranked_chunks() -> list[ChunkUpsert]:
    """Return two simple chunks used by hybrid API/retrieval tests."""
    return [
        ChunkUpsert(id="doc-1:000000", chunk_index=0, content="alpha alpha beta"),
        ChunkUpsert(id="doc-1:000001", chunk_index=1, content="alpha beta gamma"),
    ]


def dense_hits_for_two_ranked_chunks() -> list[DenseSearchHit]:
    """Return dense hits in reverse lexical order to exercise RRF fusion."""
    return [
        DenseSearchHit(chunk_id="doc-1:000001", document_id="doc-1", chunk_index=1, score=0.99),
        DenseSearchHit(chunk_id="doc-1:000000", document_id="doc-1", chunk_index=0, score=0.98),
    ]


@contextlib.contextmanager
def suppress_expected_pdf_noise() -> Iterator[None]:
    """Hide expected parser noise from malformed-PDF negative tests."""
    with contextlib.redirect_stderr(io.StringIO()):
        yield
