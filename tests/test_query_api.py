"""
Author: Patrik Kiseda
File: tests/test_query_api.py
Description: API tests for dense and lexical query endpoints and SQLite content hydration.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "documents")
os.environ.setdefault("QDRANT_VECTOR_SIZE", "8")
os.environ.setdefault("LITELLM_MODEL", "openai/gpt-4o-mini")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("EMBEDDING_API_ENABLED", "false")

from app.core.settings import Settings
from app.main import create_app
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import ChunkUpsert, replace_document_chunks
from app.storage.qdrant_store import ChunkVector, DenseSearchHit, QdrantConnectionStatus


# _InMemoryDenseStore: keeps indexed vectors in memory and returns deterministic dense hits.
class _InMemoryDenseStore:
    def __init__(self) -> None:
        self._vectors: list[ChunkVector] = []
        self.dense_hits_override: list[DenseSearchHit] | None = None

    def check_connection(self) -> QdrantConnectionStatus:
        return QdrantConnectionStatus(reachable=True)

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        return None

    def upsert_chunk_vectors(self, *, collection_name: str, vectors: list[ChunkVector]) -> None:
        self._vectors = vectors

    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
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


# _build_settings: helper creating settings for query API tests.
def _build_settings(sqlite_path: str, storage_dir: str, **overrides: object) -> Settings:
    payload = {
        "app_name": "test-app",
        "qdrant_url": "http://test-qdrant:6333",
        "qdrant_collection": "documents",
        "qdrant_vector_size": 8,
        "sqlite_path": sqlite_path,
        "storage_dir": storage_dir,
        "chunk_size_chars": 120,
        "chunk_overlap_chars": 20,
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_enabled": False,
    }
    payload.update(overrides)
    return Settings(**payload)


class TestDenseQueryApi(unittest.TestCase):
    # test_query_dense_returns_hydrated_hits: upload+index then query returns refs, score, and hydrated content.
    def test_query_dense_returns_hydrated_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 40), "text/plain")},
                )
                query_response = client.post(
                    "/api/query/dense",
                    json={"query": "gamma delta", "top_k": 3},
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(query_response.status_code, 200)
            payload = query_response.json()
            self.assertEqual(payload["mode"], "dense")
            self.assertEqual(payload["query"], "gamma delta")
            self.assertGreater(len(payload["hits"]), 0)

            first_hit = payload["hits"][0]
            self.assertIn("chunk_id", first_hit)
            self.assertIn("document_id", first_hit)
            self.assertIn("chunk_index", first_hit)
            self.assertIn("score", first_hit)
            self.assertIn("content", first_hit)
            self.assertTrue(first_hit["content"])

    # test_query_dense_returns_empty_hits_when_index_is_empty: no indexed vectors should return empty result set.
    def test_query_dense_returns_empty_hits_when_index_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                response = client.post("/api/query/dense", json={"query": "anything"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "dense")
            self.assertEqual(payload["hits"], [])

    # test_query_dense_rejects_invalid_payload: request validation should guard empty queries and invalid top_k.
    def test_query_dense_rejects_invalid_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                empty_query = client.post("/api/query/dense", json={"query": "", "top_k": 5})
                invalid_top_k = client.post("/api/query/dense", json={"query": "valid", "top_k": 0})

            self.assertEqual(empty_query.status_code, 422)
            self.assertEqual(invalid_top_k.status_code, 422)

    # test_query_lexical_returns_ranked_hydrated_hits: upload+index then lexical query returns refs, score, and hydrated content.
    def test_query_lexical_returns_ranked_hydrated_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 40), "text/plain")},
                )
                query_response = client.post(
                    "/api/query/lexical",
                    json={"query": "alpha beta", "top_k": 3},
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(query_response.status_code, 200)
            payload = query_response.json()
            self.assertEqual(payload["mode"], "lexical")
            self.assertEqual(payload["query"], "alpha beta")
            self.assertGreater(len(payload["hits"]), 0)

            first_hit = payload["hits"][0]
            self.assertIn("chunk_id", first_hit)
            self.assertIn("document_id", first_hit)
            self.assertIn("chunk_index", first_hit)
            self.assertIn("score", first_hit)
            self.assertIn("content", first_hit)
            self.assertTrue(first_hit["content"])
            self.assertGreaterEqual(first_hit["score"], 0.0)

    def test_query_lexical_returns_empty_hits_when_index_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _InMemoryDenseStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                response = client.post("/api/query/lexical", json={"query": "anything"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "lexical")
            self.assertEqual(payload["hits"], [])

    # test_query_lexical_rejects_invalid_payload: request validation should guard empty queries and invalid top_k.
    def test_query_hybrid_returns_fused_ranked_hits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            store = _InMemoryDenseStore()
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: store,  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                self._seed_document_chunks(
                    app.state.sqlite_db_path,
                    document_id="doc-1",
                    filename="notes.txt",
                    chunks=[
                        ChunkUpsert(
                            id="doc-1:000000",
                            chunk_index=0,
                            content="alpha alpha beta",
                        ),
                        ChunkUpsert(
                            id="doc-1:000001",
                            chunk_index=1,
                            content="alpha beta gamma",
                        ),
                    ],
                )
                store.dense_hits_override = [
                    DenseSearchHit(
                        chunk_id="doc-1:000001",
                        document_id="doc-1",
                        chunk_index=1,
                        score=0.99,
                    ),
                    DenseSearchHit(
                        chunk_id="doc-1:000000",
                        document_id="doc-1",
                        chunk_index=0,
                        score=0.98,
                    ),
                ]

                response = client.post(
                    "/api/query/hybrid",
                    json={"query": "alpha beta", "top_k": 2},
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "hybrid")
            self.assertEqual([hit["chunk_id"] for hit in payload["hits"]], ["doc-1:000000", "doc-1:000001"])
            self.assertTrue(all(hit["score"] > 0.0 for hit in payload["hits"]))
            self.assertEqual(payload["hits"][0]["content"], "alpha alpha beta")

    def _seed_document_chunks(
        self,
        db_path: str,
        *,
        document_id: str,
        filename: str,
        chunks: list[ChunkUpsert],
    ) -> None:
        insert_document(
            db_path,
            document_id=document_id,
            filename=filename,
            source_type="txt",
            source_path=f"/tmp/{filename}",
            size_bytes=123,
            checksum=f"checksum-{document_id}",
            status="success",
        )
        replace_document_chunks(
            db_path,
            document_id=document_id,
            chunks=chunks,
        )
