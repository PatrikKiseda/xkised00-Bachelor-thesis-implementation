"""
Author: Patrik Kiseda
File: tests/test_qdrant_store.py
Description: Unit tests for QdrantStore connectivity, dense upsert, and dense search mapping.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

# sys.path adjustment: allows test imports from src without package setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.settings import Settings
from app.storage.qdrant_store import ChunkVector, QdrantStore


# _build_settings: creates a validated baseline settings object for tests.
def _build_settings(**overrides: object) -> Settings:
    payload = {
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_collection": "documents",
        "qdrant_vector_size": 8,
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
    }
    payload.update(overrides)
    return Settings(**payload)


# _WorkingClient: test double that simulates successful Qdrant calls.
class _WorkingClient:
    def __init__(self) -> None:
        self._vectors = models.VectorParams(size=8, distance=models.Distance.COSINE)
        self.created = False
        self.upsert_calls: list[tuple[str, list[models.PointStruct], bool]] = []

    # get_collections: fake success path for connection checks.
    def get_collections(self) -> object:
        return object()

    # get_collection: returns in-memory collection info payload.
    def get_collection(self, collection_name: str) -> object:
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self._vectors))
        )

    # create_collection: toggles creation marker and stores vectors config.
    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: models.VectorParams,
    ) -> None:
        self.created = True
        self._vectors = vectors_config

    # upsert: record points for assertions.
    def upsert(
        self,
        *,
        collection_name: str,
        points: list[models.PointStruct],
        wait: bool,
    ) -> None:
        self.upsert_calls.append((collection_name, points, wait))

    # query_points: returns deterministic dense hit for mapping checks.
    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> object:
        assert collection_name == "documents"
        assert len(query) == 8
        assert limit == 3
        assert with_payload is True
        assert with_vectors is False
        return SimpleNamespace(
            points=[
                models.ScoredPoint(
                    id="point-id",
                    version=1,
                    score=0.77,
                    payload={
                        "chunk_id": "doc-1:000000",
                        "document_id": "doc-1",
                        "chunk_index": 0,
                    },
                )
            ]
        )


# _BrokenClient: test double that simulates failed Qdrant calls.
class _BrokenClient:
    def get_collections(self) -> object:
        raise RuntimeError("connection refused")


# _MissingCollectionClient: get_collection fails so store should create collection.
class _MissingCollectionClient(_WorkingClient):
    def get_collection(self, collection_name: str) -> models.CollectionInfo:
        raise RuntimeError("collection not found")


# TestQdrantStore: verifies wrapper behavior and client construction settings.
class TestQdrantStore(unittest.TestCase):
    def test_check_connection_success(self) -> None:
        store = QdrantStore(client=_WorkingClient())  # type: ignore[arg-type]
        status = store.check_connection()
        self.assertTrue(status.reachable)
        self.assertIsNone(status.error)

    def test_check_connection_failure(self) -> None:
        store = QdrantStore(client=_BrokenClient())  # type: ignore[arg-type]
        status = store.check_connection()
        self.assertFalse(status.reachable)
        self.assertIn("connection refused", status.error or "")

    def test_from_settings_uses_expected_client_configuration(self) -> None:
        settings = _build_settings(
            qdrant_url="http://qdrant.local:6333",
            qdrant_api_key="secret",
            qdrant_timeout_seconds=9.5,
        )

        with patch("app.storage.qdrant_store.QdrantClient") as mock_client_cls:
            QdrantStore.from_settings(settings)

        mock_client_cls.assert_called_once_with(
            url="http://qdrant.local:6333",
            api_key="secret",
            timeout=9.5,
        )

    def test_ensure_collection_creates_when_missing(self) -> None:
        client = _MissingCollectionClient()
        store = QdrantStore(client=client)  # type: ignore[arg-type]

        store.ensure_collection(collection_name="documents", vector_size=8)

        self.assertTrue(client.created)
        self.assertEqual(client._vectors.size, 8)

    def test_ensure_collection_rejects_size_mismatch(self) -> None:
        client = _WorkingClient()
        client._vectors = models.VectorParams(size=16, distance=models.Distance.COSINE)
        store = QdrantStore(client=client)  # type: ignore[arg-type]

        with self.assertRaises(ValueError) as ctx:
            store.ensure_collection(collection_name="documents", vector_size=8)

        self.assertIn("vector size mismatch", str(ctx.exception))

    def test_upsert_chunk_vectors_sends_minimal_payload(self) -> None:
        client = _WorkingClient()
        store = QdrantStore(client=client)  # type: ignore[arg-type]

        store.upsert_chunk_vectors(
            collection_name="documents",
            vectors=[
                ChunkVector(
                    chunk_id="doc-1:000000",
                    document_id="doc-1",
                    chunk_index=0,
                    vector=[0.1] * 8,
                )
            ],
        )

        self.assertEqual(len(client.upsert_calls), 1)
        collection_name, points, wait = client.upsert_calls[0]
        self.assertEqual(collection_name, "documents")
        self.assertTrue(wait)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].payload, {"chunk_id": "doc-1:000000", "document_id": "doc-1", "chunk_index": 0})

    def test_search_dense_maps_hits(self) -> None:
        store = QdrantStore(client=_WorkingClient())  # type: ignore[arg-type]

        hits = store.search_dense(
            collection_name="documents",
            query_vector=[0.2] * 8,
            limit=3,
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "doc-1:000000")
        self.assertEqual(hits[0].document_id, "doc-1")
        self.assertEqual(hits[0].chunk_index, 0)
        self.assertAlmostEqual(hits[0].score, 0.77)
