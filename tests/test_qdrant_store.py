"""
Author: Patrik Kiseda
File: tests/test_qdrant_store.py
Description: Unit tests for QdrantStore connectivity, dense upsert, and dense search mapping.

All Qdrant interactions are fake in-memory clients. The tests validate wrapper
behavior, client construction arguments, collection size checks, payload mapping,
and dense-hit normalization without needing a running Qdrant container.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import models

from helpers import build_settings
from app.storage.qdrant_store import ChunkVector, QdrantStore


class _WorkingClient:
    """In-memory Qdrant client fake for successful wrapper paths."""

    def __init__(self) -> None:
        """Initialize default vector schema and recorded upserts."""
        self._vectors = models.VectorParams(size=8, distance=models.Distance.COSINE)
        self.created = False
        self.upsert_calls: list[tuple[str, list[models.PointStruct], bool]] = []

    def get_collections(self) -> object:
        """Pretend Qdrant collection listing succeeds."""
        return object()

    def get_collection(self, collection_name: str) -> object:
        """Return a minimal collection info-like object."""
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=self._vectors))
        )

    def create_collection(
        self,
        *,
        collection_name: str,
        vectors_config: models.VectorParams,
    ) -> None:
        """Record that the collection was created with the given schema."""
        self.created = True
        self._vectors = vectors_config

    def upsert(
        self,
        *,
        collection_name: str,
        points: list[models.PointStruct],
        wait: bool,
    ) -> None:
        """Record upsert arguments for payload assertions."""
        self.upsert_calls.append((collection_name, points, wait))

    def query_points(
        self,
        *,
        collection_name: str,
        query: list[float],
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> object:
        """Return one deterministic scored point for dense mapping checks."""
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


class _BrokenClient:
    """Qdrant client fake that raises during connection checks."""

    def get_collections(self) -> object:
        """Simulate a connection failure."""
        raise RuntimeError("connection refused")


class _MissingCollectionClient(_WorkingClient):
    """Working client variant where get_collection reports missing collection."""

    def get_collection(self, collection_name: str) -> models.CollectionInfo:
        """Force ensure_collection down the create_collection path."""
        raise RuntimeError("collection not found")


class TestQdrantStore(unittest.TestCase):
    """QdrantStore wrapper behavior tests around client calls and mapping."""

    def test_check_connection_success(self) -> None:
        """check_connection should return reachable when client call succeeds."""
        store = QdrantStore(client=_WorkingClient())  # type: ignore[arg-type]
        status = store.check_connection()
        self.assertTrue(status.reachable)
        self.assertIsNone(status.error)

    def test_check_connection_failure(self) -> None:
        """check_connection should return unreachable instead of raising."""
        store = QdrantStore(client=_BrokenClient())  # type: ignore[arg-type]
        status = store.check_connection()
        self.assertFalse(status.reachable)
        self.assertIn("connection refused", status.error or "")

    def test_from_settings_uses_expected_client_configuration(self) -> None:
        """from_settings should pass URL, API key, and timeout to QdrantClient."""
        settings = build_settings(
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
        """ensure_collection should create the collection when lookup fails."""
        client = _MissingCollectionClient()
        store = QdrantStore(client=client)  # type: ignore[arg-type]

        store.ensure_collection(collection_name="documents", vector_size=8)

        self.assertTrue(client.created)
        self.assertEqual(client._vectors.size, 8)

    def test_ensure_collection_rejects_size_mismatch(self) -> None:
        """ensure_collection should reject existing collections with wrong size."""
        client = _WorkingClient()
        client._vectors = models.VectorParams(size=16, distance=models.Distance.COSINE)
        store = QdrantStore(client=client)  # type: ignore[arg-type]

        with self.assertRaises(ValueError) as ctx:
            store.ensure_collection(collection_name="documents", vector_size=8)

        self.assertIn("vector size mismatch", str(ctx.exception))

    def test_upsert_chunk_vectors_sends_minimal_payload(self) -> None:
        """upsert_chunk_vectors should send stable point ids and minimal payload."""
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
        """search_dense should map Qdrant scored points into DenseSearchHit."""
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
