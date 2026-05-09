"""
Author: Patrik Kiseda
File: src/app/storage/qdrant_store.py
Description: Small Qdrant client wrapper with connection check support.
"""

from __future__ import annotations
from dataclasses import dataclass
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models

from app.core.settings import Settings

@dataclass(slots=True)
class QdrantConnectionStatus:
    """Normalized result payload returned by connection checks."""

    reachable: bool
    error: str | None = None


@dataclass(slots=True)
class ChunkVector:
    """Normalized vector payload passed from indexing pipeline."""

    chunk_id: str
    document_id: str
    chunk_index: int
    vector: list[float]


@dataclass(slots=True)
class DenseSearchHit:
    """Normalized dense retrieval hit, hiding Qdrant response details."""

    chunk_id: str
    document_id: str
    chunk_index: int
    score: float

class QdrantStore:
    """Small wrapper for QdrantClient used by app and scripts."""

    def __init__(self, client: QdrantClient):
        """Store the already-created Qdrant client instance.

        Args:
            client: Qdrant client instance.
        """
        self._client = client

    @classmethod
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        """Build a configured Qdrant client using env-backed settings.

        Args:
            settings: Runtime settings.

        Returns:
            Configured Qdrant store wrapper.
        """
        client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )
        return cls(client)

    def check_connection(self) -> QdrantConnectionStatus:
        """Probe Qdrant and return a status object instead of raising.

        Returns:
            Connection status.
        """
        try:
            self._client.get_collections()
            return QdrantConnectionStatus(reachable=True)
        except Exception as exc:  # pragma: no cover - wrapped for app-level handling
            return QdrantConnectionStatus(reachable=False, error=str(exc))

    def ping(self) -> bool:
        """Quick boolean reachability check.

        Returns:
            True when Qdrant is reachable.
        """
        return self.check_connection().reachable

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        """Create collection if missing and validate vector size if present.

        Args:
            collection_name: Qdrant collection name.
            vector_size: Expected vector size.
        """
        try:
            collection = self._client.get_collection(collection_name=collection_name)
        except Exception:
            # If the collection does not exist, create it with the specified vector size and deterministic schema.
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        # If the collection exists, validate that the vector size matches the expected configuration to avoid schema mismatches.
        configured_size = _extract_vector_size(collection)
        if configured_size != vector_size:
            raise ValueError(
                f"Qdrant collection '{collection_name}' vector size mismatch: "
                f"configured={configured_size}, expected={vector_size}."
            )

    def upsert_chunk_vectors(
        self,
        *,
        collection_name: str,
        vectors: list[ChunkVector],
    ) -> None:
        """Insert or update chunk vectors with minimal retrieval metadata.

        Args:
            collection_name: Qdrant collection name.
            vectors: Chunk vectors to upsert.
        """
        if not vectors:
            return

        points = [
            models.PointStruct(
                # UUID-based deterministic point id avoids backend-specific string-id restrictions.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, item.chunk_id)),
                vector=item.vector,
                payload={
                    "chunk_id": item.chunk_id,
                    "document_id": item.document_id,
                    "chunk_index": item.chunk_index,
                },
            )
            for item in vectors
        ]
        self._client.upsert(collection_name=collection_name, points=points, wait=True)

    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
        """Run dense similarity search and normalize result payload.

        Args:
            collection_name: Qdrant collection name.
            query_vector: Query embedding vector.
            limit: Max hits to return.

        Returns:
            Simplified dense search hits without vector data.
        """
        response = self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        hits: list[DenseSearchHit] = []
        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")
            document_id = payload.get("document_id")
            chunk_index = payload.get("chunk_index")
            if not isinstance(chunk_id, str) or not isinstance(document_id, str):
                continue
            if not isinstance(chunk_index, int):
                continue

            hits.append(
                DenseSearchHit(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    score=float(point.score),
                )
            )
        return hits


def _extract_vector_size(collection: models.CollectionInfo) -> int:
    """Read vector size from collection config.

    Args:
        collection: Qdrant collection info.

    Returns:
        Configured vector size.
    """
    vectors_config = collection.config.params.vectors
    if isinstance(vectors_config, models.VectorParams):
        return int(vectors_config.size)

    if isinstance(vectors_config, dict):
        for _, params in vectors_config.items():
            if isinstance(params, models.VectorParams):
                return int(params.size)

    raise ValueError("Unable to determine Qdrant collection vector size from current config.")
