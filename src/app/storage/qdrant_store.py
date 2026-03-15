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

# QdrantConnectionStatus decorator: Status container for connectivity checks. https://docs.python.org/3/library/dataclasses.html for details.
@dataclass(slots=True)
# QdrantConnectionStatus: normalized result payload returned by connection checks.
class QdrantConnectionStatus:
    reachable: bool
    error: str | None = None


# ChunkVector: normalized vector attribute payload passed from indexing pipeline.
@dataclass(slots=True)
class ChunkVector:
    chunk_id: str
    document_id: str
    chunk_index: int
    vector: list[float]


# DenseSearchHit: normalized dense retrieval output used by query endpoint. This serves as an abstraction to isolate the rest of the app from
# qdrant-specific response formats.
# Includes minimal metadata for retrieval-augmented generation and further processing.
@dataclass(slots=True)
class DenseSearchHit:
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float

# QdrantStore: small wrapper for QdrantClient. 
# Used by app/script health checks.
class QdrantStore:
    # __init__: stores the already-created Qdrant client instance.
    def __init__(self, client: QdrantClient):
        self._client = client

    # from_settings decorator: exposes factory construction.
    @classmethod
    # from_settings: builds a configured Qdrant client using env-backed settings. See get_settings() in src/app/core/settings.py for details on how settings are loaded.
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )
        return cls(client)

    # check_connection: probes Qdrant and returns a status object instead of raising.
    def check_connection(self) -> QdrantConnectionStatus:
        try:
            self._client.get_collections()
            return QdrantConnectionStatus(reachable=True)
        except Exception as exc:  # pragma: no cover - wrapped for app-level handling
            return QdrantConnectionStatus(reachable=False, error=str(exc))

    # ping: convenience boolean check for quick reachability checks.
    def ping(self) -> bool:
        return self.check_connection().reachable

    # ensure_collection: create the collection if missing and force deterministic vector size if present.
    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
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

    # upsert_chunk_vectors: update and insert chunk vectors with minimal payload metadata for dense retrieval.
    def upsert_chunk_vectors(
        self,
        *,
        collection_name: str,
        vectors: list[ChunkVector],
    ) -> None:
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

    # search_dense: Dense similarity search and normalize result payload.
    # We return a simplified list of hits. The vector data is not included,
    # the retrieved text is accessed via the chunk_id/document_id metadata.
    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
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


# _extract_vector_size: read vector size from collection config handling single/multi-vector formats.
def _extract_vector_size(collection: models.CollectionInfo) -> int:
    vectors_config = collection.config.params.vectors
    if isinstance(vectors_config, models.VectorParams):
        return int(vectors_config.size)

    if isinstance(vectors_config, dict):
        for _, params in vectors_config.items():
            if isinstance(params, models.VectorParams):
                return int(params.size)

    raise ValueError("Unable to determine Qdrant collection vector size from current config.")
