"""
Author: Patrik Kiseda
File: src/app/retrieval/service.py
Description: Shared retrieval contract and dense retriever selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from fastapi import HTTPException

from app.embeddings.adapter import EmbeddingClient
from app.storage.indexing_repository import get_chunks_by_ids, search_chunks_lexical
from app.storage.qdrant_store import QdrantStore

RetrievalMode = Literal["dense", "lexical", "hybrid"]
# DEFAULT_RRF_K: default "k" parameter for Reciprocal Rank Fusion.
# Controlling how quickly the rank contribution decays.
DEFAULT_RRF_K = 60


@dataclass(slots=True)
class RetrievedChunk:
    """Normalized enriched retrieval result shared by query and generation."""

    source_id: str
    chunk_id: str
    document_id: str
    filename: str | None
    chunk_index: int
    score: float
    content: str


class Retriever(Protocol):
    """Retrieval adapter contract for dense, lexical, and hybrid modes."""

    mode: RetrievalMode

    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve chunks for a query.

        Args:
            query: User query text.
            top_k: Number of chunks to return.

        Returns:
            Retrieved chunks.
        """
        ...


@dataclass(slots=True)
class _FusedChunkCandidate:
    """Internal helper to track chunk candidates during RRF fusion."""

    chunk: RetrievedChunk
    fused_score: float
    best_rank: int


def fuse_ranked_chunks_rrf(
    # ranked_lists: variable number of ranked chunk lists to fuse.
    *ranked_lists: list[RetrievedChunk],
    limit: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[RetrievedChunk]:
    """Combine ranked lists using rank positions, not raw score magnitudes.

    Args:
        *ranked_lists: Ranked chunk lists to fuse.
        limit: Maximum number of fused results.
        rrf_k: Reciprocal Rank Fusion k parameter.

    Returns:
        Fused retrieved chunks.
    """
    if limit <= 0:
        return []

    fused_by_chunk_id: dict[str, _FusedChunkCandidate] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # Calculate the reciprocal rank contribution for this chunk in this list.
            reciprocal_rank = 1.0 / float(rrf_k + rank)
            existing = fused_by_chunk_id.get(chunk.chunk_id)
            if existing is None:
                fused_by_chunk_id[chunk.chunk_id] = _FusedChunkCandidate(
                    chunk=RetrievedChunk(
                        source_id=chunk.source_id,
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        chunk_index=chunk.chunk_index,
                        score=chunk.score,
                        content=chunk.content,
                    ),
                    fused_score=reciprocal_rank,
                    best_rank=rank,
                )
                continue

            existing.fused_score += reciprocal_rank
            existing.best_rank = min(existing.best_rank, rank)

    ranked_candidates = sorted(
        fused_by_chunk_id.values(),
        key=lambda candidate: (
            -candidate.fused_score,
            candidate.best_rank,
            candidate.chunk.chunk_id,
        ),
    )

    fused_results: list[RetrievedChunk] = []
    for index, candidate in enumerate(ranked_candidates[:limit], start=1):
        fused_results.append(
            RetrievedChunk(
                source_id=f"S{index}",
                chunk_id=candidate.chunk.chunk_id,
                document_id=candidate.chunk.document_id,
                filename=candidate.chunk.filename,
                chunk_index=candidate.chunk.chunk_index,
                score=candidate.fused_score,
                content=candidate.chunk.content,
            )
        )

    return fused_results

@dataclass(slots=True)
class DenseRetriever:
    """Retriever for dense vector search using Qdrant and SQLite enrichment."""

    db_path: str
    embedding_client: EmbeddingClient
    qdrant_store: QdrantStore
    qdrant_collection: str
    qdrant_vector_size: int
    mode: RetrievalMode = "dense"

    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run dense embedding + Qdrant search and hydrate chunk content.

        Args:
            query: User query text.
            top_k: Number of chunks to return.

        Returns:
            Retrieved chunks with text content.
        """
        embedding_result = self.embedding_client.embed_texts([query])
        if not embedding_result.items or not embedding_result.items[0].is_success:
            raise HTTPException(status_code=502, detail="Failed to generate query embedding.")
   
        query_item = embedding_result.items[0]
        # Validate the embeding vector size
        assert query_item.vector is not None

        if len(query_item.vector) != self.qdrant_vector_size:
            raise HTTPException(
                status_code=500,
                detail="Query embedding vector size does not match configured Qdrant vector size.",
            )

        self.qdrant_store.ensure_collection(
            collection_name=self.qdrant_collection,
            vector_size=self.qdrant_vector_size,
        )
        dense_hits = self.qdrant_store.search_dense(
            collection_name=self.qdrant_collection,
            query_vector=query_item.vector,
            limit=top_k,
        )
        if not dense_hits:
            return []

        chunks_by_id = get_chunks_by_ids(
            self.db_path,
            chunk_ids=[hit.chunk_id for hit in dense_hits],
        )

        retrieved_chunks: list[RetrievedChunk] = []
        for hit in dense_hits:
            chunk = chunks_by_id.get(hit.chunk_id)
            if chunk is None:
                continue

            retrieved_chunks.append(
                RetrievedChunk(
                    source_id=f"S{len(retrieved_chunks) + 1}",
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    filename=chunk.filename,
                    chunk_index=hit.chunk_index,
                    score=hit.score,
                    content=chunk.content,
                )
            )

        return retrieved_chunks

@dataclass(slots=True)
class LexicalRetriever:
    """Retriever for lexical search using SQLite FTS5."""

    db_path: str
    mode: RetrievalMode = "lexical"

    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run SQLite FTS5 lexical search and normalize hits.

        Args:
            query: User query text.
            top_k: Number of chunks to return.

        Returns:
            Retrieved chunks.
        """
        lexical_rows = search_chunks_lexical(
            self.db_path,
            query_text=query,
            limit=top_k,
        )

        return [
            RetrievedChunk(
                source_id=f"S{index}",
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                filename=row.filename,
                chunk_index=row.chunk_index,
                score=max(0.0, -row.raw_score),
                content=row.content,
            )
            for index, row in enumerate(lexical_rows, start=1)
        ]


@dataclass(slots=True)
class HybridRetriever:
    """Retriever that runs dense and lexical search, then fuses ranks."""

    db_path: str
    embedding_client: EmbeddingClient
    qdrant_store: QdrantStore
    qdrant_collection: str
    qdrant_vector_size: int
    rrf_k: int = DEFAULT_RRF_K
    mode: RetrievalMode = "hybrid"

    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        """Run both retrievers, fuse by chunk rank, and return shared shape.

        Args:
            query: User query text.
            top_k: Number of chunks to return.

        Returns:
            Fused retrieved chunks.
        """
        dense_results = DenseRetriever(
            db_path=self.db_path,
            embedding_client=self.embedding_client,
            qdrant_store=self.qdrant_store,
            qdrant_collection=self.qdrant_collection,
            qdrant_vector_size=self.qdrant_vector_size,
        ).retrieve(query=query, top_k=top_k)
        lexical_results = LexicalRetriever(db_path=self.db_path).retrieve(query=query, top_k=top_k)

        return fuse_ranked_chunks_rrf(
            dense_results,
            lexical_results,
            limit=top_k,
            rrf_k=self.rrf_k,
        )


def build_retriever(
    *,
    mode: RetrievalMode,
    db_path: str,
    embedding_client: EmbeddingClient,
    qdrant_store: QdrantStore,
    qdrant_collection: str,
    qdrant_vector_size: int,
) -> Retriever:  
    """Select the active retriever for dense, lexical, or hybrid modes.

    Args:
        mode: Retrieval mode to build.
        db_path: SQLite database path.
        embedding_client: Embedding client for dense modes.
        qdrant_store: Qdrant wrapper for dense modes.
        qdrant_collection: Qdrant collection name.
        qdrant_vector_size: Expected dense vector size.

    Returns:
        Configured retriever.
    """
    if mode == "dense":
        return DenseRetriever(
            db_path=db_path,
            embedding_client=embedding_client,
            qdrant_store=qdrant_store,
            qdrant_collection=qdrant_collection,
            qdrant_vector_size=qdrant_vector_size,
    )
    if mode == "lexical":
        return LexicalRetriever(db_path=db_path)
    if mode == "hybrid":
        return HybridRetriever(
            db_path=db_path,
            embedding_client=embedding_client,
            qdrant_store=qdrant_store,
            qdrant_collection=qdrant_collection,
            qdrant_vector_size=qdrant_vector_size,
        )

    raise HTTPException(
        status_code=501,
        detail=f"Retrieval mode '{mode}' is not implemented yet.",
    )
