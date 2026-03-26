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


# RetrievedChunk: normalized enriched retrieval result shared by query + generation.
@dataclass(slots=True)
class RetrievedChunk:
    source_id: str
    chunk_id: str
    document_id: str
    filename: str | None
    chunk_index: int
    score: float
    content: str


# Retriever: retrieval adapter contract for dense/lexical/hybrid modes. For now we only have dense implemented
# TODO - add lexical and hybrid retriever implementations and selection in build_retriever.
class Retriever(Protocol):
    mode: RetrievalMode

    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
        ... 

# DenseRetriever: implementation of Retriever for dense vector search using Qdrant and SQLite for chunk enrichment.
@dataclass(slots=True)
class DenseRetriever:
    db_path: str
    embedding_client: EmbeddingClient
    qdrant_store: QdrantStore
    qdrant_collection: str
    qdrant_vector_size: int
    mode: RetrievalMode = "dense"

    # retrieve: run dense embedding + Qdrant search and hydrate the chunk content from SQLite.
    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
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

# LexicalRetriever: implementation of Retriever for lexical search using SQLite FTS5.
@dataclass(slots=True)
class LexicalRetriever:
    db_path: str
    mode: RetrievalMode = "lexical"

    # retrieve: run SQLite FTS5 lexical search and normalize the hits.
    def retrieve(self, *, query: str, top_k: int) -> list[RetrievedChunk]:
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


# build_retriever: select the active retriever while reserving lexical/hybrid for later work.
def build_retriever(
    *,
    mode: RetrievalMode,
    db_path: str,
    embedding_client: EmbeddingClient,
    qdrant_store: QdrantStore,
    qdrant_collection: str,
    qdrant_vector_size: int,
) -> Retriever:  
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

    raise HTTPException(
        status_code=501,
        detail=f"Retrieval mode '{mode}' is not implemented yet.",
    )
