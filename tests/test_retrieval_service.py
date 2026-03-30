"""
Author: Patrik Kiseda
File: tests/test_retrieval_service.py
Description: Unit tests for RRF fusion and hybrid retrieval orchestration.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.embeddings.adapter import EmbeddingBatchResult, EmbeddingItemResult
from app.retrieval.service import (
    DEFAULT_RRF_K,
    HybridRetriever,
    RetrievedChunk,
    fuse_ranked_chunks_rrf,
)
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import ChunkUpsert, replace_document_chunks
from app.storage.qdrant_store import DenseSearchHit
from app.storage.sqlite_schema import initialize_sqlite_schema


class _StubEmbeddingClient:
    provider = "local"
    model = "test-embedding-model"

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1] * 8

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        return EmbeddingBatchResult(
            provider=self.provider,
            model=self.model,
            items=[
                EmbeddingItemResult(index=index, text=text, vector=list(self.vector))
                for index, text in enumerate(texts)
            ],
        )


class _StubQdrantStore:
    def __init__(self, dense_hits: list[DenseSearchHit] | None = None) -> None:
        self.dense_hits = dense_hits or []

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        return None

    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
        return self.dense_hits[:limit]


class TestRetrievalService(unittest.TestCase):
    def test_fuse_ranked_chunks_rrf_accumulates_scores_and_reassigns_source_ids(self) -> None:
        dense_results = [
            self._chunk(chunk_id="chunk-a", source_id="S1", chunk_index=0, content="dense-a"),
            self._chunk(chunk_id="chunk-b", source_id="S2", chunk_index=1, content="dense-b"),
        ]
        lexical_results = [
            self._chunk(chunk_id="chunk-b", source_id="S1", chunk_index=1, content="lexical-b"),
            self._chunk(chunk_id="chunk-c", source_id="S2", chunk_index=2, content="lexical-c"),
        ]

        fused_results = fuse_ranked_chunks_rrf(
            dense_results,
            lexical_results,
            limit=5,
            rrf_k=DEFAULT_RRF_K,
        )

        self.assertEqual([chunk.chunk_id for chunk in fused_results], ["chunk-b", "chunk-a", "chunk-c"])
        self.assertEqual([chunk.source_id for chunk in fused_results], ["S1", "S2", "S3"])
        self.assertAlmostEqual(
            fused_results[0].score,
            (1.0 / (DEFAULT_RRF_K + 2)) + (1.0 / (DEFAULT_RRF_K + 1)),
        )
        self.assertEqual(fused_results[0].content, "dense-b")

    def test_fuse_ranked_chunks_rrf_uses_chunk_id_tiebreaker_when_scores_tie(self) -> None:
        dense_results = [
            self._chunk(chunk_id="chunk-a", source_id="S1", chunk_index=0, content="dense-a"),
            self._chunk(chunk_id="chunk-b", source_id="S2", chunk_index=1, content="dense-b"),
        ]
        lexical_results = [
            self._chunk(chunk_id="chunk-b", source_id="S1", chunk_index=1, content="lexical-b"),
            self._chunk(chunk_id="chunk-a", source_id="S2", chunk_index=0, content="lexical-a"),
        ]

        fused_results = fuse_ranked_chunks_rrf(dense_results, lexical_results, limit=5, rrf_k=DEFAULT_RRF_K)

        self.assertEqual([chunk.chunk_id for chunk in fused_results], ["chunk-a", "chunk-b"])

    def test_hybrid_retriever_merges_dense_and_lexical_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            self._insert_document(db_path, document_id="doc-1", filename="notes.txt")
            replace_document_chunks(
                db_path,
                document_id="doc-1",
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

            retriever = HybridRetriever(
                db_path=db_path,
                embedding_client=_StubEmbeddingClient(),
                qdrant_store=_StubQdrantStore(
                    dense_hits=[
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
                ),
                qdrant_collection="documents",
                qdrant_vector_size=8,
            )

            results = retriever.retrieve(query="alpha beta", top_k=2)

            self.assertEqual([chunk.chunk_id for chunk in results], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual([chunk.source_id for chunk in results], ["S1", "S2"])
            self.assertEqual(results[0].filename, "notes.txt")
            self.assertEqual(results[0].content, "alpha alpha beta")
            self.assertAlmostEqual(
                results[0].score,
                (1.0 / (DEFAULT_RRF_K + 2)) + (1.0 / (DEFAULT_RRF_K + 1)),
            )

    def test_hybrid_retriever_returns_dense_only_results_when_lexical_side_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            self._insert_document(db_path, document_id="doc-1", filename="notes.txt")
            replace_document_chunks(
                db_path,
                document_id="doc-1",
                chunks=[
                    ChunkUpsert(
                        id="doc-1:000000",
                        chunk_index=0,
                        content="alpha beta gamma",
                    )
                ],
            )

            retriever = HybridRetriever(
                db_path=db_path,
                embedding_client=_StubEmbeddingClient(),
                qdrant_store=_StubQdrantStore(
                    dense_hits=[
                        DenseSearchHit(
                            chunk_id="doc-1:000000",
                            document_id="doc-1",
                            chunk_index=0,
                            score=0.99,
                        )
                    ]
                ),
                qdrant_collection="documents",
                qdrant_vector_size=8,
            )

            results = retriever.retrieve(query="!!!", top_k=3)

            self.assertEqual([chunk.chunk_id for chunk in results], ["doc-1:000000"])
            self.assertEqual(results[0].source_id, "S1")
            self.assertAlmostEqual(results[0].score, 1.0 / (DEFAULT_RRF_K + 1))

    def test_hybrid_retriever_returns_lexical_only_results_when_dense_side_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            self._insert_document(db_path, document_id="doc-1", filename="notes.txt")
            replace_document_chunks(
                db_path,
                document_id="doc-1",
                chunks=[
                    ChunkUpsert(
                        id="doc-1:000000",
                        chunk_index=0,
                        content="alpha beta gamma",
                    )
                ],
            )

            retriever = HybridRetriever(
                db_path=db_path,
                embedding_client=_StubEmbeddingClient(),
                qdrant_store=_StubQdrantStore(dense_hits=[]),
                qdrant_collection="documents",
                qdrant_vector_size=8,
            )

            results = retriever.retrieve(query="alpha beta", top_k=3)

            self.assertEqual([chunk.chunk_id for chunk in results], ["doc-1:000000"])
            self.assertEqual(results[0].source_id, "S1")
            self.assertAlmostEqual(results[0].score, 1.0 / (DEFAULT_RRF_K + 1))

    def _chunk(
        self,
        *,
        chunk_id: str,
        source_id: str,
        chunk_index: int,
        content: str,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            source_id=source_id,
            chunk_id=chunk_id,
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=chunk_index,
            score=0.9,
            content=content,
        )

    def _insert_document(self, db_path: str, *, document_id: str, filename: str) -> None:
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
