"""
Retrieval tests for lexical search, RRF fusion, and hybrid orchestration.

SQLite FTS5 is exercised against temporary databases. Dense retrieval uses small
stub clients/stores, so no Qdrant service or embedding API is needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from helpers import insert_test_document, seed_document_chunks, two_ranked_chunks
from app.embeddings.adapter import EmbeddingBatchResult, EmbeddingItemResult
from app.retrieval.service import DEFAULT_RRF_K, HybridRetriever, RetrievedChunk, fuse_ranked_chunks_rrf
from app.storage.indexing_repository import (
    ChunkUpsert,
    normalize_fts5_query,
    replace_document_chunks,
    search_chunks_lexical,
)
from app.storage.qdrant_store import DenseSearchHit
from app.storage.sqlite_schema import initialize_sqlite_schema


class _StubEmbeddingClient:
    """Embedding fake for dense side of hybrid retrieval tests."""

    provider = "local"
    model = "test-embedding-model"

    def __init__(self, vector: list[float] | None = None) -> None:
        """Store the vector returned for every embedded input."""
        self.vector = vector or [0.1] * 8

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Return the configured vector for each text."""
        return EmbeddingBatchResult(
            provider=self.provider,
            model=self.model,
            items=[
                EmbeddingItemResult(index=index, text=text, vector=list(self.vector))
                for index, text in enumerate(texts)
            ],
        )


class _StubQdrantStore:
    """Qdrant fake that returns pre-seeded dense hits."""

    def __init__(self, dense_hits: list[DenseSearchHit] | None = None) -> None:
        """Store dense hits that search_dense should return."""
        self.dense_hits = dense_hits or []

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        """Pretend collection validation succeeds."""
        return None

    def search_dense(
        self,
        *,
        collection_name: str,
        query_vector: list[float],
        limit: int,
    ) -> list[DenseSearchHit]:
        """Return configured dense hits up to the requested limit."""
        return self.dense_hits[:limit]


class TestRetrieval(unittest.TestCase):
    """Lexical, RRF, and hybrid retrieval behavior tests."""

    def test_normalize_fts5_query_quotes_terms_and_handles_empty_input(self) -> None:
        """FTS5 query normalization should quote terms and drop empty input."""
        self.assertEqual(normalize_fts5_query("alpha, beta! gamma?"), '"alpha" AND "beta" AND "gamma"')
        self.assertIsNone(normalize_fts5_query("... !!! ---"))

    def test_search_chunks_lexical_returns_ranked_hydrated_rows(self) -> None:
        """Lexical search should rank and hydrate rows from SQLite FTS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            insert_test_document(db_path, document_id="doc-1", filename="notes.txt")
            replace_document_chunks(
                db_path,
                document_id="doc-1",
                chunks=[
                    ChunkUpsert(id="doc-1:000000", chunk_index=0, content="alpha alpha beta"),
                    ChunkUpsert(id="doc-1:000001", chunk_index=1, content="alpha beta gamma"),
                    ChunkUpsert(id="doc-1:000002", chunk_index=2, content="gamma delta epsilon"),
                ],
            )

            rows = search_chunks_lexical(db_path, query_text="alpha beta", limit=5)

            self.assertEqual([row.chunk_id for row in rows], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual(rows[0].document_id, "doc-1")
            self.assertEqual(rows[0].filename, "notes.txt")
            self.assertEqual(rows[0].chunk_index, 0)
            self.assertEqual(rows[0].content, "alpha alpha beta")
            self.assertLess(rows[0].raw_score, rows[1].raw_score)

    def test_search_chunks_lexical_tiebreaker_empty_query_and_or_fallback(self) -> None:
        """Lexical search should handle ties, empty terms, and OR fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            insert_test_document(db_path, document_id="doc-1", filename="a.txt")
            insert_test_document(db_path, document_id="doc-2", filename="b.txt")
            replace_document_chunks(
                db_path,
                document_id="doc-1",
                chunks=[ChunkUpsert(id="doc-1:000000", chunk_index=0, content="shared lexical token")],
            )
            replace_document_chunks(
                db_path,
                document_id="doc-2",
                chunks=[ChunkUpsert(id="doc-2:000000", chunk_index=0, content="shared lexical token")],
            )

            tied_rows = search_chunks_lexical(db_path, query_text="shared lexical", limit=5)
            empty_rows = search_chunks_lexical(db_path, query_text="!!!", limit=5)
            fallback_rows = search_chunks_lexical(db_path, query_text="What does shared mean?", limit=5)

            self.assertEqual([row.chunk_id for row in tied_rows], ["doc-1:000000", "doc-2:000000"])
            self.assertEqual(empty_rows, [])
            self.assertEqual([row.chunk_id for row in fallback_rows], ["doc-1:000000", "doc-2:000000"])

    def test_fuse_ranked_chunks_rrf_accumulates_scores_and_uses_tiebreakers(self) -> None:
        """RRF should accumulate duplicate chunks and use stable tie ordering."""
        dense_results = [
            self._chunk(chunk_id="chunk-a", source_id="S1", chunk_index=0, content="dense-a"),
            self._chunk(chunk_id="chunk-b", source_id="S2", chunk_index=1, content="dense-b"),
        ]
        lexical_results = [
            self._chunk(chunk_id="chunk-b", source_id="S1", chunk_index=1, content="lexical-b"),
            self._chunk(chunk_id="chunk-c", source_id="S2", chunk_index=2, content="lexical-c"),
        ]
        tie_dense = [
            self._chunk(chunk_id="chunk-a", source_id="S1", chunk_index=0, content="dense-a"),
            self._chunk(chunk_id="chunk-b", source_id="S2", chunk_index=1, content="dense-b"),
        ]
        tie_lexical = [
            self._chunk(chunk_id="chunk-b", source_id="S1", chunk_index=1, content="lexical-b"),
            self._chunk(chunk_id="chunk-a", source_id="S2", chunk_index=0, content="lexical-a"),
        ]

        fused_results = fuse_ranked_chunks_rrf(dense_results, lexical_results, limit=5, rrf_k=DEFAULT_RRF_K)
        tied_results = fuse_ranked_chunks_rrf(tie_dense, tie_lexical, limit=5, rrf_k=DEFAULT_RRF_K)

        self.assertEqual([chunk.chunk_id for chunk in fused_results], ["chunk-b", "chunk-a", "chunk-c"])
        self.assertEqual([chunk.source_id for chunk in fused_results], ["S1", "S2", "S3"])
        self.assertAlmostEqual(
            fused_results[0].score,
            (1.0 / (DEFAULT_RRF_K + 2)) + (1.0 / (DEFAULT_RRF_K + 1)),
        )
        self.assertEqual(fused_results[0].content, "dense-b")
        self.assertEqual([chunk.chunk_id for chunk in tied_results], ["chunk-a", "chunk-b"])

    def test_hybrid_retriever_merges_dense_lexical_and_single_sided_results(self) -> None:
        """HybridRetriever should work with both sides or only one side returning hits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))
            seed_document_chunks(db_path, chunks=two_ranked_chunks())

            hybrid = HybridRetriever(
                db_path=db_path,
                embedding_client=_StubEmbeddingClient(),
                qdrant_store=_StubQdrantStore(
                    dense_hits=[
                        DenseSearchHit(chunk_id="doc-1:000001", document_id="doc-1", chunk_index=1, score=0.99),
                        DenseSearchHit(chunk_id="doc-1:000000", document_id="doc-1", chunk_index=0, score=0.98),
                    ]
                ),
                qdrant_collection="documents",
                qdrant_vector_size=8,
            )
            dense_only = HybridRetriever(
                db_path=db_path,
                embedding_client=_StubEmbeddingClient(),
                qdrant_store=_StubQdrantStore(
                    dense_hits=[
                        DenseSearchHit(chunk_id="doc-1:000000", document_id="doc-1", chunk_index=0, score=0.99)
                    ]
                ),
                qdrant_collection="documents",
                qdrant_vector_size=8,
            )
            lexical_only = HybridRetriever(
                db_path=db_path,
                embedding_client=_StubEmbeddingClient(),
                qdrant_store=_StubQdrantStore(dense_hits=[]),
                qdrant_collection="documents",
                qdrant_vector_size=8,
            )

            fused_results = hybrid.retrieve(query="alpha beta", top_k=2)
            dense_only_results = dense_only.retrieve(query="!!!", top_k=3)
            lexical_only_results = lexical_only.retrieve(query="alpha beta", top_k=3)

            self.assertEqual([chunk.chunk_id for chunk in fused_results], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual([chunk.source_id for chunk in fused_results], ["S1", "S2"])
            self.assertEqual(fused_results[0].filename, "notes.txt")
            self.assertAlmostEqual(
                fused_results[0].score,
                (1.0 / (DEFAULT_RRF_K + 2)) + (1.0 / (DEFAULT_RRF_K + 1)),
            )
            self.assertEqual([chunk.chunk_id for chunk in dense_only_results], ["doc-1:000000"])
            self.assertEqual([chunk.chunk_id for chunk in lexical_only_results], ["doc-1:000000", "doc-1:000001"])

    def _chunk(
        self,
        *,
        chunk_id: str,
        source_id: str,
        chunk_index: int,
        content: str,
    ) -> RetrievedChunk:
        """Build a RetrievedChunk with common document metadata."""
        return RetrievedChunk(
            source_id=source_id,
            chunk_id=chunk_id,
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=chunk_index,
            score=0.9,
            content=content,
        )
