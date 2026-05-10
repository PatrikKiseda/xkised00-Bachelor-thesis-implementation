"""
Author: Patrik Kiseda
File: tests/test_indexing_pipeline.py
Description: Unit tests for background indexing pipeline with embedding outcomes.

The pipeline is exercised against temporary SQLite databases and stored files.
Embedding clients and Qdrant stores are fake so the tests can verify success,
partial failure, all-failure, Qdrant write failure, and extraction failure states
without external services.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from helpers import suppress_expected_pdf_noise
from app.embeddings.adapter import EmbeddingBatchResult, EmbeddingItemResult
from app.ingestion.indexing_pipeline import run_indexing_pipeline
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import create_job
from app.storage.sqlite_schema import initialize_sqlite_schema


class _SuccessEmbeddingClient:
    """Embedding fake where every chunk gets a valid vector."""

    provider = "local"
    model = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Return one deterministic success item per input text."""
        return EmbeddingBatchResult(
            provider=self.provider,
            model=self.model,
            items=[
                EmbeddingItemResult(index=index, text=text, vector=[float(index), 0.5])
                for index, text in enumerate(texts)
            ],
        )


class _PartialFailEmbeddingClient:
    """Embedding fake that fails every odd chunk index."""

    provider = "local"
    model = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Return mixed success/failure items to exercise warnings."""
        items: list[EmbeddingItemResult] = []
        for index, text in enumerate(texts):
            if index % 2 == 1:
                items.append(
                    EmbeddingItemResult(
                        index=index,
                        text=text,
                        vector=None,
                        error=f"simulated-failure-{index}",
                    )
                )
            else:
                items.append(
                    EmbeddingItemResult(index=index, text=text, vector=[float(index), 0.25])
                )
        return EmbeddingBatchResult(provider=self.provider, model=self.model, items=items)


class _AllFailEmbeddingClient:
    """Embedding fake where every chunk fails."""

    provider = "local"
    model = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Return failed items for all input texts."""
        return EmbeddingBatchResult(
            provider=self.provider,
            model=self.model,
            items=[
                EmbeddingItemResult(
                    index=index,
                    text=text,
                    vector=None,
                    error="simulated-all-fail",
                )
                for index, text in enumerate(texts)
            ],
        )


class _RecordingQdrantStore:
    """Qdrant fake that records collection and upsert calls."""

    def __init__(self) -> None:
        """Initialize call recording lists."""
        self.ensure_calls: list[tuple[str, int]] = []
        self.upsert_calls: list[tuple[str, int]] = []

    def ensure_collection(self, *, collection_name: str, vector_size: int) -> None:
        """Record collection validation/creation request."""
        self.ensure_calls.append((collection_name, vector_size))

    def upsert_chunk_vectors(self, *, collection_name: str, vectors: list[object]) -> None:
        """Record dense vector upsert request and vector count."""
        self.upsert_calls.append((collection_name, len(vectors)))


class _FailingQdrantStore(_RecordingQdrantStore):
    """Qdrant fake that raises during vector upsert."""

    def upsert_chunk_vectors(self, *, collection_name: str, vectors: list[object]) -> None:
        """Simulate a Qdrant write failure after SQLite persistence."""
        raise RuntimeError("simulated qdrant write failure")


class TestIndexingPipeline(unittest.TestCase):
    """Indexing pipeline success/failure state transition tests."""

    def _prepare_document_and_job(self, *, temp_path: Path, source_name: str, source_bytes: bytes) -> str:
        """Create temp DB, source file, document row, and pending job."""
        db_path = str(initialize_sqlite_schema(str(temp_path / "app.db")))
        source_path = temp_path / source_name
        source_path.write_bytes(source_bytes)

        document = insert_document(
            db_path,
            document_id="doc-1",
            filename=source_name,
            source_type=Path(source_name).suffix.lstrip("."),
            source_path=str(source_path),
            size_bytes=source_path.stat().st_size,
            checksum="checksum",
            status="pending",
        )
        create_job(
            db_path,
            job_id="job-1",
            job_type="indexing",
            document_id=document.id,
            status="pending",
        )
        return db_path

    def test_create_job_returns_complete_job_record(self) -> None:
        """create_job should map the inserted row into a complete JobRecord."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(initialize_sqlite_schema(str(Path(temp_dir) / "app.db")))

            job = create_job(
                db_path,
                job_id="job-1",
                job_type="indexing",
                document_id="doc-1",
                status="pending",
            )

            self.assertEqual(job.id, "job-1")
            self.assertEqual(job.job_type, "indexing")
            self.assertEqual(job.status, "pending")
            self.assertEqual(job.document_id, "doc-1")

    def test_success_persists_chunks_and_embedding_payload(self) -> None:
        """All-success embedding should persist chunks and success payload."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 80).encode("utf-8"),
            )
            store = _RecordingQdrantStore()

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_path=str(temp_path / "doc.txt"),
                job_id="job-1",
                chunk_size_chars=100,
                chunk_overlap_chars=15,
                embedding_client=_SuccessEmbeddingClient(),
                qdrant_store=store,
                qdrant_collection="documents",
                qdrant_vector_size=2,
            )
            self.assertEqual(outcome, "success")

            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    "SELECT status, error_message, payload_json FROM jobs WHERE id = 'job-1'"
                ).fetchone()
                chunk_rows = connection.execute(
                    """
                    SELECT id, chunk_index, content, embedding_model
                    FROM chunks
                    WHERE document_id = 'doc-1'
                    ORDER BY chunk_index ASC
                    """
                ).fetchall()
                fts_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE document_id = 'doc-1'"
                ).fetchone()[0]

            self.assertIsNotNone(job_row)
            assert job_row is not None
            self.assertEqual(job_row[0], "success")
            self.assertIsNone(job_row[1])

            payload = json.loads(job_row[2] or "{}")
            embedding_payload = payload.get("embedding", {})
            self.assertEqual(embedding_payload.get("successful_embeddings"), len(chunk_rows))
            self.assertEqual(embedding_payload.get("failed_embeddings"), 0)
            self.assertEqual(embedding_payload.get("warnings"), [])
            dense_payload = payload.get("dense_indexing", {})
            self.assertEqual(dense_payload.get("indexed_vectors"), len(chunk_rows))
            self.assertIsNone(dense_payload.get("error"))

            self.assertGreater(len(chunk_rows), 1)
            self.assertEqual(fts_count, len(chunk_rows))
            for chunk_id, chunk_index, content, embedding_model in chunk_rows:
                self.assertEqual(chunk_id, f"doc-1:{chunk_index:06d}")
                self.assertTrue(content)
                self.assertEqual(embedding_model, "test-embedding-model")
            self.assertEqual(store.ensure_calls, [("documents", 2)])
            self.assertEqual(len(store.upsert_calls), 1)

    def test_partial_embedding_failures_keep_job_success(self) -> None:
        """Partial embedding failures should keep job success with warnings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 60).encode("utf-8"),
            )
            store = _RecordingQdrantStore()

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_path=str(temp_path / "doc.txt"),
                job_id="job-1",
                chunk_size_chars=90,
                chunk_overlap_chars=15,
                embedding_client=_PartialFailEmbeddingClient(),
                qdrant_store=store,
                qdrant_collection="documents",
                qdrant_vector_size=2,
            )
            self.assertEqual(outcome, "success")

            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    "SELECT status, error_message, payload_json FROM jobs WHERE id = 'job-1'"
                ).fetchone()
                chunk_rows = connection.execute(
                    """
                    SELECT chunk_index, embedding_model
                    FROM chunks
                    WHERE document_id = 'doc-1'
                    ORDER BY chunk_index ASC
                    """
                ).fetchall()

            self.assertIsNotNone(job_row)
            assert job_row is not None
            self.assertEqual(job_row[0], "success")
            self.assertIsNone(job_row[1])

            payload = json.loads(job_row[2] or "{}")
            embedding_payload = payload.get("embedding", {})
            self.assertGreater(embedding_payload.get("failed_embeddings", 0), 0)
            self.assertGreater(len(embedding_payload.get("warnings", [])), 0)
            dense_payload = payload.get("dense_indexing", {})
            self.assertGreater(dense_payload.get("indexed_vectors", 0), 0)
            self.assertIsNone(dense_payload.get("error"))

            self.assertGreater(len(chunk_rows), 1)
            for chunk_index, embedding_model in chunk_rows:
                if chunk_index % 2 == 1:
                    self.assertIsNone(embedding_model)
                else:
                    self.assertEqual(embedding_model, "test-embedding-model")
            self.assertEqual(store.ensure_calls, [("documents", 2)])
            self.assertEqual(len(store.upsert_calls), 1)

    def test_all_embedding_failures_mark_job_fail_but_keep_chunks(self) -> None:
        """All embedding failures should fail the job but keep SQLite chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 50).encode("utf-8"),
            )
            store = _RecordingQdrantStore()

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_path=str(temp_path / "doc.txt"),
                job_id="job-1",
                chunk_size_chars=90,
                chunk_overlap_chars=15,
                embedding_client=_AllFailEmbeddingClient(),
                qdrant_store=store,
                qdrant_collection="documents",
                qdrant_vector_size=2,
            )
            self.assertEqual(outcome, "fail")

            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    "SELECT status, error_message, payload_json FROM jobs WHERE id = 'job-1'"
                ).fetchone()
                document_row = connection.execute(
                    "SELECT status FROM documents WHERE id = 'doc-1'"
                ).fetchone()
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = 'doc-1'"
                ).fetchone()[0]

            self.assertIsNotNone(job_row)
            self.assertIsNotNone(document_row)
            assert job_row is not None
            assert document_row is not None
            self.assertEqual(job_row[0], "fail")
            self.assertIn("Embedding generation failed for all chunks.", job_row[1] or "")
            self.assertEqual(document_row[0], "fail")
            self.assertGreater(chunk_count, 0)

            payload = json.loads(job_row[2] or "{}")
            embedding_payload = payload.get("embedding", {})
            self.assertEqual(embedding_payload.get("successful_embeddings"), 0)
            self.assertEqual(embedding_payload.get("failed_embeddings"), chunk_count)
            dense_payload = payload.get("dense_indexing", {})
            self.assertEqual(dense_payload.get("indexed_vectors"), 0)
            self.assertIsNone(dense_payload.get("error"))
            self.assertEqual(store.ensure_calls, [])
            self.assertEqual(store.upsert_calls, [])

    def test_qdrant_upsert_failure_marks_job_fail(self) -> None:
        """Qdrant write errors should fail after chunk persistence."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 40).encode("utf-8"),
            )

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_path=str(temp_path / "doc.txt"),
                job_id="job-1",
                chunk_size_chars=100,
                chunk_overlap_chars=15,
                embedding_client=_SuccessEmbeddingClient(),
                qdrant_store=_FailingQdrantStore(),
                qdrant_collection="documents",
                qdrant_vector_size=2,
            )
            self.assertEqual(outcome, "fail")

            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    "SELECT status, error_message, payload_json FROM jobs WHERE id = 'job-1'"
                ).fetchone()
                document_row = connection.execute(
                    "SELECT status FROM documents WHERE id = 'doc-1'"
                ).fetchone()
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = 'doc-1'"
                ).fetchone()[0]

            self.assertIsNotNone(job_row)
            self.assertIsNotNone(document_row)
            assert job_row is not None
            assert document_row is not None
            self.assertEqual(job_row[0], "fail")
            self.assertIn("Failed to index vectors in Qdrant", job_row[1] or "")
            self.assertEqual(document_row[0], "fail")
            self.assertGreater(chunk_count, 0)

            payload = json.loads(job_row[2] or "{}")
            dense_payload = payload.get("dense_indexing", {})
            self.assertEqual(dense_payload.get("indexed_vectors"), 0)
            self.assertIn("simulated qdrant write failure", dense_payload.get("error", ""))

    def test_failure_marks_job_and_document_as_fail(self) -> None:
        """Extraction errors should fail job/document before chunks are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="broken.pdf",
                source_bytes=b"not-a-real-pdf",
            )

            with suppress_expected_pdf_noise():
                outcome = run_indexing_pipeline(
                    db_path=db_path,
                    document_id="doc-1",
                    filename="broken.pdf",
                    source_path=str(temp_path / "broken.pdf"),
                    job_id="job-1",
                    chunk_size_chars=1000,
                    chunk_overlap_chars=150,
                    embedding_client=_SuccessEmbeddingClient(),
                    qdrant_store=_RecordingQdrantStore(),
                    qdrant_collection="documents",
                    qdrant_vector_size=2,
                )
            self.assertEqual(outcome, "fail")

            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    """
                    SELECT status, started_at, finished_at, error_message
                    FROM jobs
                    WHERE id = 'job-1'
                    """
                ).fetchone()
                document_row = connection.execute(
                    "SELECT status FROM documents WHERE id = 'doc-1'"
                ).fetchone()
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = 'doc-1'"
                ).fetchone()[0]

            self.assertIsNotNone(job_row)
            self.assertIsNotNone(document_row)
            assert job_row is not None
            assert document_row is not None
            self.assertEqual(job_row[0], "fail")
            self.assertIsNotNone(job_row[1])
            self.assertIsNotNone(job_row[2])
            self.assertIn("Failed to extract text from PDF.", job_row[3] or "")
            self.assertEqual(document_row[0], "fail")
            self.assertEqual(chunk_count, 0)
