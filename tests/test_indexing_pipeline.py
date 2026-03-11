"""
Author: Patrik Kiseda
File: tests/test_indexing_pipeline.py
Description: Unit tests for background indexing pipeline with embedding outcomes.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.embeddings.adapter import EmbeddingBatchResult, EmbeddingItemResult
from app.ingestion.indexing_pipeline import run_indexing_pipeline
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import create_job
from app.storage.sqlite_schema import initialize_sqlite_schema


# _SuccessEmbeddingClient: deterministic test double where all chunks embed successfully.
class _SuccessEmbeddingClient:
    provider = "local"
    model = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        return EmbeddingBatchResult(
            provider=self.provider,
            model=self.model,
            items=[
                EmbeddingItemResult(index=index, text=text, vector=[float(index), 0.5])
                for index, text in enumerate(texts)
            ],
        )


# _PartialFailEmbeddingClient: fails every odd chunk index to validate best-effort handling.
class _PartialFailEmbeddingClient:
    provider = "local"
    model = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
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


# _AllFailEmbeddingClient: fails every chunk so job should be marked as fail.
class _AllFailEmbeddingClient:
    provider = "local"
    model = "test-embedding-model"

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
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


# TestIndexingPipeline: verifies chunk persistence, embedding metadata, and job transitions.
class TestIndexingPipeline(unittest.TestCase):
    # _prepare_document_and_job: helper to avoid repeating setup boilerplate per test.
    def _prepare_document_and_job(self, *, temp_path: Path, source_name: str, source_bytes: bytes) -> str:
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

    # test_success_persists_chunks_and_embedding_payload: all-success embedding keeps success status with stats payload.
    def test_success_persists_chunks_and_embedding_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 80).encode("utf-8"),
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

            self.assertGreater(len(chunk_rows), 1)
            self.assertEqual(fts_count, len(chunk_rows))
            for chunk_id, chunk_index, content, embedding_model in chunk_rows:
                self.assertEqual(chunk_id, f"doc-1:{chunk_index:06d}")
                self.assertTrue(content)
                self.assertEqual(embedding_model, "test-embedding-model")

    # test_partial_embedding_failures_keep_job_success: partial failures should preserve success with warning payload.
    def test_partial_embedding_failures_keep_job_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 60).encode("utf-8"),
            )

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_path=str(temp_path / "doc.txt"),
                job_id="job-1",
                chunk_size_chars=90,
                chunk_overlap_chars=15,
                embedding_client=_PartialFailEmbeddingClient(),
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

            self.assertGreater(len(chunk_rows), 1)
            for chunk_index, embedding_model in chunk_rows:
                if chunk_index % 2 == 1:
                    self.assertIsNone(embedding_model)
                else:
                    self.assertEqual(embedding_model, "test-embedding-model")

    # test_all_embedding_failures_mark_job_fail_but_keep_chunks: all failures should fail job but keep persisted chunks.
    def test_all_embedding_failures_mark_job_fail_but_keep_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="doc.txt",
                source_bytes=("Alpha beta gamma " * 50).encode("utf-8"),
            )

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_path=str(temp_path / "doc.txt"),
                job_id="job-1",
                chunk_size_chars=90,
                chunk_overlap_chars=15,
                embedding_client=_AllFailEmbeddingClient(),
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

    # test_failure_marks_job_and_document_as_fail: extraction errors should still fail before embedding step.
    def test_failure_marks_job_and_document_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = self._prepare_document_and_job(
                temp_path=temp_path,
                source_name="broken.pdf",
                source_bytes=b"not-a-real-pdf",
            )

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id="doc-1",
                filename="broken.pdf",
                source_path=str(temp_path / "broken.pdf"),
                job_id="job-1",
                chunk_size_chars=1000,
                chunk_overlap_chars=150,
                embedding_client=_SuccessEmbeddingClient(),
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
