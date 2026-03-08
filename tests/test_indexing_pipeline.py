"""
Author: Patrik Kiseda
File: tests/test_indexing_pipeline.py
Description: Unit tests for background indexing pipeline and job state persistence.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.ingestion.indexing_pipeline import run_indexing_pipeline
from app.storage.document_repository import insert_document
from app.storage.indexing_repository import create_job
from app.storage.sqlite_schema import initialize_sqlite_schema

# TestIndexingPipeline: verifies the behavior of the indexing pipeline, including successful chunk persistence and proper job/document status updates on both success and failure scenarios.
class TestIndexingPipeline(unittest.TestCase):
    def test_success_persists_chunks_and_job_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = str(initialize_sqlite_schema(str(temp_path / "app.db")))
            source_path = temp_path / "doc.txt"
            source_path.write_text(
                ("Alpha beta gamma delta epsilon zeta eta theta iota. " * 40),
                encoding="utf-8",
            )

            document = insert_document(
                db_path,
                document_id="doc-1",
                filename="doc.txt",
                source_type="txt",
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

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id=document.id,
                filename="doc.txt",
                source_path=str(source_path),
                job_id="job-1",
                chunk_size_chars=100,
                chunk_overlap_chars=15,
            )

            self.assertEqual(outcome, "success")

            # After the pipeline runs, the test verifies that the job status is marked as ```success```, the document status is updated to ```success```, and that multiple chunks were created with the expected content. It also checks that the full-text search index was populated with the correct number of entries corresponding to the chunks.
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
                chunk_rows = connection.execute(
                    """
                    SELECT id, chunk_index, content
                    FROM chunks
                    WHERE document_id = 'doc-1'
                    ORDER BY chunk_index ASC
                    """
                ).fetchall()
                fts_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks_fts WHERE document_id = 'doc-1'"
                ).fetchone()[0]

            self.assertIsNotNone(job_row)
            self.assertIsNotNone(document_row)
            assert job_row is not None
            assert document_row is not None
            self.assertEqual(job_row[0], "success")
            self.assertIsNotNone(job_row[1])
            self.assertIsNotNone(job_row[2])
            self.assertIsNone(job_row[3])
            self.assertEqual(document_row[0], "success")

            self.assertGreater(len(chunk_rows), 1)
            for chunk_id, chunk_index, content in chunk_rows:
                self.assertEqual(chunk_id, f"doc-1:{chunk_index:06d}")
                self.assertTrue(content)
            self.assertEqual(fts_count, len(chunk_rows))

    #   The second test verifies that when the indexing pipeline encounters a failure (in this case, due to an invalid PDF file), the job is marked as ```fail``` with an appropriate error message, the document status is updated to ```fail```, and no chunks are created for the document. This ensures that failure scenarios are properly handled and persisted in the database.
    def test_failure_marks_job_and_document_as_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            db_path = str(initialize_sqlite_schema(str(temp_path / "app.db")))
            source_path = temp_path / "broken.pdf"
            source_path.write_bytes(b"not-a-real-pdf")

            document = insert_document(
                db_path,
                document_id="doc-fail",
                filename="broken.pdf",
                source_type="pdf",
                source_path=str(source_path),
                size_bytes=source_path.stat().st_size,
                checksum="checksum",
                status="pending",
            )
            create_job(
                db_path,
                job_id="job-fail",
                job_type="indexing",
                document_id=document.id,
                status="pending",
            )

            outcome = run_indexing_pipeline(
                db_path=db_path,
                document_id=document.id,
                filename="broken.pdf",
                source_path=str(source_path),
                job_id="job-fail",
                chunk_size_chars=1000,
                chunk_overlap_chars=150,
            )

            self.assertEqual(outcome, "fail")

            # After the pipeline runs, the test verifies that the job status is marked as ```fail``` with an appropriate error message, the document status is updated to ```fail```, and that no chunks were created for the document. This ensures that failure scenarios are properly handled and persisted in the database.
            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    """
                    SELECT status, started_at, finished_at, error_message
                    FROM jobs
                    WHERE id = 'job-fail'
                    """
                ).fetchone()
                document_row = connection.execute(
                    "SELECT status FROM documents WHERE id = 'doc-fail'"
                ).fetchone()
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = 'doc-fail'"
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
