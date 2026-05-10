"""
API workflow tests for the FastAPI app.

These tests cover health, document upload/listing, job visibility, retrieval
query modes, answer generation, prompt debugging, and the local HTML UI. Qdrant
and generation are replaced by small fakes from helpers, so the tests stay fast
and do not need external services.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from helpers import (
    HealthyStore,
    InMemoryDenseStore,
    RecordingGenerationClient,
    UnhealthyStore,
    build_settings,
    dense_hits_for_two_ranked_chunks,
    seed_document_chunks,
    suppress_expected_pdf_noise,
    two_ranked_chunks,
)
from app.main import create_app
from app.storage.indexing_repository import ChunkUpsert


class TestApiWorkflows(unittest.TestCase):
    """End-to-end API behavior through FastAPI TestClient and local fakes."""

    def _create_test_app(
        self,
        temp_dir: str,
        *,
        store: object | None = None,
        generation_client: RecordingGenerationClient | None = None,
    ):
        """Build a test app with temp storage and optional fake dependencies."""
        temp_path = Path(temp_dir)
        return create_app(
            settings=build_settings(
                sqlite_path=str(temp_path / "app.db"),
                storage_dir=str(temp_path / "uploads"),
            ),
            store_factory=lambda _: store or InMemoryDenseStore(),  # type: ignore[arg-type]
            generation_client_factory=(
                (lambda _: generation_client) if generation_client is not None else None
            ),
        )

    def test_health_reports_reachable_and_unreachable_qdrant(self) -> None:
        """Health endpoint should show ok/degraded based on Qdrant store status."""
        ok_app = create_app(
            settings=build_settings(sqlite_path=":memory:"),
            store_factory=lambda _: HealthyStore(),  # type: ignore[arg-type]
        )
        degraded_app = create_app(
            settings=build_settings(sqlite_path=":memory:"),
            store_factory=lambda _: UnhealthyStore(),  # type: ignore[arg-type]
        )

        with TestClient(ok_app) as client:
            ok_payload = client.get("/api/health").json()
        with TestClient(degraded_app) as client:
            degraded_payload = client.get("/api/health").json()

        self.assertEqual(ok_payload["status"], "ok")
        self.assertTrue(ok_payload["qdrant"]["reachable"])
        self.assertTrue(ok_payload["qdrant"]["reachable_on_startup"])
        self.assertEqual(degraded_payload["status"], "degraded")
        self.assertFalse(degraded_payload["qdrant"]["reachable"])
        self.assertEqual(degraded_payload["qdrant"]["startup_error"], "qdrant unreachable")

    def test_upload_stores_file_metadata_and_finished_indexing_state(self) -> None:
        """Upload should store file bytes, metadata, job payload, and chunks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", b"first line\r\nsecond line  \r\n", "text/plain")},
                )

            payload = response.json()
            self.assertEqual(response.status_code, 201)
            self.assertEqual(payload["filename"], "notes.txt")
            self.assertEqual(payload["source_type"], "txt")
            self.assertTrue(Path(payload["storage_path"]).exists())

            with sqlite3.connect(db_path) as connection:
                document_row = connection.execute(
                    "SELECT filename, source_type, size_bytes, status FROM documents WHERE id = ?",
                    (payload["id"],),
                ).fetchone()
                job_row = connection.execute(
                    "SELECT id, status, payload_json FROM jobs WHERE id = ?",
                    (payload["job_id"],),
                ).fetchone()
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                    (payload["id"],),
                ).fetchone()[0]

            self.assertEqual(document_row, ("notes.txt", "txt", len(b"first line\r\nsecond line  \r\n"), "success"))
            self.assertIsNotNone(job_row)
            assert job_row is not None
            self.assertEqual(job_row[0], payload["job_id"])
            self.assertEqual(job_row[1], "success")
            self.assertIsNotNone(job_row[2])
            self.assertGreater(chunk_count, 0)

    def test_upload_rejects_invalid_files_without_metadata_rows(self) -> None:
        """Empty or unsupported uploads should fail before creating documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                empty_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("empty.txt", b"", "text/plain")},
                )
                unsupported_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("image.png", b"png-bytes", "image/png")},
                )

            self.assertEqual(empty_response.status_code, 400)
            self.assertEqual(unsupported_response.status_code, 415)
            with sqlite3.connect(db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            self.assertEqual(count, 0)

    def test_list_endpoint_and_jobs_endpoint_return_upload_records(self) -> None:
        """Document and job list endpoints should expose uploaded records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                upload_one = client.post(
                    "/api/documents/upload",
                    files={"file": ("a.txt", b"alpha beta gamma " * 20, "text/plain")},
                )
                upload_two = client.post(
                    "/api/documents/upload",
                    files={"file": ("b.md", b"# beta " * 20, "text/markdown")},
                )
                documents_response = client.get("/api/documents")
                jobs_limited = client.get("/api/jobs?limit=1")
                jobs_for_doc = client.get(f"/api/jobs?document_id={upload_one.json()['id']}&limit=10")

            self.assertEqual(upload_one.status_code, 201)
            self.assertEqual(upload_two.status_code, 201)
            self.assertEqual(documents_response.status_code, 200)
            self.assertEqual(jobs_limited.status_code, 200)
            self.assertEqual(jobs_for_doc.status_code, 200)

            documents = documents_response.json()["documents"]
            self.assertEqual({item["filename"] for item in documents}, {"a.txt", "b.md"})
            self.assertEqual(len(jobs_limited.json()["jobs"]), 1)
            filtered_jobs = jobs_for_doc.json()["jobs"]
            self.assertGreaterEqual(len(filtered_jobs), 1)
            self.assertTrue(all(job["document_id"] == upload_one.json()["id"] for job in filtered_jobs))

    def test_malformed_pdf_upload_creates_failed_job_without_parser_noise(self) -> None:
        """Malformed PDF uploads should become failed indexing jobs cleanly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                with suppress_expected_pdf_noise():
                    response = client.post(
                        "/api/documents/upload",
                        files={"file": ("broken.pdf", b"not-a-real-pdf", "application/pdf")},
                    )

            self.assertEqual(response.status_code, 201)
            payload = response.json()
            with sqlite3.connect(db_path) as connection:
                job_row = connection.execute(
                    "SELECT status, error_message, started_at, finished_at FROM jobs WHERE id = ?",
                    (payload["job_id"],),
                ).fetchone()
                document_status = connection.execute(
                    "SELECT status FROM documents WHERE id = ?",
                    (payload["id"],),
                ).fetchone()[0]
                chunk_count = connection.execute(
                    "SELECT COUNT(*) FROM chunks WHERE document_id = ?",
                    (payload["id"],),
                ).fetchone()[0]

            self.assertIsNotNone(job_row)
            assert job_row is not None
            self.assertEqual(job_row[0], "fail")
            self.assertIn("Failed to extract text from PDF.", job_row[1] or "")
            self.assertIsNotNone(job_row[2])
            self.assertIsNotNone(job_row[3])
            self.assertEqual(document_status, "fail")
            self.assertEqual(chunk_count, 0)

    def test_dense_and_lexical_query_endpoints_return_hydrated_hits(self) -> None:
        """Dense and lexical query endpoints should return hydrated chunk hits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 40), "text/plain")},
                )
                dense_response = client.post("/api/query/dense", json={"query": "gamma delta", "top_k": 3})
                lexical_response = client.post("/api/query/lexical", json={"query": "alpha beta", "top_k": 3})

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(dense_response.status_code, 200)
            self.assertEqual(lexical_response.status_code, 200)
            dense_payload = dense_response.json()
            lexical_payload = lexical_response.json()
            self.assertEqual(dense_payload["mode"], "dense")
            self.assertEqual(lexical_payload["mode"], "lexical")
            self.assertGreater(len(dense_payload["hits"]), 0)
            self.assertGreater(len(lexical_payload["hits"]), 0)
            for hit in (dense_payload["hits"][0], lexical_payload["hits"][0]):
                self.assertTrue(hit["chunk_id"])
                self.assertTrue(hit["document_id"])
                self.assertIn("content", hit)
                self.assertGreaterEqual(hit["score"], 0.0)

    def test_query_endpoints_handle_empty_index_and_invalid_payload(self) -> None:
        """Query endpoints should handle empty indexes and validation errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                dense_empty = client.post("/api/query/dense", json={"query": "anything"})
                lexical_empty = client.post("/api/query/lexical", json={"query": "anything"})
                empty_query = client.post("/api/query/dense", json={"query": "", "top_k": 5})
                invalid_top_k = client.post("/api/query/dense", json={"query": "valid", "top_k": 0})

            self.assertEqual(dense_empty.status_code, 200)
            self.assertEqual(dense_empty.json()["hits"], [])
            self.assertEqual(lexical_empty.status_code, 200)
            self.assertEqual(lexical_empty.json()["hits"], [])
            self.assertEqual(empty_query.status_code, 422)
            self.assertEqual(invalid_top_k.status_code, 422)

    def test_hybrid_query_endpoint_returns_fused_ranked_hits(self) -> None:
        """Hybrid query endpoint should return RRF-fused ranked hits."""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = InMemoryDenseStore()
            app = self._create_test_app(temp_dir, store=store)

            with TestClient(app) as client:
                seed_document_chunks(app.state.sqlite_db_path, chunks=two_ranked_chunks())
                store.dense_hits_override = dense_hits_for_two_ranked_chunks()
                response = client.post("/api/query/hybrid", json={"query": "alpha beta", "top_k": 2})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["mode"], "hybrid")
            self.assertEqual([hit["chunk_id"] for hit in payload["hits"]], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual(payload["hits"][0]["content"], "alpha alpha beta")

    def test_answer_endpoint_returns_answer_sources_and_no_context_path(self) -> None:
        """Answer endpoint should return sources and handle no-hit prompting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = RecordingGenerationClient("Grounded answer [S1]")
            app = self._create_test_app(temp_dir, generation_client=generation_client)

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                answer_response = client.post(
                    "/api/query/answer",
                    json={"query": "What does the note mention?", "top_k": 3, "mode": "dense"},
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(answer_response.status_code, 200)
            payload = answer_response.json()
            self.assertEqual(payload["answer"], "Grounded answer [S1]")
            self.assertGreater(len(payload["sources"]), 0)
            self.assertEqual(payload["sources"][0]["filename"], "notes.txt")
            self.assertEqual(len(generation_client.calls), 1)

        with tempfile.TemporaryDirectory() as temp_dir:
            no_hit_client = RecordingGenerationClient("Answer with no retrieved context.")
            app = self._create_test_app(temp_dir, generation_client=no_hit_client)
            with TestClient(app) as client:
                response = client.post(
                    "/api/query/answer",
                    json={"query": "What does the note mention?", "mode": "dense"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["sources"], [])
            self.assertIn("No retrieved context was available", no_hit_client.calls[0][0])

    def test_answer_and_prompt_debug_support_context_disabled_lexical_and_hybrid_modes(self) -> None:
        """Answer/prompt-debug should cover raw, lexical, and hybrid paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generation_client = RecordingGenerationClient("Answer without retrieval in prompt.")
            store = InMemoryDenseStore()
            app = self._create_test_app(temp_dir, store=store, generation_client=generation_client)

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", (b"alpha beta gamma delta " * 30), "text/plain")},
                )
                raw_answer = client.post(
                    "/api/query/answer",
                    json={
                        "query": "What does the note mention?",
                        "top_k": 3,
                        "mode": "dense",
                        "include_context_in_prompt": False,
                    },
                )
                lexical_prompt = client.post(
                    "/api/query/prompt-debug",
                    json={"query": "What does alpha mean?", "top_k": 3, "mode": "lexical"},
                )
                seed_document_chunks(
                    app.state.sqlite_db_path,
                    chunks=[
                        ChunkUpsert(id="doc-1:000000", chunk_index=0, content="unique unique fusion"),
                        ChunkUpsert(id="doc-1:000001", chunk_index=1, content="unique fusion gamma"),
                    ],
                )
                store.dense_hits_override = dense_hits_for_two_ranked_chunks()
                hybrid_answer = client.post(
                    "/api/query/answer",
                    json={"query": "unique fusion", "top_k": 2, "mode": "hybrid"},
                )
                hybrid_prompt = client.post(
                    "/api/query/prompt-debug",
                    json={"query": "unique fusion", "top_k": 2, "mode": "hybrid"},
                )

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(raw_answer.status_code, 200)
            self.assertEqual(generation_client.calls[0][0], "What does the note mention?")
            self.assertEqual(lexical_prompt.status_code, 200)
            self.assertIn("Retrieved context:", lexical_prompt.json()["prompt"])
            self.assertEqual(hybrid_answer.status_code, 200)
            self.assertEqual(hybrid_answer.json()["mode"], "hybrid")
            self.assertEqual([source["chunk_id"] for source in hybrid_answer.json()["sources"]], ["doc-1:000000", "doc-1:000001"])
            self.assertEqual(hybrid_prompt.status_code, 200)
            self.assertIn("[S1]", hybrid_prompt.json()["prompt"])

    def test_root_serves_localhost_ui(self) -> None:
        """Root endpoint should serve the local HTML client."""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = self._create_test_app(temp_dir)

            with TestClient(app) as client:
                response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn("RAG client", response.text)
            self.assertIn("/api/query/answer", response.text)
            self.assertIn("value=\"lexical\"", response.text)
            self.assertIn("value=\"hybrid\"", response.text)
