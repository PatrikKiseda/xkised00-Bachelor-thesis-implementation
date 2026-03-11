"""
Author: Patrik Kiseda
File: tests/test_jobs_api.py
Description: API tests for jobs visibility endpoint.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "documents")
os.environ.setdefault("LITELLM_MODEL", "openai/gpt-4o-mini")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")
os.environ.setdefault("EMBEDDING_API_ENABLED", "false")

from app.core.settings import Settings
from app.main import create_app
from app.storage.qdrant_store import QdrantConnectionStatus

# _HealthyStore: a simple mock Qdrant store that always reports healthy connectivity, used for testing the jobs API without relying on an actual Qdrant instance. This allows the tests to focus on the jobs endpoint behavior without external dependencies.
class _HealthyStore:
    def check_connection(self) -> QdrantConnectionStatus:
        return QdrantConnectionStatus(reachable=True)

# _build_settings: helper function to create Settings objects with specified sqlite_path and storage_dir, along with default values for other required settings. This simplifies test setup by allowing easy configuration of the app's settings for each test case.
def _build_settings(sqlite_path: str, storage_dir: str, **overrides: object) -> Settings:
    payload = {
        "app_name": "test-app",
        "qdrant_url": "http://test-qdrant:6333",
        "qdrant_collection": "documents",
        "sqlite_path": sqlite_path,
        "storage_dir": storage_dir,
        "chunk_size_chars": 120,
        "chunk_overlap_chars": 20,
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
        "embedding_api_enabled": False,
    }
    payload.update(overrides)
    return Settings(**payload)

# TestJobsApi: contains tests for the /api/jobs endpoint, verifying that it correctly returns indexing jobs with appropriate filtering and limit parameters. The tests ensure that when a document is uploaded, a corresponding indexing job is created and can be retrieved through the jobs API, and that the API supports filtering by document_id and limiting the number of returned jobs.
class TestJobsApi(unittest.TestCase):
    def test_jobs_endpoint_returns_upload_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                upload_response = client.post(
                    "/api/documents/upload",
                    files={"file": ("a.txt", b"alpha beta gamma " * 30, "text/plain")},
                )
                jobs_response = client.get("/api/jobs")

            self.assertEqual(upload_response.status_code, 201)
            self.assertEqual(jobs_response.status_code, 200)

            upload_payload = upload_response.json()
            jobs = jobs_response.json()["jobs"]
            self.assertGreaterEqual(len(jobs), 1)

            matching_jobs = [job for job in jobs if job["id"] == upload_payload["job_id"]]
            self.assertEqual(len(matching_jobs), 1)
            self.assertEqual(matching_jobs[0]["document_id"], upload_payload["id"])
            self.assertIn(matching_jobs[0]["status"], {"running", "success", "fail", "pending"})
            # payload_json should include embedding stats once indexing finishes.
            if matching_jobs[0]["status"] == "success":
                self.assertIsNotNone(matching_jobs[0]["payload_json"])
            
    # test_jobs_endpoint_supports_document_filter_and_limit: verifies that the /api/jobs endpoint correctly supports filtering by document_id and limiting the number of returned jobs. The test uploads multiple documents, retrieves jobs with a limit parameter to ensure only a certain number of jobs are returned, and retrieves jobs filtered by a specific document_id to ensure only relevant jobs are included in the response.
    def test_jobs_endpoint_supports_document_filter_and_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                upload_one = client.post(
                    "/api/documents/upload",
                    files={"file": ("one.txt", b"alpha beta gamma " * 20, "text/plain")},
                )
                upload_two = client.post(
                    "/api/documents/upload",
                    files={"file": ("two.txt", b"delta epsilon zeta " * 20, "text/plain")},
                )

                jobs_limited = client.get("/api/jobs?limit=1")
                document_id = upload_one.json()["id"]
                jobs_for_doc = client.get(f"/api/jobs?document_id={document_id}&limit=10")

            self.assertEqual(upload_one.status_code, 201)
            self.assertEqual(upload_two.status_code, 201)
            self.assertEqual(jobs_limited.status_code, 200)
            self.assertEqual(jobs_for_doc.status_code, 200)

            limited_payload = jobs_limited.json()
            self.assertEqual(len(limited_payload["jobs"]), 1)

            filtered_payload = jobs_for_doc.json()
            self.assertGreaterEqual(len(filtered_payload["jobs"]), 1)
            self.assertTrue(all(job["document_id"] == document_id for job in filtered_payload["jobs"]))
