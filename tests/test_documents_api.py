"""
Author: Patrik Kiseda
File: tests/test_documents_api.py
Description: API tests for document upload and SQLite metadata persistence.
"""

from __future__ import annotations

import os
import sqlite3
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

from app.core.settings import Settings
from app.main import create_app
from app.storage.qdrant_store import QdrantConnectionStatus

# _HealthyStore: fake store reporting healthy Qdrant connection. Used for positive health endpoint tests.
class _HealthyStore:
    def check_connection(self) -> QdrantConnectionStatus:
        return QdrantConnectionStatus(reachable=True)

# _build_settings: creates a validated baseline settings object for document API tests.
def _build_settings(sqlite_path: str, storage_dir: str, **overrides: object) -> Settings:
    payload = {
        "app_name": "test-app",
        "qdrant_url": "http://test-qdrant:6333",
        "qdrant_collection": "documents",
        "sqlite_path": sqlite_path,
        "storage_dir": storage_dir,
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
    }
    payload.update(overrides)
    return Settings(**payload)

# TestDocumentsApi: verifies document upload behavior, including file storage and SQLite metadata insertion.
class TestDocumentsApi(unittest.TestCase):
    # test_upload_stores_file_and_inserts_metadata: successful upload should store file and create correct SQLite record.
    def test_upload_stores_file_and_inserts_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
            )

            # perform upload via test client.
            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("notes.txt", b"first line\r\nsecond line  \r\n", "text/plain")},
                )

            payload = response.json()
            self.assertEqual(response.status_code, 201)
            self.assertEqual(payload["filename"], "notes.txt")
            self.assertEqual(payload["source_type"], "txt")
            self.assertEqual(payload["status"], "ready")

            stored_file_path = Path(payload["storage_path"])
            self.assertTrue(stored_file_path.exists())
            self.assertEqual(stored_file_path.read_bytes(), b"first line\r\nsecond line  \r\n")

            with sqlite3.connect(db_path) as connection:
                row = connection.execute(
                    """
                    SELECT filename, source_type, size_bytes, status
                    FROM documents
                    WHERE id = ?
                    """,
                    (payload["id"],),
                ).fetchone()

            self.assertIsNotNone(row)
            assert row is not None
            self.assertEqual(row[0], "notes.txt")
            self.assertEqual(row[1], "txt")
            self.assertEqual(row[2], len(b"first line\r\nsecond line  \r\n"))
            self.assertEqual(row[3], "ready")

    # test_upload_rejects_empty_file: empty uploads should fail with 400 and no metadata row.
    def test_upload_rejects_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("empty.txt", b"", "text/plain")},
                )

            self.assertEqual(response.status_code, 400)

            with sqlite3.connect(db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            self.assertEqual(count, 0)

    # test_upload_rejects_unsupported_extension: unsupported files should fail with 415.
    def test_upload_rejects_unsupported_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("image.png", b"png-bytes", "image/png")},
                )

            self.assertEqual(response.status_code, 415)

            with sqlite3.connect(db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            self.assertEqual(count, 0)

    # test_list_endpoint_returns_stored_documents: list endpoint should return uploaded metadata.
    def test_list_endpoint_returns_stored_documents(self) -> None:
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
                    files={"file": ("a.txt", b"alpha", "text/plain")},
                )
                upload_two = client.post(
                    "/api/documents/upload",
                    files={"file": ("b.md", b"# beta", "text/markdown")},
                )
                list_response = client.get("/api/documents")

            self.assertEqual(upload_one.status_code, 201)
            self.assertEqual(upload_two.status_code, 201)
            self.assertEqual(list_response.status_code, 200)

            payload = list_response.json()
            self.assertIn("documents", payload)
            self.assertEqual(len(payload["documents"]), 2)
            returned_filenames = {item["filename"] for item in payload["documents"]}
            self.assertEqual(returned_filenames, {"a.txt", "b.md"})

    # test_upload_malformed_pdf_returns_422: PDF extraction failures should be handled and surfaced.
    def test_upload_malformed_pdf_returns_422(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "app.db")
            storage_dir = str(Path(temp_dir) / "uploads")
            app = create_app(
                settings=_build_settings(sqlite_path=db_path, storage_dir=storage_dir),
                store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
            )

            with TestClient(app) as client:
                response = client.post(
                    "/api/documents/upload",
                    files={"file": ("broken.pdf", b"not-a-real-pdf", "application/pdf")},
                )

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["detail"], "Failed to extract text from PDF.")

            with sqlite3.connect(db_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            self.assertEqual(count, 0)
