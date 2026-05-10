"""
Author: Patrik Kiseda
File: tests/test_sqlite_schema.py
Description: Unit tests for SQLite schema initialization and migration.

The test database is always a temporary SQLite file. The tests verify the base schema,
FTS5 table, and compatibility path for older documents tables that need new
metadata columns added.
"""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import helpers
from app.storage.sqlite_schema import initialize_sqlite_schema


class TestSqliteSchema(unittest.TestCase):
    """Schema initialization and lightweight migration tests."""

    def test_initialize_creates_documents_chunks_jobs_and_fts(self) -> None:
        """Schema initialization should create metadata tables and FTS5 index."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "metadata" / "app.db"

            created_path = initialize_sqlite_schema(str(db_path))

            self.assertEqual(created_path, db_path)
            self.assertTrue(db_path.exists())

            with sqlite3.connect(db_path) as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table')"
                    ).fetchall()
                }
                fts_definition = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE name='chunks_fts'"
                ).fetchone()
                document_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(documents)").fetchall()
                }

            self.assertIn("documents", tables)
            self.assertIn("chunks", tables)
            self.assertIn("jobs", tables)
            self.assertIn("chunks_fts", tables)
            self.assertIn("source_type", document_columns)
            self.assertIn("filename", document_columns)
            self.assertIn("size_bytes", document_columns)
            self.assertIsNotNone(fts_definition)
            self.assertIn("fts5", (fts_definition[0] or "").lower())

    def test_initialize_migrates_existing_documents_table_columns(self) -> None:
        """Existing documents table should gain new metadata columns."""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "metadata" / "legacy.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(db_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE documents (
                        id TEXT PRIMARY KEY,
                        source_path TEXT NOT NULL,
                        title TEXT,
                        checksum TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                connection.commit()

            initialize_sqlite_schema(str(db_path))

            with sqlite3.connect(db_path) as connection:
                document_columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(documents)").fetchall()
                }

            self.assertIn("source_type", document_columns)
            self.assertIn("filename", document_columns)
            self.assertIn("size_bytes", document_columns)
