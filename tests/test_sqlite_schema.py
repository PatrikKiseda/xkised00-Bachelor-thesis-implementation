"""
Author: Patrik Kiseda
File: tests/test_sqlite_schema.py
Description: Unit tests for SQLite schema initialization compulsory table check.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# sys.path: allows test imports from src 
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.storage.sqlite_schema import initialize_sqlite_schema


# TestSqliteSchema: verifies required tables and FTS5 virtual table are created.
class TestSqliteSchema(unittest.TestCase):
    # test_initialize_creates_documents_chunks_jobs_and_fts: checks required schema objects.
    def test_initialize_creates_documents_chunks_jobs_and_fts(self) -> None:
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

            self.assertIn("documents", tables)
            self.assertIn("chunks", tables)
            self.assertIn("jobs", tables)
            self.assertIn("chunks_fts", tables)
            self.assertIsNotNone(fts_definition)
            self.assertIn("fts5", (fts_definition[0] or "").lower())
