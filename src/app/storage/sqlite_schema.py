"""
Author: Patrik Kiseda
File: src/app/storage/sqlite_schema.py
Description: SQLite schema initialization for local metadata and lexical search placeholders.
"""

from __future__ import annotations
import sqlite3
from pathlib import Path


# initialize_sqlite_schema: creates required metadata tables and FTS5 index placeholders. 
# this is useful for adding information linking of the retrieved document/chunk to the 
# location of the original document, and for enabling local lexical search capabilities in the future.

# DOCUMENTS_TABLE_ADDITIONAL_COLUMNS: extra metadata fields for documents.
DOCUMENTS_TABLE_ADDITIONAL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("source_type", "TEXT"),
    ("filename", "TEXT"),
    ("size_bytes", "INTEGER"),
)

def initialize_sqlite_schema(db_path: str) -> Path:
    resolved_db_path = Path(db_path).expanduser()
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                title TEXT,
                checksum TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER,
                embedding_model TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, chunk_index)
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                document_id TEXT,
                payload_json TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                finished_at TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                document_id UNINDEXED,
                content
            );
            """
        )
        connection.commit()

    return resolved_db_path
