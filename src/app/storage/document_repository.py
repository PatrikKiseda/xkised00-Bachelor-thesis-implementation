"""
Author: Patrik Kiseda
File: src/app/storage/document_repository.py
Description: SQLite operations for document metadata persistence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


# DocumentRecord: represents a document metadata record from the `documents` table,
# includes additional fields for source type, filename, and size.
@dataclass(slots=True)
class DocumentRecord:
    id: str
    filename: str
    source_type: str
    source_path: str
    size_bytes: int
    checksum: str
    status: str
    created_at: str
    updated_at: str

# insert_document: adds a new document record to the `documents` table with the provided metadata.
def insert_document(
    db_path: str,
    *,
    document_id: str,
    filename: str,
    source_type: str,
    source_path: str,
    size_bytes: int,
    checksum: str,
    status: str = "ready",
) -> DocumentRecord:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO documents (
                id,
                source_path,
                source_type,
                filename,
                size_bytes,
                checksum,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                document_id,
                source_path,
                source_type,
                filename,
                size_bytes,
                checksum,
                status,
            ),
        )
        row = connection.execute(
            """
            SELECT id, filename, source_type, source_path, size_bytes, checksum, status, created_at, updated_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()
        connection.commit()

    # sanity check: the inserted row should be retrievable.
    if row is None:
        raise RuntimeError("Document insert succeeded but document row was not found.")

    # return a structured DocumentRecord for the inserted row.
    return DocumentRecord(
        id=row[0],
        filename=row[1],
        source_type=row[2],
        source_path=row[3],
        size_bytes=row[4],
        checksum=row[5],
        status=row[6],
        created_at=row[7],
        updated_at=row[8],
    )

# list_documents: retrieves all document records from the `documents` table, ordered by date of creation.
def list_documents(db_path: str) -> list[DocumentRecord]:
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT id, filename, source_type, source_path, size_bytes, checksum, status, created_at, updated_at
            FROM documents
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()
    
    # return a list of DocumentRecord objects for all retrieved rows.
    return [
        DocumentRecord(
            id=row[0],
            filename=row[1],
            source_type=row[2],
            source_path=row[3],
            size_bytes=row[4],
            checksum=row[5],
            status=row[6],
            created_at=row[7],
            updated_at=row[8],
        )
        for row in rows
    ]

# update_document_status: updates the status of a document record in the ```documents``` table, id by document_id.
def update_document_status(
    db_path: str,
    *,
    document_id: str,
    status: str,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE documents
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, document_id),
        )
        connection.commit()
