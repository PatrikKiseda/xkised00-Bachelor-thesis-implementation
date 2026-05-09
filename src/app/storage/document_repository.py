"""
Author: Patrik Kiseda
File: src/app/storage/document_repository.py
Description: SQLite operations for document metadata persistence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(slots=True)
class DocumentRecord:
    """Document metadata record from the `documents` table."""

    id: str
    filename: str
    source_type: str
    source_path: str
    size_bytes: int
    checksum: str
    status: str
    created_at: str
    updated_at: str

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
    """Add a new document record with provided metadata.

    Args:
        db_path: SQLite database path.
        document_id: New document id.
        filename: Original filename.
        source_type: Source type like txt, md, or pdf.
        source_path: Stored file path.
        size_bytes: File size in bytes.
        checksum: File checksum.
        status: Initial document status.

    Returns:
        Inserted document record.
    """
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

def list_documents(db_path: str) -> list[DocumentRecord]:
    """Retrieve all document records ordered by creation date.

    Args:
        db_path: SQLite database path.

    Returns:
        Document records.
    """
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

def update_document_status(
    db_path: str,
    *,
    document_id: str,
    status: str,
) -> None:
    """Update status of a document record by id.

    Args:
        db_path: SQLite database path.
        document_id: Document id to update.
        status: New status value.
    """
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
