"""
Author: Patrik Kiseda
File: src/app/storage/indexing_repository.py
Description: SQLite helpers for indexing jobs and chunk records.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

# JobRecord: representation of a job record from the ```jobs``` table.
@dataclass(slots=True)
class JobRecord:
    id: str
    job_type: str
    status: str
    document_id: str | None
    payload_json: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

# ChunkUpsert: represents a chunk record to be inserted into a document.
@dataclass(slots=True)
class ChunkUpsert:
    id: str
    chunk_index: int
    content: str
    token_count: int | None = None
    embedding_model: str | None = None


# ChunkLookupRecord: chunk row used by dense query endpoint to hydrate hit content.
@dataclass(slots=True)
class ChunkLookupRecord:
    id: str
    document_id: str
    chunk_index: int
    content: str

# list_jobs: retrieves a list of job records from the ```jobs``` table, optional filtering by document_id and limited in count.
def create_job(
    db_path: str,
    *,
    job_id: str,
    job_type: str,
    document_id: str | None,
    status: str = "pending",
    payload_json: str | None = None,
) -> JobRecord:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (id, job_type, status, document_id, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_id, job_type, status, document_id, payload_json),
        )
        row = connection.execute(
            """
            SELECT id, job_type, status, document_id, payload_json, error_message,
                   created_at, started_at, finished_at
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        connection.commit()

    if row is None:
        raise RuntimeError("Job insert succeeded but row was not found.")
    return _row_to_job_record(row)

# mark_job_running: updates a job record's status to ```running```  and sets the started_at timestamp if not set.
def mark_job_running(db_path: str, *, job_id: str) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'running',
                started_at = COALESCE(started_at, CURRENT_TIMESTAMP),
                error_message = NULL
            WHERE id = ?
            """,
            (job_id,),
        )
        connection.commit()

# mark_job_success: updates a job record's status to ```success``` and sets the finished_at timestamp.
def mark_job_success(
    db_path: str,
    *,
    job_id: str,
    payload_json: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'success',
                finished_at = CURRENT_TIMESTAMP,
                error_message = NULL,
                payload_json = ?
            WHERE id = ?
            """,
            (payload_json, job_id),
        )
        connection.commit()

# mark_job_fail: updates a job record's status to ```fail```, sets the finished_at timestamp, and records an error message.
def mark_job_fail(
    db_path: str,
    *,
    job_id: str,
    error_message: str,
    payload_json: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'fail',
                finished_at = CURRENT_TIMESTAMP,
                error_message = ?,
                payload_json = ?
            WHERE id = ?
            """,
            (error_message, payload_json, job_id),
        )
        connection.commit()

# list_jobs: retrieves a list of job records from the ```jobs``` table, optional filtering by document_id and limited in count.
def list_jobs(db_path: str, *, document_id: str | None = None, limit: int = 50) -> list[JobRecord]:
    effective_limit = max(1, min(limit, 200))

    with sqlite3.connect(db_path) as connection:
        if document_id:
            rows = connection.execute(
                """
                SELECT id, job_type, status, document_id, payload_json, error_message,
                       created_at, started_at, finished_at
                FROM jobs
                WHERE document_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (document_id, effective_limit),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, job_type, status, document_id, payload_json, error_message,
                       created_at, started_at, finished_at
                FROM jobs
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (effective_limit,),
            ).fetchall()

    return [_row_to_job_record(row) for row in rows]

# replace_document_chunks: deletes existing chunk records for a document, inserts instead new ones based on the provided list of ChunkUpsert objects.
def replace_document_chunks(
    db_path: str,
    *,
    document_id: str,
    chunks: list[ChunkUpsert],
) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON;")
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))

        for chunk in chunks:
            connection.execute(
                """
                INSERT INTO chunks (id, document_id, chunk_index, content, token_count, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.token_count,
                    chunk.embedding_model,
                ),
            )
            connection.execute(
                """
                INSERT INTO chunks_fts (chunk_id, document_id, content)
                VALUES (?, ?, ?)
                """,
                (chunk.id, document_id, chunk.content),
            )
        connection.commit()

# get_chunks_by_ids: retrieves chunk records from the ```chunks``` table based on a list of chunk IDs. Returns a mapping of chunk ID to ChunkLookupRecord.
def get_chunks_by_ids(
    db_path: str,
    *,
    chunk_ids: list[str],
) -> dict[str, ChunkLookupRecord]:
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)
    query = f"""
        SELECT id, document_id, chunk_index, content
        FROM chunks
        WHERE id IN ({placeholders})
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, chunk_ids).fetchall()

    return {
        str(row[0]): ChunkLookupRecord(
            id=str(row[0]),
            document_id=str(row[1]),
            chunk_index=int(row[2]),
            content=str(row[3]),
        )
        for row in rows
    }

# separate lookups and upserts for better clarity and separation of functions.

# _row_to_chunk_lookup_record: helper to convert a SQL row tuple into a ChunkLookupRecord dataclass instance.
def _row_to_chunk_lookup_record(row:  tuple[object, ...]) -> ChunkLookupRecord:
    return ChunkLookupRecord(
        id=str(row[0]),
        document_id=str(row[1]),
        chunk_index=int(row[2]),
        content=str(row[3]),
    ) 

# _row_to_job_record: helper to convert a SQL row tuple into a JobRecord dataclass instance.
def _row_to_job_record(row: tuple[object, ...]) -> JobRecord:
    return JobRecord(
        id=str(row[0]),
        job_type=str(row[1]),
        status=str(row[2]),
        document_id=str(row[3]) if row[3] is not None else None,
        payload_json=str(row[4]) if row[4] is not None else None,
        error_message=str(row[5]) if row[5] is not None else None,
        created_at=str(row[6]),
        started_at=str(row[7]) if row[7] is not None else None,
        finished_at=str(row[8]) if row[8] is not None else None,
    )
