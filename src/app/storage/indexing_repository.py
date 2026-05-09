"""
Author: Patrik Kiseda
File: src/app/storage/indexing_repository.py
Description: SQLite helpers for indexing jobs and chunk records.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

@dataclass(slots=True)
class JobRecord:
    """Job record from the `jobs` table."""

    id: str
    job_type: str
    status: str
    document_id: str | None
    payload_json: str | None
    error_message: str | None
    created_at: str
    started_at: str | None
    finished_at: str | None

@dataclass(slots=True)
class ChunkUpsert:
    """Chunk record to insert for a document."""

    id: str
    chunk_index: int
    content: str
    token_count: int | None = None
    embedding_model: str | None = None


@dataclass(slots=True)
class ChunkLookupRecord:
    """Chunk row used by dense query endpoint to hydrate hit content."""

    id: str
    document_id: str
    filename: str | None
    chunk_index: int
    content: str

@dataclass(slots=True)
class LexicalSearchRow:
    """Row returned from lexical search, including raw BM25 score."""

    chunk_id: str
    document_id: str
    filename: str | None
    chunk_index: int
    content: str
    raw_score: float

def create_job(
    db_path: str,
    *,
    job_id: str,
    job_type: str,
    document_id: str | None,
    status: str = "pending",
    payload_json: str | None = None,
) -> JobRecord:
    """Create a job record in the `jobs` table.

    Args:
        db_path: SQLite database path.
        job_id: New job id.
        job_type: Job type name.
        document_id: Related document id, if any.
        status: Initial job status.
        payload_json: Optional job payload.

    Returns:
        Inserted job record.
    """
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

def mark_job_running(db_path: str, *, job_id: str) -> None:
    """Mark a job as running and set started_at if missing.

    Args:
        db_path: SQLite database path.
        job_id: Job id to update.
    """
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

def mark_job_success(
    db_path: str,
    *,
    job_id: str,
    payload_json: str | None = None,
) -> None:
    """Mark a job as successful and store optional payload.

    Args:
        db_path: SQLite database path.
        job_id: Job id to update.
        payload_json: Optional payload to store.
    """
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

def mark_job_fail(
    db_path: str,
    *,
    job_id: str,
    error_message: str,
    payload_json: str | None = None,
) -> None:
    """Mark a job as failed and record an error message.

    Args:
        db_path: SQLite database path.
        job_id: Job id to update.
        error_message: Error message to store.
        payload_json: Optional payload to store.
    """
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

def list_jobs(db_path: str, *, document_id: str | None = None, limit: int = 50) -> list[JobRecord]:
    """Retrieve job records, optionally filtered by document id.

    Args:
        db_path: SQLite database path.
        document_id: Optional document filter.
        limit: Max rows to return.

    Returns:
        Job records.
    """
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

def replace_document_chunks(
    db_path: str,
    *,
    document_id: str,
    chunks: list[ChunkUpsert],
) -> None:
    """Replace all chunk records for a document.

    Args:
        db_path: SQLite database path.
        document_id: Document id whose chunks are replaced.
        chunks: New chunk records to insert.
    """
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

def get_chunks_by_ids(
    db_path: str,
    *,
    chunk_ids: list[str],
) -> dict[str, ChunkLookupRecord]:
    """Retrieve chunk records by ids.

    Args:
        db_path: SQLite database path.
        chunk_ids: Chunk ids to fetch.

    Returns:
        Mapping of chunk id to lookup record.
    """
    if not chunk_ids:
        return {}

    placeholders = ",".join("?" for _ in chunk_ids)
    query = f"""
        SELECT chunks.id, chunks.document_id, chunks.chunk_index, chunks.content, documents.filename
        FROM chunks
        LEFT JOIN documents ON documents.id = chunks.document_id
        WHERE chunks.id IN ({placeholders})
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, chunk_ids).fetchall()

    return {
        str(row[0]): ChunkLookupRecord(
            id=str(row[0]),
            document_id=str(row[1]),
            filename=str(row[4]) if row[4] is not None else None,
            chunk_index=int(row[2]),
            content=str(row[3]),
        )
        for row in rows
    }

def search_chunks_lexical(
    db_path: str,
    *,
    query_text: str,
    limit: int,
) -> list[LexicalSearchRow]:
    """Run lexical search against the `chunks_fts` virtual table.

    Args:
        db_path: SQLite database path.
        query_text: Raw user query.
        limit: Max rows to return.

    Returns:
        Lexical search rows with raw BM25 scores.
    """
    terms = _extract_fts5_terms(query_text)
    if not terms:
        return []

    strict_rows = _run_lexical_query(
        db_path=db_path,
        match_query=_join_fts5_terms(terms, operator="AND"),
        limit=limit,
    )
    if strict_rows or len(terms) == 1:
        return [_row_to_lexical_search_row(row) for row in strict_rows]

    fallback_rows = _run_lexical_query(
        db_path=db_path,
        match_query=_join_fts5_terms(terms, operator="OR"),
        limit=limit,
    )
    return [_row_to_lexical_search_row(row) for row in fallback_rows]


def normalize_fts5_query(query_text: str) -> str | None:
    """Normalize raw query text into an FTS5 AND query.

    Args:
        query_text: Raw query text.

    Returns:
        FTS5 query string, or None when there are no terms.
    """
    terms = _extract_fts5_terms(query_text)
    if not terms:
        return None
    return _join_fts5_terms(terms, operator="AND")

def _run_lexical_query(
    *,
    db_path: str,
    match_query: str,
    limit: int,
) -> list[tuple[object, ...]]:
    """Execute raw FTS5 query and return rows for mapping.

    Args:
        db_path: SQLite database path.
        match_query: FTS5 MATCH query.
        limit: Max rows to return.

    Returns:
        Raw SQL rows.
    """
    with sqlite3.connect(db_path) as connection:
        return connection.execute(
            """
            SELECT chunks.id, chunks.document_id, documents.filename, chunks.chunk_index,
                   chunks.content, bm25(chunks_fts) AS raw_score
            FROM chunks_fts
            JOIN chunks ON chunks.id = chunks_fts.chunk_id
            LEFT JOIN documents ON documents.id = chunks.document_id
            WHERE chunks_fts MATCH ?
            ORDER BY raw_score ASC, chunks.id ASC
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()

def _extract_fts5_terms(query_text: str) -> list[str]:
    """Extract searchable terms from raw query text.

    Args:
        query_text: Raw query text.

    Returns:
        Search terms.
    """
    return re.findall(r"\w+", query_text, flags=re.UNICODE)

def _join_fts5_terms(terms: list[str], *, operator: str) -> str:
    """Join terms into a valid FTS5 MATCH query string.

    Args:
        terms: Search terms.
        operator: FTS5 operator like AND or OR.

    Returns:
        Joined FTS5 query.
    """
    return f" {operator} ".join(f'"{term}"' for term in terms)

# separate lookups and upserts for better clarity and separation of functions.

def _row_to_chunk_lookup_record(row:  tuple[object, ...]) -> ChunkLookupRecord:
    """Convert SQL row into a ChunkLookupRecord.

    Args:
        row: SQL row tuple.

    Returns:
        Chunk lookup record.
    """
    return ChunkLookupRecord(
        id=str(row[0]),
        document_id=str(row[1]),
        filename=str(row[4]) if row[4] is not None else None,
        chunk_index=int(row[2]),
        content=str(row[3]),
    ) 

def _row_to_lexical_search_row(row: tuple[object, ...]) -> LexicalSearchRow:
    """Convert SQL row into a LexicalSearchRow.

    Args:
        row: SQL row tuple.

    Returns:
        Lexical search row.
    """
    return LexicalSearchRow(
        chunk_id=str(row[0]),
        document_id=str(row[1]),
        filename=str(row[2]) if row[2] is not None else None,
        chunk_index=int(row[3]),
        content=str(row[4]),
        raw_score=float(row[5]),
    )

def _row_to_job_record(row: tuple[object, ...]) -> JobRecord:
    """Convert SQL row into a JobRecord.

    Args:
        row: SQL row tuple.

    Returns:
        Job record.
    """
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
