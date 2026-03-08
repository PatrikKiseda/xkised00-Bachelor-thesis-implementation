"""
Author: Patrik Kiseda
File: src/app/ingestion/indexing_pipeline.py
Description: Background indexing pipeline (extract -> chunk -> SQLite persistence).
"""

from __future__ import annotations

from pathlib import Path

# Importing functions and classes from other modules in the app for use in the indexing pipeline.
from app.ingestion.chunker import chunk_text_recursive
from app.ingestion.extractors import extract_text
from app.storage.document_repository import update_document_status
# Importing functions and classes related to job management and chunk persistence from the indexing repository.
from app.storage.indexing_repository import (
    ChunkUpsert,
    mark_job_fail,
    mark_job_running,
    mark_job_success,
    replace_document_chunks,
)

# run_indexing_pipeline: main function that initializes the indexing process for a document.
def run_indexing_pipeline(
    *,
    db_path: str,
    document_id: str,
    filename: str,
    source_path: str,
    job_id: str,
    chunk_size_chars: int,
    chunk_overlap_chars: int,
) -> str:
    try:
        # Mark the job as running and update the document status to ```running``` at the start of the pipeline.
        mark_job_running(db_path, job_id=job_id)
        # Update the document status to ```running``` to reflect that the indexing process has started.
        update_document_status(db_path, document_id=document_id, status="running")

        source_bytes = Path(source_path).read_bytes()
        extraction = extract_text(filename, source_bytes)
        chunk_texts = chunk_text_recursive(
            extraction.text,
            chunk_size_chars=chunk_size_chars,
            chunk_overlap_chars=chunk_overlap_chars,
        )

        chunks = [
            ChunkUpsert(
                # The chunk ID is constructed using the document ID and the chunk index, ensuring it is unique.
                id=build_chunk_id(document_id, chunk_index),
                chunk_index=chunk_index,
                content=chunk_text,
            )
            for chunk_index, chunk_text in enumerate(chunk_texts)
        ]
        # The existing chunks for the document are replaced with the new ones generated from the extracted text.
        replace_document_chunks(db_path, document_id=document_id, chunks=chunks)

        mark_job_success(db_path, job_id=job_id)
        update_document_status(db_path, document_id=document_id, status="success")
        return "success"
    except Exception as exc:
        # In case of any exception during the indexing process, the failure is persisted by marking the job as failed and updating the document status to ```fail```.
        _persist_failure(
            db_path=db_path,
            document_id=document_id,
            job_id=job_id,
            error_message=str(exc),
        )
        return "fail"

# build_chunk_id: constructs a unique chunk ID using the document ID and the chunk index, formatted as `document_id:chunk_index`.
def build_chunk_id(document_id: str, chunk_index: int) -> str:
    return f"{document_id}:{chunk_index:06d}"

# _persist_failure: helper function to handle failure scenarios, marks the job as failed with an error message and updates the document status to ```fail```.
def _persist_failure(
    *,
    db_path: str,
    document_id: str,
    job_id: str,
    error_message: str,
) -> None:
    safe_error_message = error_message.strip() or "Indexing pipeline failed."

    try:
        mark_job_fail(db_path, job_id=job_id, error_message=safe_error_message)
    finally:
        update_document_status(db_path, document_id=document_id, status="fail")
