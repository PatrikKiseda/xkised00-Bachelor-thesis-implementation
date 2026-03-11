"""
Author: Patrik Kiseda
File: src/app/ingestion/indexing_pipeline.py
Description: Background indexing pipeline (extract -> chunk -> embed -> SQLite persistence).
"""

from __future__ import annotations

import json
from pathlib import Path

# Importing functions and classes from other modules in the app for use in the indexing pipeline.
from app.embeddings.adapter import EmbeddingClient, EmbeddingItemResult
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
    embedding_client: EmbeddingClient,
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
        # Embedding all produced chunks in one batch and keep per-chunk outcomes.
        embedding_result = embedding_client.embed_texts(chunk_texts)
        items_by_index = {item.index: item for item in embedding_result.items}
        failed_embedding_items: list[EmbeddingItemResult] = []

        # Persist chunks even if some embeddings fail; model marker is only set for successes.
        chunks = [
            ChunkUpsert(
                # The chunk ID is constructed using the document ID and the chunk index, ensuring it is unique.
                id=build_chunk_id(document_id, chunk_index),
                chunk_index=chunk_index,
                content=chunk_text,
                # The embedding model is set based on the success of the embedding operation for each chunk. 
                # If the embedding was successful, the model name is recorded; otherwise, it is set to None.
                # This allows partial success state in the embedding step, where chunks can still be stored 
                # even if their embedding fails. 
                embedding_model=(
                    embedding_result.model
                    if _is_embedding_success(items_by_index.get(chunk_index))
                    else None
                ),
            )
            # The chunk index is used to pair the original chunk text with its embedding result,
            # allowing the tracking of which chunks were successfully embedded and which were not.
            for chunk_index, chunk_text in enumerate(chunk_texts)
        ]
        # The loop iterates through each chunk index and its corresponding text, checking the embedding result for each chunk.        
        for chunk_index, chunk_text in enumerate(chunk_texts):
            item = items_by_index.get(chunk_index)
            if not _is_embedding_success(item):
                failed_embedding_items.append(
                    item
                    or EmbeddingItemResult(
                        index=chunk_index,
                        text=chunk_text,
                        vector=None,
                        error="Embedding result missing for chunk index.",
                    )
                )

        # The existing chunks for the document are replaced with the new ones generated from the extracted text.
        replace_document_chunks(db_path, document_id=document_id, chunks=chunks)

        # The payload is built and includes details about the embedding process. 
        payload_json = _build_embedding_payload_json(
            provider=embedding_result.provider,
            model=embedding_result.model,
            total_chunks=len(chunk_texts),
            successful_embeddings=embedding_result.success_count,
            failed_items=failed_embedding_items,
        )

        # Still fail a job when no chunk gets a successful embedding.
        if chunk_texts and embedding_result.success_count == 0:
            _persist_failure(
                db_path=db_path,
                document_id=document_id,
                job_id=job_id,
                error_message="Embedding generation failed for all chunks.",
                payload_json=payload_json,
            )
            return "fail"

        mark_job_success(db_path, job_id=job_id, payload_json=payload_json)
        update_document_status(db_path, document_id=document_id, status="success")
        return "success"
    except Exception as exc:
        # In case of any exception during the indexing process, the failure is persisted by marking the job as failed and updating the document status to ```fail```.
        _persist_failure(
            db_path=db_path,
            document_id=document_id,
            job_id=job_id,
            error_message=str(exc),
            payload_json=None,
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
    payload_json: str | None = None,
) -> None:
    safe_error_message = error_message.strip() or "Indexing pipeline failed."

    try:
        mark_job_fail(
            db_path,
            job_id=job_id,
            error_message=safe_error_message,
            payload_json=payload_json,
        )
    finally:
        update_document_status(db_path, document_id=document_id, status="fail")


# _is_embedding_success: helper to make success checks explicit.
def _is_embedding_success(item: EmbeddingItemResult | None) -> bool:
    return bool(item and item.is_success)


# _build_embedding_payload_json: serialize embedding stats and error details for jobs API.
def _build_embedding_payload_json(
    *,
    provider: str,
    model: str,
    total_chunks: int,
    successful_embeddings: int,
    failed_items: list[EmbeddingItemResult],
) -> str:
    warning_errors = [
        {
            "chunk_index": item.index,
            "error": item.error or "Embedding generation failed.",
        }
        for item in failed_items[:10]
    ]
    payload = {
        "embedding": {
            "provider": provider,
            "model": model,
            "total_chunks": total_chunks,
            "successful_embeddings": successful_embeddings,
            "failed_embeddings": len(failed_items),
            "warnings": warning_errors,
        }
    }
    return json.dumps(payload, ensure_ascii=True)
