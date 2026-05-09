"""
Author: Patrik Kiseda
File: src/app/api/documents.py
Description: Document ingestion API routes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile

from app.ingestion.extractors import SOURCE_TYPE_BY_EXTENSION
from app.ingestion.indexing_pipeline import run_indexing_pipeline
from app.storage.document_repository import insert_document, list_documents, update_document_status
from app.storage.indexing_repository import create_job

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.post("/upload", status_code=201)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, object]:
    """Handle file upload, metadata save, and background indexing job.

    Args:
        request: FastAPI request with app state.
        background_tasks: FastAPI background task manager.
        file: Uploaded file to ingest.

    Returns:
        Created document metadata and job id.
    """
    # validation of file presence.
    if not file.filename:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    # validation of supported file extension.
    extension = Path(file.filename).suffix.lower()
    if extension not in SOURCE_TYPE_BY_EXTENSION:
        raise HTTPException(status_code=415, detail="Unsupported file type.")

    # read file content and validate non-empty.
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    document_id = str(uuid4())
    job_id = str(uuid4())
    source_type = SOURCE_TYPE_BY_EXTENSION[extension]

    stored_path = Path(request.app.state.storage_dir) / f"{document_id}{extension}"
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)

    # attempt metadata recording + job creation, with cleanup on failure.
    try:
        record = insert_document(
            request.app.state.sqlite_db_path,
            document_id=document_id,
            filename=file.filename,
            source_type=source_type,
            source_path=str(stored_path),
            size_bytes=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
            status="pending",
        )
        create_job(
            request.app.state.sqlite_db_path,
            job_id=job_id,
            job_type="indexing",
            document_id=document_id,
            status="pending",
        )
    # In a case of any exception during metadata saving or job creation, the uploaded file is deleted and the document status is updated to ```fail``` to indicate the unsuccessful ingestion.
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        update_document_status(
            request.app.state.sqlite_db_path,
            document_id=document_id,
            status="fail",
        )
        raise HTTPException(status_code=500, detail="Failed to store uploaded document.") from exc

    # Adds the indexing pipeline to a list of background tasks to be executed asynchronously. 
    # Allows the API to return a response immediatelly while the indexing process runs in the background.
    background_tasks.add_task(
        run_indexing_pipeline,
        db_path=request.app.state.sqlite_db_path,
        document_id=document_id,
        filename=file.filename,
        source_path=str(stored_path),
        job_id=job_id,
        chunk_size_chars=request.app.state.settings.chunk_size_chars,
        chunk_overlap_chars=request.app.state.settings.chunk_overlap_chars,
        embedding_client=request.app.state.embedding_client,
        qdrant_store=request.app.state.qdrant_store,
        qdrant_collection=request.app.state.settings.qdrant_collection,
        qdrant_vector_size=request.app.state.settings.qdrant_vector_size,
    )

    # return of the created document metadata.
    return {
        "id": record.id,
        "job_id": job_id,
        "filename": record.filename,
        "source_type": record.source_type,
        "status": record.status,
        "storage_path": record.source_path,
        "created_at": record.created_at,
    }


@router.get("")
def get_documents(request: Request) -> dict[str, list[dict[str, object]]]:
    """List ingested documents, useful for checking uploads.

    Args:
        request: FastAPI request with app state.

    Returns:
        API response with document records.
    """
    records = list_documents(request.app.state.sqlite_db_path)
    return {
        "documents": [
            {
                "id": record.id,
                "filename": record.filename,
                "source_type": record.source_type,
                "status": record.status,
                "storage_path": record.source_path,
                "size_bytes": record.size_bytes,
                "created_at": record.created_at,
            }
            for record in records
        ]
    }
