"""
Author: Patrik Kiseda
File: src/app/api/documents.py
Description: Document ingestion API routes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.extractors import SOURCE_TYPE_BY_EXTENSION, ExtractionError, extract_text
from app.storage.document_repository import insert_document, list_documents

router = APIRouter(prefix="/api/documents", tags=["documents"])

# upload_document: handles file uploads, text extraction, and metadata storage in SQLite.
@router.post("/upload", status_code=201)
async def upload_document(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
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

    file_id = str(uuid4())
    stored_path = Path(request.app.state.storage_dir) / f"{file_id}{extension}"
    stored_path.parent.mkdir(parents=True, exist_ok=True)
    stored_path.write_bytes(content)

    # attempt extraction and metadata recording, with cleanup on failure.
    try:
        extracted = extract_text(file.filename, content)
        record = insert_document(
            request.app.state.sqlite_db_path,
            document_id=file_id,
            filename=file.filename,
            source_type=extracted.source_type,
            source_path=str(stored_path),
            size_bytes=len(content),
            checksum=hashlib.sha256(content).hexdigest(),
            status="ready",
        )
    except ExtractionError as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Failed to store uploaded document.") from exc

    # return of the created document metadata.
    return {
        "id": record.id,
        "filename": record.filename,
        "source_type": record.source_type,
        "status": record.status,
        "storage_path": record.source_path,
        "created_at": record.created_at,
    }


# get_documents: retrieves list of ingeted documents, useful for verifying sucessful uploads.
@router.get("")
def get_documents(request: Request) -> dict[str, list[dict[str, object]]]:
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
