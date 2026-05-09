"""
Author: Patrik Kiseda
File: src/app/api/jobs.py
Description: Jobs API routes for indexing lifecycle visibility.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from app.storage.indexing_repository import list_jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.get("")
def get_jobs(
    request: Request,
    document_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, list[dict[str, object | None]]]:
    """Retrieve indexing jobs, optionally filtered by document id.

    Args:
        request: FastAPI request with app state.
        document_id: Optional document id filter.
        limit: Maximum number of jobs to return.

    Returns:
        API response with job records.
    """
    records = list_jobs(
        request.app.state.sqlite_db_path,
        document_id=document_id,
        limit=limit,
    )
    return {
        "jobs": [
            {
                "id": record.id,
                "job_type": record.job_type,
                "status": record.status,
                "document_id": record.document_id,
                "payload_json": record.payload_json,
                "error_message": record.error_message,
                "created_at": record.created_at,
                "started_at": record.started_at,
                "finished_at": record.finished_at,
            }
            for record in records
        ]
    }
