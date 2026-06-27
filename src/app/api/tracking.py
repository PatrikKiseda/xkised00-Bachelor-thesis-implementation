"""
Author: Patrik Kiseda
File: src/app/api/tracking.py
Description: Browser usage tracking endpoint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/tracking", tags=["tracking"])


class TrackingEventRequest(BaseModel):
    """Client-side usage event payload."""

    event: str = Field(..., min_length=1, max_length=80)
    details: dict[str, Any] = Field(default_factory=dict)


@router.post("/event", status_code=202)
def track_event(
    request: Request,
    background_tasks: BackgroundTasks,
    payload: TrackingEventRequest,
) -> dict[str, str]:
    """Accept a fire-and-forget usage event from the browser.

    Args:
        request: Incoming FastAPI request.
        background_tasks: Background task collector for webhook delivery.
        payload: Event name and optional details.

    Returns:
        A small acknowledgement.
    """
    request.app.state.usage_tracker.track(
        background_tasks=background_tasks,
        request=request,
        event=payload.event,
        details=payload.details,
    )
    return {"status": "accepted"}
