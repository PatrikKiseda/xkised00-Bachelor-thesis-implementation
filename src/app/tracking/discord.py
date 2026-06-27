"""
Author: Patrik Kiseda
File: src/app/tracking/discord.py
Description: Lightweight Discord webhook usage tracking for reviewer activity.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any
from urllib.error import URLError
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from fastapi import BackgroundTasks, Request

logger = logging.getLogger(__name__)

MAX_FIELD_VALUE_LENGTH = 900
MAX_EVENT_LENGTH = 80


class DiscordUsageTracker:
    """Send best-effort usage events to a Discord webhook."""

    def __init__(self, *, webhook_url: str | None, enabled: bool = True) -> None:
        """Create a tracker.

        Args:
            webhook_url: Discord webhook URL. Missing value disables delivery.
            enabled: Explicit feature flag for usage tracking.
        """
        self.webhook_url = webhook_url
        self.enabled = enabled and bool(webhook_url)

    def track(
        self,
        *,
        background_tasks: BackgroundTasks | None,
        request: Request,
        event: str,
        details: dict[str, object] | None = None,
    ) -> None:
        """Queue a usage event without blocking the request path.

        Args:
            background_tasks: FastAPI background task collector.
            request: Incoming request used for client metadata.
            event: Short event name.
            details: Optional event-specific fields.
        """
        if not self.enabled or not self.webhook_url:
            return

        payload = self._build_payload(request=request, event=event, details=details or {})
        if background_tasks is None:
            self._send(payload)
            return

        background_tasks.add_task(self._send, payload)

    def _build_payload(
        self,
        *,
        request: Request,
        event: str,
        details: dict[str, object],
    ) -> dict[str, Any]:
        """Build the Discord webhook payload."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        safe_event = _truncate(str(event).strip() or "usage_event", MAX_EVENT_LENGTH)
        fields = [
            {"name": "event", "value": safe_event, "inline": True},
            {"name": "path", "value": request.url.path, "inline": True},
            {"name": "method", "value": request.method, "inline": True},
            {"name": "client", "value": _client_ip(request), "inline": True},
        ]

        user_agent = request.headers.get("user-agent")
        if user_agent:
            fields.append({"name": "user-agent", "value": _truncate(user_agent, MAX_FIELD_VALUE_LENGTH)})

        referer = request.headers.get("referer")
        if referer:
            fields.append({"name": "referer", "value": _truncate(referer, MAX_FIELD_VALUE_LENGTH)})

        for key, value in details.items():
            if value is None:
                continue
            fields.append(
                {
                    "name": _truncate(str(key), 256),
                    "value": _truncate(_format_detail_value(value), MAX_FIELD_VALUE_LENGTH),
                    "inline": len(str(value)) <= 60,
                }
            )

        return {
            "username": "Thesis app tracker",
            "allowed_mentions": {"parse": []},
            "embeds": [
                {
                    "title": "Thesis app usage",
                    "timestamp": now,
                    "color": 0x2563EB,
                    "fields": fields[:25],
                }
            ],
        }

    def _send(self, payload: dict[str, Any]) -> None:
        """Post the payload to Discord and swallow transient failures."""
        if not self.webhook_url:
            return

        request = UrlRequest(
            self.webhook_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "rag-thesis-app-tracker/1.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                if response.status >= 400:
                    logger.warning("Discord tracking webhook returned HTTP %s.", response.status)
        except (OSError, URLError) as exc:
            logger.warning("Failed to send Discord tracking event: %s", exc)


def _client_ip(request: Request) -> str:
    """Resolve the likely public client IP from proxy headers."""
    for header_name in ("cf-connecting-ip", "x-real-ip", "x-forwarded-for"):
        header_value = request.headers.get(header_name)
        if header_value:
            return header_value.split(",", maxsplit=1)[0].strip()

    if request.client is None:
        return "unknown"
    return request.client.host


def _format_detail_value(value: object) -> str:
    """Format detail values for Discord embed fields."""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _truncate(value: str, max_length: int) -> str:
    """Trim a string to fit Discord field limits."""
    if len(value) <= max_length:
        return value
    return f"{value[: max_length - 3]}..."
