"""
Author: Patrik Kiseda
File: src/app/storage/qdrant_store.py
Description: Small Qdrant client wrapper with connection check support.
"""

from __future__ import annotations
from dataclasses import dataclass
from qdrant_client import QdrantClient
from app.core.settings import Settings

# QdrantConnectionStatus decorator: Status container for connectivity checks. https://docs.python.org/3/library/dataclasses.html for details.
@dataclass(slots=True)
# QdrantConnectionStatus: normalized result payload returned by connection checks.
class QdrantConnectionStatus:
    reachable: bool
    error: str | None = None

# QdrantStore: small wrapper for QdrantClient. 
# Used by app/script health checks.
class QdrantStore:
    # __init__: stores the already-created Qdrant client instance.
    def __init__(self, client: QdrantClient):
        self._client = client

    # from_settings decorator: exposes factory construction.
    @classmethod
    # from_settings: builds a configured Qdrant client using env-backed settings. See get_settings() in src/app/core/settings.py for details on how settings are loaded.
    def from_settings(cls, settings: Settings) -> "QdrantStore":
        client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout_seconds,
        )
        return cls(client)

    # check_connection: probes Qdrant and returns a status object instead of raising.
    def check_connection(self) -> QdrantConnectionStatus:
        try:
            self._client.get_collections()
            return QdrantConnectionStatus(reachable=True)
        except Exception as exc:  # pragma: no cover - wrapped for app-level handling
            return QdrantConnectionStatus(reachable=False, error=str(exc))

    # ping: convenience boolean check for quick reachability checks.
    def ping(self) -> bool:
        return self.check_connection().reachable
