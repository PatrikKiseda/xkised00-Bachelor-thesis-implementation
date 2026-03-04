"""
Author: Patrik Kiseda
File: tests/test_health.py
Description: Unit tests for health endpoint behavior with reachable and unreachable Qdrant states.
"""

from __future__ import annotations
import os
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# sys.path adjustment: allows test imports from src without package setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# bootstrap env: provide critical settings so importing app.main does not fail.
os.environ.setdefault("QDRANT_URL", "http://127.0.0.1:6333")
os.environ.setdefault("QDRANT_COLLECTION", "documents")
os.environ.setdefault("LITELLM_MODEL", "openai/gpt-4o-mini")
os.environ.setdefault("EMBEDDING_PROVIDER", "local")
os.environ.setdefault("EMBEDDING_MODEL", "text-embedding-3-small")

from app.core.settings import Settings
from app.main import create_app
from app.storage.qdrant_store import QdrantConnectionStatus

# _build_settings: creates a validated baseline settings object for health tests.
def _build_settings(**overrides: object) -> Settings:
    payload = {
        "app_name": "test-app",
        "qdrant_url": "http://test-qdrant:6333",
        "qdrant_collection": "documents",
        "sqlite_path": ":memory:",
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
    }
    payload.update(overrides)
    return Settings(**payload)

# _HealthyStore: fake store reporting healthy Qdrant connection. Used for positive health endpoint tests.
class _HealthyStore:
    # check_connection: returns a reachable status for positive health tests.
    def check_connection(self) -> QdrantConnectionStatus:
        return QdrantConnectionStatus(reachable=True)


# _UnhealthyStore: fake store reporting unhealthy Qdrant connection. Used for negative health endpoint tests.
class _UnhealthyStore:
    # check_connection: returns an unreachable status for degraded health tests.
    def check_connection(self) -> QdrantConnectionStatus:
        return QdrantConnectionStatus(reachable=False, error="qdrant unreachable")


# TestHealthEndpoint: verifies health output for both reachable and unreachable states.
class TestHealthEndpoint(unittest.TestCase):
    # test_health_reports_qdrant_reachable: expects overall `ok` when store is healthy.
    def test_health_reports_qdrant_reachable(self) -> None:
        app = create_app(
            settings=_build_settings(),
            store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
        )

        with TestClient(app) as client:
            response = client.get("/health")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["qdrant"]["reachable"])
        self.assertTrue(payload["qdrant"]["reachable_on_startup"])
        self.assertEqual(payload["sqlite"]["path"], ":memory:")
        self.assertTrue(payload["sqlite"]["schema_initialized"])

    # test_health_reports_qdrant_unreachable: expects `degraded` when store is unhealthy.
    def test_health_reports_qdrant_unreachable(self) -> None:
        app = create_app(
            settings=_build_settings(),
            store_factory=lambda _: _UnhealthyStore(),  # type: ignore[arg-type]
        )

        with TestClient(app) as client:
            response = client.get("/health")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["qdrant"]["reachable"])
        self.assertFalse(payload["qdrant"]["reachable_on_startup"])
        self.assertEqual(payload["qdrant"]["startup_error"], "qdrant unreachable")
        self.assertEqual(payload["sqlite"]["path"], ":memory:")
        self.assertTrue(payload["sqlite"]["schema_initialized"])

