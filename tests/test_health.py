"""
Author: Patrik Kiseda
File: tests/test_health.py
Description: Unit tests for health endpoint behavior with reachable and unreachable Qdrant states.
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

# sys.path adjustment: allows test imports from src without package setup.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.settings import Settings
from app.main import create_app
from app.storage.qdrant_store import QdrantConnectionStatus


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
            settings=Settings(app_name="test-app", qdrant_url="http://test-qdrant:6333"),
            store_factory=lambda _: _HealthyStore(),  # type: ignore[arg-type]
        )

        with TestClient(app) as client:
            response = client.get("/health")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["qdrant"]["reachable"])
        self.assertTrue(payload["qdrant"]["reachable_on_startup"])

    # test_health_reports_qdrant_unreachable: expects `degraded` when store is unhealthy.
    def test_health_reports_qdrant_unreachable(self) -> None:
        app = create_app(
            settings=Settings(app_name="test-app", qdrant_url="http://test-qdrant:6333"),
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
