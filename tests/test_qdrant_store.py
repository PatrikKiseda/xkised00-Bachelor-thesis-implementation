"""
Author: Patrik Kiseda
File: tests/test_qdrant_store.py
Description: Unit tests for QdrantStore connectivity and settings-based client setup.
"""

from __future__ import annotations
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# sys.path adjustment: allows test imports from src without package setup. 
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.settings import Settings
from app.storage.qdrant_store import QdrantStore

# _build_settings: creates a validated baseline settings object for tests.
def _build_settings(**overrides: object) -> Settings:
    payload = {
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_collection": "documents",
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "local",
        "embedding_model": "text-embedding-3-small",
    }
    payload.update(overrides)
    return Settings(**payload)

# _WorkingClient: test double that simulates successful Qdrant calls.
class _WorkingClient:
    # get_collections: fake success path for connection checks.
    def get_collections(self) -> object:
        return object()


# _BrokenClient: test double that simulates failed Qdrant calls.
class _BrokenClient:
    # get_collections: fake failure path for connection checks.
    def get_collections(self) -> object:
        raise RuntimeError("connection refused")


# TestQdrantStore: verifies wrapper behavior and client construction settings.
class TestQdrantStore(unittest.TestCase):
    # test_check_connection_success: expects reachable status when client responds.
    def test_check_connection_success(self) -> None:
        store = QdrantStore(client=_WorkingClient())  # type: ignore[arg-type]

        status = store.check_connection()

        self.assertTrue(status.reachable)
        self.assertIsNone(status.error)

    # test_check_connection_failure: expects error propagation in status object.
    def test_check_connection_failure(self) -> None:
        store = QdrantStore(client=_BrokenClient())  # type: ignore[arg-type]

        status = store.check_connection()

        self.assertFalse(status.reachable)
        self.assertIn("connection refused", status.error or "")

    # test_from_settings_uses_expected_client_configuration: validates env -> client mapping.
    def test_from_settings_uses_expected_client_configuration(self) -> None:
        settings = _build_settings(
            qdrant_url="http://qdrant.local:6333",
            qdrant_api_key="secret",
            qdrant_timeout_seconds=9.5,
        )

        # patch decorator target: intercepts QdrantClient construction for assertion.
        with patch("app.storage.qdrant_store.QdrantClient") as mock_client_cls:
            QdrantStore.from_settings(settings)

        mock_client_cls.assert_called_once_with(
            url="http://qdrant.local:6333",
            api_key="secret",
            timeout=9.5,
        )
