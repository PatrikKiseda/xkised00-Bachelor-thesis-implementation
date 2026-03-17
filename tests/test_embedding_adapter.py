"""
Author: Patrik Kiseda
File: tests/test_embedding_adapter.py
Description: Unit tests for embedding adapter registry and response normalization.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.settings import Settings
from app.embeddings.providers import (
    DeterministicEmbeddingClient,
    OpenAIEmbeddingClient,
    build_embedding_client,
)


# _build_settings: helper to create a validated Settings payload for embedding tests.
def _build_settings(**overrides: object) -> Settings:
    payload = {
        "qdrant_url": "http://127.0.0.1:6333",
        "qdrant_collection": "documents",
        "qdrant_vector_size": 8,
        "litellm_model": "openai/gpt-4o-mini",
        "embedding_provider": "openai",
        "embedding_model": "text-embedding-3-small",
        "openai_api_key": "test-key",
        "embedding_api_enabled": True,
    }
    payload.update(overrides)
    return Settings(**payload)


# TestEmbeddingAdapter: validates provider selection and normalized embedding outcomes.
class TestEmbeddingAdapter(unittest.TestCase):
    # test_registry_returns_deterministic_client_when_api_disabled: runtime test mode should avoid API calls.
    def test_registry_returns_deterministic_client_when_api_disabled(self) -> None:
        settings = _build_settings(
            embedding_provider="local",
            embedding_api_enabled=False,
            openai_api_key=None,
        )

        client = build_embedding_client(settings)
        self.assertIsInstance(client, DeterministicEmbeddingClient)

        result = client.embed_texts(["alpha", "beta"])
        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(len(result.items[0].vector or []), 8)

    # test_registry_rejects_unsupported_provider_in_api_mode: API mode must fail on unknown providers.
    def test_registry_rejects_unsupported_provider_in_api_mode(self) -> None:
        settings = _build_settings(embedding_provider="local", embedding_api_enabled=True)

        with self.assertRaises(ValueError) as ctx:
            build_embedding_client(settings)

        self.assertIn("Unsupported EMBEDDING_PROVIDER='local'", str(ctx.exception))

    # test_openai_client_maps_successful_response: verifies index-aligned vector mapping from LiteLLM payload.
    def test_openai_client_maps_successful_response(self) -> None:
        settings = _build_settings()
        client = build_embedding_client(settings)
        self.assertIsInstance(client, OpenAIEmbeddingClient)

        mocked_response = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }
        with patch("app.embeddings.providers.litellm_embedding", return_value=mocked_response):
            result = client.embed_texts(["first", "second"])

        self.assertEqual(result.success_count, 2)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(result.items[0].vector, [0.1, 0.2, 0.3])
        self.assertEqual(result.items[1].vector, [0.4, 0.5, 0.6])

    # test_openai_client_returns_per_item_errors_on_request_failure: failed API call should mark all items as failed.
    def test_openai_client_returns_per_item_errors_on_request_failure(self) -> None:
        client = OpenAIEmbeddingClient(
            provider="openai",
            model="text-embedding-3-small",
            api_key="test-key",
        )

        with patch("app.embeddings.providers.litellm_embedding", side_effect=RuntimeError("boom")):
            result = client.embed_texts(["first", "second"])

        self.assertEqual(result.success_count, 0)
        self.assertEqual(result.failed_count, 2)
        self.assertIn("Embedding API request failed", result.items[0].error or "")
