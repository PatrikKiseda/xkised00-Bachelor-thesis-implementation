"""
Author: Patrik Kiseda
File: tests/test_embedding_adapter.py
Description: Unit tests for embedding adapter registry and response normalization.

Provider selection is tested with local/deterministic settings, while LiteLLM
embedding calls are patched. This keeps the suite API-free but still checks the
response mapping and failure behavior.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from helpers import build_settings
from app.embeddings.providers import (
    DeterministicEmbeddingClient,
    OpenAIEmbeddingClient,
    build_embedding_client,
)


class TestEmbeddingAdapter(unittest.TestCase):
    """Provider registry and embedding result normalization tests."""

    def test_registry_returns_deterministic_client_when_api_disabled(self) -> None:
        """API-disabled mode should use deterministic local embeddings."""
        settings = build_settings(
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

    def test_registry_rejects_unsupported_provider_in_api_mode(self) -> None:
        """API mode should reject providers without a real adapter."""
        settings = build_settings(
            embedding_provider="local",
            embedding_api_enabled=True,
            openai_api_key="test-key",
        )

        with self.assertRaises(ValueError) as ctx:
            build_embedding_client(settings)

        self.assertIn("Unsupported EMBEDDING_PROVIDER='local'", str(ctx.exception))

    def test_openai_client_maps_successful_response(self) -> None:
        """OpenAI adapter should map LiteLLM vectors back by input index."""
        settings = build_settings(
            embedding_provider="openai",
            embedding_api_enabled=True,
            openai_api_key="test-key",
        )
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

    def test_openai_client_returns_per_item_errors_on_request_failure(self) -> None:
        """Embedding request failure should become per-item errors."""
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
