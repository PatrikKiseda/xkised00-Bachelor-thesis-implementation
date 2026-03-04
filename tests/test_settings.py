"""
Author: Patrik Kiseda
File: tests/test_settings.py
Description: Unit tests for .env loading and critical config validation behavior.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.core.settings import Settings


# _write_env_file: helper that writes temporary .env contents for settings tests.
def _write_env_file(content: str) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".env", encoding="utf-8")
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


# TestSettings: verifies config loading and clear failures for invalid/missing critical values.
class TestSettings(unittest.TestCase):
    # test_loads_values_from_env_file: ensures settings are read from provided .env file.
    def test_loads_values_from_env_file(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "SQLITE_PATH=./data/custom-metadata.db",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=env_file)

        self.assertEqual(settings.qdrant_url, "http://127.0.0.1:6333")
        self.assertEqual(settings.qdrant_collection, "documents")
        self.assertEqual(settings.sqlite_path, "./data/custom-metadata.db")
        self.assertEqual(settings.embedding_provider, "local")

    # test_missing_critical_fields_fail_clearly: missing required keys should trigger validation errors.
    def test_missing_critical_fields_fail_clearly(self) -> None:
        env_file = _write_env_file("QDRANT_URL=http://127.0.0.1:6333")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        error_text = str(ctx.exception)
        self.assertIn("qdrant_collection", error_text)
        self.assertIn("litellm_model", error_text)
        self.assertIn("embedding_provider", error_text)
        self.assertIn("embedding_model", error_text)

    # test_invalid_critical_values_fail_clearly: malformed URL and non-positive timeout should fail.
    def test_invalid_critical_values_fail_clearly(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=127.0.0.1:6333",
                    "QDRANT_TIMEOUT_SECONDS=0",
                    "QDRANT_COLLECTION=documents",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        error_text = str(ctx.exception)
        self.assertIn("QDRANT_URL must start with http:// or https://.", error_text)
        self.assertIn("QDRANT_TIMEOUT_SECONDS must be greater than 0.", error_text)

    # test_openai_provider_requires_api_key: provider-specific dependency should fail without key.
    def test_openai_provider_requires_api_key(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=openai",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        self.assertIn(
            "OPENAI_API_KEY is required when EMBEDDING_PROVIDER is set to 'openai'.",
            str(ctx.exception),
        )

    # test_sqlite_path_cannot_be_blank: sqlite path must reject empty values.
    def test_sqlite_path_cannot_be_blank(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "SQLITE_PATH=   ",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        self.assertIn("sqlite_path", str(ctx.exception))
