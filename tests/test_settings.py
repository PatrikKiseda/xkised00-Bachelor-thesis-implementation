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

from app.core.settings import (
    DEFAULT_CHUNK_OVERLAP_CHARS,
    DEFAULT_CHUNK_SIZE_CHARS,
    Settings,
)


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
                    "QDRANT_VECTOR_SIZE=8",
                    "SQLITE_PATH=./data/custom-metadata.db",
                    "STORAGE_DIR=./data/custom-uploads",
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
        self.assertEqual(settings.storage_dir, "./data/custom-uploads")
        self.assertEqual(settings.embedding_provider, "local")
        self.assertTrue(settings.embedding_api_enabled)
        self.assertEqual(settings.chunk_size_chars, DEFAULT_CHUNK_SIZE_CHARS)
        self.assertEqual(settings.chunk_overlap_chars, DEFAULT_CHUNK_OVERLAP_CHARS)

    # test_env_file_overrides_inherited_environment: repo-local .env should win over stray shell exports.
    def test_env_file_overrides_inherited_environment(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=8",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=openai",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                    "OPENAI_API_KEY=real-from-env-file",
                ]
            )
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "placeholder-from-shell"}, clear=True):
            settings = Settings(_env_file=env_file)

        self.assertEqual(settings.openai_api_key, "real-from-env-file")

    # test_missing_critical_fields_fail_clearly: missing required keys should trigger validation errors.
    def test_missing_critical_fields_fail_clearly(self) -> None:
        env_file = _write_env_file("QDRANT_URL=http://127.0.0.1:6333")

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        error_text = str(ctx.exception)
        self.assertIn("qdrant_collection", error_text)
        self.assertIn("qdrant_vector_size", error_text)
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
                    "QDRANT_VECTOR_SIZE=8",
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
                    "QDRANT_VECTOR_SIZE=8",
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

    # test_openai_without_key_is_allowed_when_api_mode_disabled: runtime test mode must bypass API-key requirement.
    def test_openai_without_key_is_allowed_when_api_mode_disabled(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=8",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=openai",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                    "EMBEDDING_API_ENABLED=false",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            settings = Settings(_env_file=env_file)

        self.assertFalse(settings.embedding_api_enabled)

    # test_sqlite_path_cannot_be_blank: sqlite path must reject empty values.
    def test_sqlite_path_cannot_be_blank(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=8",
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

    # test_storage_dir_cannot_be_blank: storage dir must reject empty values.
    def test_storage_dir_cannot_be_blank(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=8",
                    "SQLITE_PATH=./data/app.db",
                    "STORAGE_DIR=    ",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        self.assertIn("storage_dir", str(ctx.exception))

    # test_chunk_overlap_must_be_smaller_than_chunk_size: invalid overlap-size relation should fail.
    def test_chunk_overlap_must_be_smaller_than_chunk_size(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=8",
                    "SQLITE_PATH=./data/app.db",
                    "STORAGE_DIR=./data/uploads",
                    f"CHUNK_SIZE_CHARS={DEFAULT_CHUNK_SIZE_CHARS}",
                    f"CHUNK_OVERLAP_CHARS={DEFAULT_CHUNK_SIZE_CHARS}",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        self.assertIn("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS.", str(ctx.exception))

    # test_chunk_overlap_cannot_be_negative: chunk overlap must reject negative values.
    def test_chunk_overlap_cannot_be_negative(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=8",
                    "SQLITE_PATH=./data/app.db",
                    "STORAGE_DIR=./data/uploads",
                    f"CHUNK_SIZE_CHARS={DEFAULT_CHUNK_SIZE_CHARS}",
                    "CHUNK_OVERLAP_CHARS=-1",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        self.assertIn("CHUNK_OVERLAP_CHARS must be greater than or equal to 0.", str(ctx.exception))

    # test_qdrant_vector_size_must_be_positive: deterministic dense schema must reject non-positive sizes.
    def test_qdrant_vector_size_must_be_positive(self) -> None:
        env_file = _write_env_file(
            "\n".join(
                [
                    "QDRANT_URL=http://127.0.0.1:6333",
                    "QDRANT_COLLECTION=documents",
                    "QDRANT_VECTOR_SIZE=0",
                    "LITELLM_MODEL=openai/gpt-4o-mini",
                    "EMBEDDING_PROVIDER=local",
                    "EMBEDDING_MODEL=text-embedding-3-small",
                ]
            )
        )

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as ctx:
                Settings(_env_file=env_file)

        self.assertIn("QDRANT_VECTOR_SIZE must be greater than 0.", str(ctx.exception))
