"""
Author: Patrik Kiseda
File: tests/test_chunker.py
Description: Unit tests for recursive chunking with overlap.

The tests focus on stable behavior: empty input, small text, deterministic
recursive splitting, and overlap continuity. No external services or file system
fixtures are needed.
"""

from __future__ import annotations

import unittest

import helpers
from app.core.settings import DEFAULT_CHUNK_OVERLAP_CHARS, DEFAULT_CHUNK_SIZE_CHARS
from app.ingestion.chunker import chunk_text_recursive

# Shared defaults imported from settings so tests stay aligned with runtime defaults.
DEFAULT_SIZE = DEFAULT_CHUNK_SIZE_CHARS
DEFAULT_OVERLAP = DEFAULT_CHUNK_OVERLAP_CHARS

# Smaller values used in behavior-specific tests where we need many chunks quickly.
SMALL_TEST_SIZE = 120
SMALL_TEST_OVERLAP = 20


class TestChunker(unittest.TestCase):
    """Behavior checks for recursive chunk splitting and overlap."""

    def test_empty_text_returns_empty_chunks(self) -> None:
        """Whitespace-only input should not create chunks."""
        chunks = chunk_text_recursive(
            "   \n\t  ",
            chunk_size_chars=DEFAULT_SIZE,
            chunk_overlap_chars=DEFAULT_OVERLAP,
        )

        self.assertEqual(chunks, [])

    def test_small_text_returns_single_chunk(self) -> None:
        """Short text should stay as one unchanged chunk."""
        text = "short text for chunking"

        chunks = chunk_text_recursive(
            text,
            chunk_size_chars=DEFAULT_SIZE,
            chunk_overlap_chars=DEFAULT_OVERLAP,
        )

        self.assertEqual(chunks, [text])

    def test_recursive_chunking_is_deterministic(self) -> None:
        """Repeated chunking with same input/settings should be stable."""
        text = (
            "Paragraph one. Sentence one and sentence two.\n\n"
            "Paragraph two with more words and more words.\n"
        ) * 30

        # Run twice with identical settings and verify stable ordering/content.
        chunks_one = chunk_text_recursive(
            text,
            chunk_size_chars=SMALL_TEST_SIZE,
            chunk_overlap_chars=SMALL_TEST_OVERLAP,
        )
        chunks_two = chunk_text_recursive(
            text,
            chunk_size_chars=SMALL_TEST_SIZE,
            chunk_overlap_chars=SMALL_TEST_OVERLAP,
        )

        self.assertEqual(chunks_one, chunks_two)
        self.assertGreater(len(chunks_one), 1)
        self.assertTrue(all(len(chunk) <= SMALL_TEST_SIZE for chunk in chunks_one))

    def test_overlap_is_applied_between_adjacent_chunks(self) -> None:
        """Neighboring chunks should share the configured overlap window."""
        text = (
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
            "nu xi omicron pi rho sigma tau upsilon phi chi psi omega. "
        ) * 8
        overlap = 15

        chunks = chunk_text_recursive(
            text,
            chunk_size_chars=100,
            chunk_overlap_chars=overlap,
        )

        self.assertGreater(len(chunks), 1)

        # Neighboring chunks should share the configured overlap window.
        for index in range(1, len(chunks)):
            self.assertEqual(chunks[index - 1][-overlap:], chunks[index][:overlap])
