"""
Author: Patrik Kiseda
File: tests/test_extractors.py
Description: Unit tests for text extraction and normalization.

These tests cover supported text, markdown, and PDF inputs plus expected failure
paths. The malformed-PDF case suppresses parser noise because the error is
intentional and asserted through ExtractionError.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from helpers import suppress_expected_pdf_noise
from app.ingestion.extractors import ExtractionError, extract_text


class TestExtractors(unittest.TestCase):
    """Extractor behavior for supported files and expected failures."""

    def test_txt_extraction_normalizes_text(self) -> None:
        """Text extraction should normalize newlines and trailing spaces."""
        result = extract_text("notes.txt", b"  first line  \r\nsecond line\t \r\n\r\n")

        self.assertEqual(result.source_type, "txt")
        self.assertEqual(result.text, "first line\nsecond line")

    def test_md_extraction_sets_source_type(self) -> None:
        """Markdown files should be treated as normalized plain text."""
        result = extract_text("readme.md", b"# Heading \n\n- item  \n")

        self.assertEqual(result.source_type, "md")
        self.assertEqual(result.text, "# Heading\n\n- item")

    def test_unsupported_extension_raises_extraction_error(self) -> None:
        """Unsupported extensions should fail with ExtractionError."""
        with self.assertRaises(ExtractionError) as ctx:
            extract_text("image.png", b"not-an-image")

        self.assertIn("Unsupported file extension", str(ctx.exception))

    def test_pdf_extraction_reads_fixture(self) -> None:
        """PDF extraction should read expected text from the fixture."""
        fixture_path = Path("tests/fixtures/sample_text.pdf")
        result = extract_text("sample_text.pdf", fixture_path.read_bytes())

        self.assertEqual(result.source_type, "pdf")
        self.assertIn("Sample PDF text", result.text)

    def test_pdf_extraction_failure_raises_error(self) -> None:
        """Malformed PDF bytes should fail cleanly."""
        with suppress_expected_pdf_noise():
            with self.assertRaises(ExtractionError) as ctx:
                extract_text("broken.pdf", b"not-a-real-pdf")

        self.assertIn("Failed to extract text from PDF.", str(ctx.exception))
