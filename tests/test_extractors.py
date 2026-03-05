"""
Author: Patrik Kiseda
File: tests/test_extractors.py
Description: Unit tests for plain text/markdown extraction and normalization.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.ingestion.extractors import ExtractionError, extract_text


class TestExtractors(unittest.TestCase):
    # test_txt_extraction_normalizes_text: validation of newlines and ending-space normalization.
    def test_txt_extraction_normalizes_text(self) -> None:
        result = extract_text("notes.txt", b"  first line  \r\nsecond line\t \r\n\r\n")

        self.assertEqual(result.source_type, "txt")
        self.assertEqual(result.text, "first line\nsecond line")

    # test_md_extraction_sets_source_type: markdown should be handled as normalized plain text.
    def test_md_extraction_sets_source_type(self) -> None:
        result = extract_text("readme.md", b"# Heading \n\n- item  \n")

        self.assertEqual(result.source_type, "md")
        self.assertEqual(result.text, "# Heading\n\n- item")

    # test_unsupported_extension_raises_extraction_error: unsupported files should fail clearly.
    def test_unsupported_extension_raises_extraction_error(self) -> None:
        with self.assertRaises(ExtractionError) as ctx:
            extract_text("image.png", b"not-an-image")

        self.assertIn("Unsupported file extension", str(ctx.exception))

    # test_pdf_extraction_reads_fixture: PDF text should be extracted from the sample fixture.
    def test_pdf_extraction_reads_fixture(self) -> None:
        fixture_path = Path(__file__).resolve().parent / "fixtures" / "sample_text.pdf"
        result = extract_text("sample_text.pdf", fixture_path.read_bytes())

        self.assertEqual(result.source_type, "pdf")
        self.assertIn("Sample PDF text", result.text)

    # test_pdf_extraction_failure_raises_error: failed PDF bytes should fail cleanly.
    def test_pdf_extraction_failure_raises_error(self) -> None:
        with self.assertRaises(ExtractionError) as ctx:
            extract_text("broken.pdf", b"not-a-real-pdf")

        self.assertIn("Failed to extract text from PDF.", str(ctx.exception))
