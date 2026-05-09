"""
Author: Patrik Kiseda
File: src/app/ingestion/extractors.py
Description: Lightweight file-text extractors used by ingestion endpoints.
"""

from __future__ import annotations

from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


class ExtractionError(ValueError):
    """Raised when text extraction fails for supported/known file types."""


@dataclass(slots=True)
class ExtractionResult:
    """Extracted text plus detected source type."""

    source_type: str
    text: str


SOURCE_TYPE_BY_EXTENSION: dict[str, str] = {
    ".txt": "txt",
    ".md": "md",
    ".pdf": "pdf",
}


def normalize_text(text: str) -> str:
    """Normalize newlines and trim trailing whitespace per line.

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned text.
    """
    # Normalization of line endings and trim of trailing whitespace per line.
    normalized_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized_newlines.split("\n")]
    return "\n".join(lines).strip()

def extract_text(filename: str, content: bytes) -> ExtractionResult:
    """Extract text from supported PDF and UTF-8 text files.

    Args:
        filename: Original filename used to detect file type.
        content: Uploaded file bytes.

    Returns:
        Extracted and normalized text result.
    """
    extension = Path(filename).suffix.lower()
    source_type = SOURCE_TYPE_BY_EXTENSION.get(extension)
    if source_type is None:
        raise ExtractionError(f"Unsupported file extension: {extension or '<none>'}")

    if source_type == "pdf":
        try:
            pdf_reader = PdfReader(BytesIO(content))
            page_texts = [(page.extract_text() or "") for page in pdf_reader.pages]
        except Exception as exc:
            raise ExtractionError("Failed to extract text from PDF.") from exc
        decoded = "\n".join(page_texts)
    else:
        try:
            decoded = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError("Failed to decode file as UTF-8 text.") from exc

    normalized = normalize_text(decoded)
    if not normalized:
        raise ExtractionError("No extractable text found in the uploaded file.")

    return ExtractionResult(
        source_type=source_type,
        text=normalized,
    )
