"""
Author: Patrik Kiseda
File: src/app/embeddings/adapter.py
Description: Embedding adapter layer that defines a standardized interface for embedding generation across different providers.
    This module includes data structures for embedding results and a simple protocol for embedding clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class EmbeddingItemResult:
    """Normalized per-input embedding outcome, including failures."""

    index: int
    text: str
    vector: list[float] | None
    error: str | None = None

    @property
    def is_success(self) -> bool:
        """Check if this item has a usable vector.

        Returns:
            True when there is no error and vector exists.
        """
        return self.error is None and self.vector is not None

@dataclass(slots=True)
class EmbeddingBatchResult:
    """Batch-level result used by indexing for stats and error handling."""

    provider: str
    model: str
    items: list[EmbeddingItemResult]

    @property
    def success_count(self) -> int:
        """Count chunks that produced a usable vector.

        Returns:
            Number of successful embedding items.
        """
        return sum(1 for item in self.items if item.is_success)

    @property
    def failed_count(self) -> int:
        """Count chunks that failed embedding generation.

        Returns:
            Number of failed embedding items.
        """
        return len(self.items) - self.success_count

class EmbeddingClient(Protocol):
    """Adapter interface for all embedding providers."""

    provider: str
    model: str

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Create embeddings for texts in index-aligned order.

        Args:
            texts: Text inputs to embed.

        Returns:
            Batch result with one item per input text.
        """
        ...
