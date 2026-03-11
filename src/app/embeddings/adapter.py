"""
Author: Patrik Kiseda
File: src/app/embeddings/adapter.py
Description: Embedding adapter layer that defines a standardized interface for embedding generation across different providers.
    This module includes data structures for embedding results and a simple protocol for embedding clients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


# EmbeddingItemResult: normalized per-input embedding outcome, including failures.
@dataclass(slots=True)
class EmbeddingItemResult:
    index: int
    text: str
    vector: list[float] | None
    error: str | None = None

    # is_success: helper to keep success checks readable in pipeline code.
    @property
    def is_success(self) -> bool:
        return self.error is None and self.vector is not None

# EmbeddingBatchResult: batch-level result used by indexing for stats and error handling.
@dataclass(slots=True)
class EmbeddingBatchResult:
    provider: str
    model: str
    items: list[EmbeddingItemResult]

    # success_count: number of chunks that produced a usable vector.
    @property
    def success_count(self) -> int:
        return sum(1 for item in self.items if item.is_success)

    # failed_count: number of chunks that failed embedding generation.
    @property
    def failed_count(self) -> int:
        return len(self.items) - self.success_count

# EmbeddingClient: adapter interface for all embedding providers.
class EmbeddingClient(Protocol):
    provider: str
    model: str

    # embed_texts: create embeddings for the given text list in index-aligned order.
    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        ...
