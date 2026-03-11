"""
Author: Patrik Kiseda
File: src/app/embeddings/__init__.py
Description: Public exports for embedding adapter layer.
    This module is used to define the public API for the embedding layer.
"""

from app.embeddings.adapter import EmbeddingBatchResult, EmbeddingClient, EmbeddingItemResult
from app.embeddings.providers import DeterministicEmbeddingClient, OpenAIEmbeddingClient, build_embedding_client

# includes embedding client interface.
__all__ = [
    "EmbeddingBatchResult",
    "EmbeddingClient",
    "EmbeddingItemResult",
    "DeterministicEmbeddingClient",
    "OpenAIEmbeddingClient",
    "build_embedding_client",
]
