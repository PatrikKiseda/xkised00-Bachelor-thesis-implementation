"""
Author: Patrik Kiseda
File: src/app/embeddings/providers.py
Description: Embedding client implementations and provider registry. 
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from litellm import embedding as litellm_embedding

from app.core.settings import Settings
from app.embeddings.adapter import EmbeddingBatchResult, EmbeddingClient, EmbeddingItemResult


@dataclass(slots=True)
class OpenAIEmbeddingClient:
    """API-first embedding adapter using LiteLLM."""

    provider: str
    model: str
    api_key: str | None

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Call embedding API once for the whole batch and normalize output.

        Args:
            texts: Text inputs to embed.

        Returns:
            Batch result with vectors or per-item errors.
        """
        if not texts:
            return EmbeddingBatchResult(provider=self.provider, model=self.model, items=[])

        # API call with error handling to avoid exceptions coming out of the embedding pipeline.
        try:
            response = litellm_embedding(
                model=_to_litellm_model(provider=self.provider, model=self.model),
                input=texts,
                api_key=self.api_key,
            )
        except Exception as exc:
            error = f"Embedding API request failed: {exc}"
            return EmbeddingBatchResult(
                provider=self.provider,
                model=self.model,
                items=[
                    EmbeddingItemResult(index=index, text=text, vector=None, error=error)
                    for index, text in enumerate(texts)
                ],
            )

        return _map_litellm_response(
            provider=self.provider,
            model=self.model,
            texts=texts,
            response=response,
        )


@dataclass(slots=True)
class DeterministicEmbeddingClient:
    """Testing-only embedding adapter generating pseudo-vectors."""

    provider: str
    model: str
    vector_size: int = 8

    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
        """Generate deterministic pseudo-vectors so tests avoid external APIs.

        Args:
            texts: Text inputs to embed.

        Returns:
            Batch result with deterministic vectors.
        """
        items = [
            EmbeddingItemResult(
                index=index,
                text=text,
                # hash-based pseudo-embedding.
                vector=_hash_to_vector(text, vector_size=self.vector_size),
            )
            for index, text in enumerate(texts)
        ]
        return EmbeddingBatchResult(provider=self.provider, model=self.model, items=items)


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    """Select API or deterministic embedding runtime mode.

    Args:
        settings: Runtime settings for provider selection.

    Returns:
        Configured embedding client.
    """
    normalized_provider = settings.embedding_provider.strip().lower()

    if not settings.embedding_api_enabled:
        return DeterministicEmbeddingClient(
            provider=normalized_provider or "local",
            model=settings.embedding_model,
            # set deterministic local vectors size to be aligned with configured Qdrant scheme.
            vector_size=settings.qdrant_vector_size,
        )

    if normalized_provider == "openai":
        return OpenAIEmbeddingClient(
            provider=normalized_provider,
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{settings.embedding_provider}' for API embedding mode."
    )


def _to_litellm_model(*, provider: str, model: str) -> str:
    """Map provider/model into LiteLLM model naming.

    Args:
        provider: Provider name.
        model: Model name, with or without provider prefix.

    Returns:
        LiteLLM-compatible model string.
    """
    if "/" in model:
        return model
    return f"{provider}/{model}"


def _map_litellm_response(
    *,
    provider: str,
    model: str,
    texts: list[str],
    response: Any,
) -> EmbeddingBatchResult:
    """Normalize LiteLLM response payload to index-aligned result items.

    Args:
        provider: Provider name used for the request.
        model: Embedding model name used for the request.
        texts: Original input texts.
        response: LiteLLM embedding response.

    Returns:
        Batch result with vectors matched back to input indexes.
    """
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")

    # validate that we have a list of results that can be processed, return item-level errors for all inputs if not processable.
    if not isinstance(data, list):
        error = "Embedding API response did not contain a valid data list."
        return EmbeddingBatchResult(
            provider=provider,
            model=model,
            items=[
                EmbeddingItemResult(index=index, text=text, vector=None, error=error)
                for index, text in enumerate(texts)
            ],
        )

    # mapping of response items by input index, supporting both dict and object style response items.
    vectors_by_index: dict[int, list[float]] = {}
    for list_index, item in enumerate(data):
        raw_index = _read_item_field(item, "index")
        item_index = raw_index if isinstance(raw_index, int) else list_index
        raw_embedding = _read_item_field(item, "embedding")

        if isinstance(raw_embedding, list) and all(
            isinstance(value, (float, int)) for value in raw_embedding
        ):
            vectors_by_index[item_index] = [float(value) for value in raw_embedding]

    items: list[EmbeddingItemResult] = []
    for index, text in enumerate(texts):
        vector = vectors_by_index.get(index)
        if vector is None:
            items.append(
                EmbeddingItemResult(
                    index=index,
                    text=text,
                    vector=None,
                    error="Embedding API response missing vector for input index.",
                )
            )
        else:
            items.append(EmbeddingItemResult(index=index, text=text, vector=vector))

    return EmbeddingBatchResult(provider=provider, model=model, items=items)


def _read_item_field(item: Any, field_name: str) -> Any:
    """Read a field from dict-like or object-like response items.

    Args:
        item: Response item.
        field_name: Field name to read.

    Returns:
        Field value, or None when it is missing.
    """
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _hash_to_vector(text: str, *, vector_size: int) -> list[float]:
    """Create deterministic pseudo-embedding for testing runtime mode.

    Args:
        text: Text to hash into a vector.
        vector_size: Wanted vector size.

    Returns:
        Stable float vector.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector: list[float] = []

    for offset in range(vector_size):
        start = (offset * 4) % len(digest)
        chunk = digest[start : start + 4]
        if len(chunk) < 4:
            chunk += digest[: 4 - len(chunk)]
        integer_value = int.from_bytes(chunk, byteorder="big", signed=False)
        vector.append(integer_value / 4294967295.0)

    return vector
