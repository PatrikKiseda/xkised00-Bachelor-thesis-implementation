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


# OpenAIEmbeddingClient: API-first embedding adapter using LiteLLM.
@dataclass(slots=True)
class OpenAIEmbeddingClient:
    provider: str
    model: str
    api_key: str | None

    # embed_texts: call embedding API once for the whole batch and normalize output.
    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
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


# DeterministicEmbeddingClient: TESTING-ONLY embedding adapter generating pseudo-vectors..
@dataclass(slots=True)
class DeterministicEmbeddingClient:
    provider: str
    model: str
    vector_size: int = 8

    # embed_texts: generate deterministic pseudo-vectors so tests avoid external APIs.
    def embed_texts(self, texts: list[str]) -> EmbeddingBatchResult:
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


# build_embedding_client: provider registry selecting API or deterministic runtime mode.
def build_embedding_client(settings: Settings) -> EmbeddingClient:
    normalized_provider = settings.embedding_provider.strip().lower()

    if not settings.embedding_api_enabled:
        return DeterministicEmbeddingClient(
            provider=normalized_provider or "local",
            model=settings.embedding_model,
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


# _to_litellm_model: map provider/model into LiteLLM model naming.
def _to_litellm_model(*, provider: str, model: str) -> str:
    if "/" in model:
        return model
    return f"{provider}/{model}"


# _map_litellm_response: normalize LiteLLM response payload to index-aligned result items.
def _map_litellm_response(
    *,
    provider: str,
    model: str,
    texts: list[str],
    response: Any,
) -> EmbeddingBatchResult:
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


# _read_item_field: helper to read fields from response items.
def _read_item_field(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)


# _hash_to_vector: deterministic pseudo-embedding for testing runtime mode.
def _hash_to_vector(text: str, *, vector_size: int) -> list[float]:
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
