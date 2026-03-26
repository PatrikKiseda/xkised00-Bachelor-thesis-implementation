"""
Author: Patrik Kiseda
File: src/app/api/query.py
Description: Dense query endpoint backed by Qdrant and SQLite chunk hydration.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Request

from app.storage.indexing_repository import get_chunks_by_ids

router = APIRouter(prefix="/api/query", tags=["query"])


# DenseQueryRequest: request schema for dense retrieval mode, this is minimal for now, will be extended later (filters, hybrid mode parameters, etc.).
class DenseQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


# query_dense: run embedding + dense vector search and return hydrated chunk hits.
@router.post("/dense")
def query_dense(request: Request, payload: DenseQueryRequest) -> dict[str, object]:
    # Step 1: Generate embedding for the query text using the selected embedding client.
    embedding_result = request.app.state.embedding_client.embed_texts([payload.query])
    # error handling: if embedding generation fails, return a 502 Bad Gateway since this is a dependency failure. 
    if not embedding_result.items or not embedding_result.items[0].is_success:
        raise HTTPException(status_code=502, detail="Failed to generate query embedding.")

    # embedding_result is guaranteed to have at least one item due to the check above.
    query_item = embedding_result.items[0]
    assert query_item.vector is not None

    # Validation of embedding vector size against Qdrant vector size form configuration. 
    if len(query_item.vector) != request.app.state.settings.qdrant_vector_size:
        raise HTTPException(
            status_code=500, # Server error
            detail="Query embedding vector size does not match configured Qdrant vector size.",
        )

    # Make sure the collection exists so empty index query returns 200 + empty hits/results.
    request.app.state.qdrant_store.ensure_collection(
        collection_name=request.app.state.settings.qdrant_collection,
        vector_size=request.app.state.settings.qdrant_vector_size,
    )
    # Step 2: Perform a dense vector search in Qdrant using the query embedding. Check out: https://qdrant.tech/documentation/concepts/points/#searching-by-vector.
    # search_dense defined in src/app/storage/qdrant_store.py, a further abstraction over Qdrant base api.
    dense_hits = request.app.state.qdrant_store.search_dense(
        collection_name=request.app.state.settings.qdrant_collection, 
        # Pass the query vector directly.
        query_vector=query_item.vector,
        # top_k is the number of closest vectors to return, defined in settings.
        limit=payload.top_k,
    )
    if not dense_hits:
        return {
            "mode": "dense",
            "query": payload.query,
            "hits": [],
        }

    # Step 3: Pair the chunk metadata with content from SQLite based on the chunk IDs returned from Qdrant.
    chunks_by_id = get_chunks_by_ids(
        request.app.state.sqlite_db_path,
        chunk_ids=[hit.chunk_id for hit in dense_hits],
    )

    # hits: list of matched chunks with metadata from response of Qdrant search and content from SQLite. Output of endpoint.
    hits: list[dict[str, object]] = []
    for hit in dense_hits:
        chunk = chunks_by_id.get(hit.chunk_id)
        if chunk is None:
            # Skip stale vector hits that no longer exist in SQLite metadata.
            continue
        hits.append(
            {
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "chunk_index": hit.chunk_index,
                "score": hit.score,
                "content": chunk.content,
            }
        )

    # Step 4: Return the list of qdrant queries as the response. Each response from Qdrant is paired with content from SQLite.
    return {
        "mode": "dense",
        "query": payload.query,
        "hits": hits,
    }
