"""
Author: Patrik Kiseda
File: src/app/api/query.py
Description: Query and answer endpoints.
"""

from __future__ import annotations

from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.generation.service import AnswerGenerator, resolve_final_prompt
from app.retrieval.service import RetrievedChunk, build_retriever

router = APIRouter(prefix="/api/query", tags=["query"])

QueryMode = Literal["dense", "lexical", "hybrid"]


# DenseQueryRequest: request schema for dense retrieval mode, this is minimal for now, will be extended later (filters, hybrid mode parameters, etc.).
class DenseQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)

# AnswerQueryRequest: request schema for answer generation endpoint, extending DenseQueryRequest with additional parameters for answer generation.
class AnswerQueryRequest(DenseQueryRequest):
    mode: QueryMode = "dense"
    include_context_in_prompt: bool = True


# query_dense: run embedding + dense vector search and return hydrated chunk hits.
@router.post("/dense")
def query_dense(request: Request, payload: DenseQueryRequest) -> dict[str, object]:
    retriever = build_retriever(
        mode="dense",
        db_path=request.app.state.sqlite_db_path,
        embedding_client=request.app.state.embedding_client,
        qdrant_store=request.app.state.qdrant_store,
        qdrant_collection=request.app.state.settings.qdrant_collection,
        qdrant_vector_size=request.app.state.settings.qdrant_vector_size,
    )
    hits = retriever.retrieve(query=payload.query, top_k=payload.top_k)
    if not hits:
        return {
            "mode": "dense",
            "query": payload.query,
            "hits": [],
        }

    # Step 4: Return the list of qdrant queries as the response. Each response from Qdrant is paired with content from SQLite.
    return {
        "mode": "dense",
        "query": payload.query,
        "hits": [_serialize_dense_hit(hit) for hit in hits],
    }

# query_answer: run retrieval (dense, lexical, or hybrid) and then generate an answer using the retrieved sources.
@router.post("/answer")
def query_answer(request: Request, payload: AnswerQueryRequest) -> dict[str, object]:
    retriever = build_retriever(
        mode=payload.mode,
        db_path=request.app.state.sqlite_db_path,
        embedding_client=request.app.state.embedding_client,
        qdrant_store=request.app.state.qdrant_store,
        qdrant_collection=request.app.state.settings.qdrant_collection,
        qdrant_vector_size=request.app.state.settings.qdrant_vector_size,
    )
    sources = retriever.retrieve(query=payload.query, top_k=payload.top_k)
    answer_generator = AnswerGenerator(generation_client=request.app.state.generation_client)

    # Step 4: Generate the final answer using the retrieved sources and return it. If generation fails, return an error response.
    try:
        result = answer_generator.generate_answer(
            query=payload.query,
            sources=sources,
            include_context_in_prompt=payload.include_context_in_prompt,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Failed to generate final answer.") from exc

    return {
        "mode": payload.mode,
        "query": payload.query,
        "answer": result.answer,
        "sources": [_serialize_source(source) for source in result.sources],
    }

# debug endpoint to return the final prompt without actually calling the generation service, used for testing.
@router.post("/prompt-debug")
def query_prompt_debug(request: Request, payload: AnswerQueryRequest) -> dict[str, object]:
    retriever = build_retriever(
        mode=payload.mode,
        db_path=request.app.state.sqlite_db_path,
        embedding_client=request.app.state.embedding_client,
        qdrant_store=request.app.state.qdrant_store,
        qdrant_collection=request.app.state.settings.qdrant_collection,
        qdrant_vector_size=request.app.state.settings.qdrant_vector_size,
    )
    sources = retriever.retrieve(query=payload.query, top_k=payload.top_k)
    prompt = resolve_final_prompt(
        query=payload.query,
        sources=sources,
        include_context_in_prompt=payload.include_context_in_prompt,
    )

    return {
        "mode": payload.mode,
        "query": payload.query,
        "include_context_in_prompt": payload.include_context_in_prompt,
        "prompt": prompt,
        "sources": [_serialize_source(source) for source in sources],
    }

# Serialization helpers to convert RetrievedChunk objects into dicts for API responses. 
def _serialize_dense_hit(hit: RetrievedChunk) -> dict[str, object]:
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "content": hit.content,
    }

# Serialization helper for RetrievedChunk -> dict.
def _serialize_source(source: RetrievedChunk) -> dict[str, object]:
    return {
        "source_id": source.source_id,
        "chunk_id": source.chunk_id,
        "document_id": source.document_id,
        "filename": source.filename,
        "chunk_index": source.chunk_index,
        "score": source.score,
        "content": source.content,
    }
