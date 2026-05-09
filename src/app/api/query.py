"""
Author: Patrik Kiseda
File: src/app/api/query.py
Description: Query and answer endpoints.
"""

from __future__ import annotations

import logging
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.generation.service import AnswerGenerator, resolve_final_prompt
from app.retrieval.service import RetrievedChunk, build_retriever

router = APIRouter(prefix="/api/query", tags=["query"])
logger = logging.getLogger(__name__)

QueryMode = Literal["dense", "lexical", "hybrid"]


class RetrievalQueryRequest(BaseModel):
    """Request schema for direct retrieval endpoints."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)

class AnswerQueryRequest(RetrievalQueryRequest):
    """Request schema for answer generation endpoint."""

    mode: QueryMode = "dense"
    include_context_in_prompt: bool = True


@router.post("/dense")
def query_dense(request: Request, payload: RetrievalQueryRequest) -> dict[str, object]:
    """Run embedding + dense vector search.

    Args:
        request: FastAPI request with app state.
        payload: Retrieval query payload.

    Returns:
        API response with hydrated chunk hits.
    """
    return _run_retrieval_query(request=request, mode="dense", query=payload.query, top_k=payload.top_k)

@router.post("/lexical")
def query_lexical(request: Request, payload: RetrievalQueryRequest) -> dict[str, object]:
    """Run lexical search using SQLite FTS5.

    Args:
        request: FastAPI request with app state.
        payload: Retrieval query payload.

    Returns:
        API response with hits in the shared retrieval shape.
    """
    return _run_retrieval_query(request=request, mode="lexical", query=payload.query, top_k=payload.top_k)

@router.post("/hybrid")
def query_hybrid(request: Request, payload: RetrievalQueryRequest) -> dict[str, object]:
    """Run dense + lexical retrieval and merge ranked results with RRF.

    Args:
        request: FastAPI request with app state.
        payload: Retrieval query payload.

    Returns:
        API response with fused chunk hits.
    """
    return _run_retrieval_query(request=request, mode="hybrid", query=payload.query, top_k=payload.top_k)

@router.post("/answer")
def query_answer(request: Request, payload: AnswerQueryRequest) -> dict[str, object]:
    """Run retrieval and generate an answer from the retrieved sources.

    Args:
        request: FastAPI request with app state.
        payload: Answer query payload.

    Returns:
        API response with answer and sources.
    """
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
        logger.exception("Failed to generate final answer.")
        raise HTTPException(status_code=502, detail="Failed to generate final answer.") from exc

    return {
        "mode": payload.mode,
        "query": payload.query,
        "answer": result.answer,
        "sources": [_serialize_source(source) for source in result.sources],
    }

@router.post("/prompt-debug")
def query_prompt_debug(request: Request, payload: AnswerQueryRequest) -> dict[str, object]:
    """Return final prompt without calling the generation service.

    Args:
        request: FastAPI request with app state.
        payload: Answer query payload.

    Returns:
        API response with final prompt and sources.
    """
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

def _run_retrieval_query(
    *,
    request: Request,
    mode: QueryMode,
    query: str,
    top_k: int,
) -> dict[str, object]:
    """Run retrieval queries and return hits in consistent API format.

    Args:
        request: FastAPI request with app state.
        mode: Retrieval mode to use.
        query: User query text.
        top_k: Number of hits to return.

    Returns:
        API response with retrieval hits.
    """
    retriever = build_retriever(
        mode=mode,
        db_path=request.app.state.sqlite_db_path,
        embedding_client=request.app.state.embedding_client,
        qdrant_store=request.app.state.qdrant_store,
        qdrant_collection=request.app.state.settings.qdrant_collection,
        qdrant_vector_size=request.app.state.settings.qdrant_vector_size,
    )
    hits = retriever.retrieve(query=query, top_k=top_k)

    return {
        "mode": mode,
        "query": query,
        "hits": [_serialize_hit(hit) for hit in hits],
    }


def _serialize_hit(hit: RetrievedChunk) -> dict[str, object]:
    """Serialize a RetrievedChunk as a direct retrieval hit.

    Args:
        hit: Retrieved chunk to serialize.

    Returns:
        Dict used in retrieval responses.
    """
    return {
        "chunk_id": hit.chunk_id,
        "document_id": hit.document_id,
        "chunk_index": hit.chunk_index,
        "score": hit.score,
        "content": hit.content,
    }

def _serialize_source(source: RetrievedChunk) -> dict[str, object]:
    """Serialize a RetrievedChunk as an answer source.

    Args:
        source: Retrieved source to serialize.

    Returns:
        Dict used in answer responses.
    """
    return {
        "source_id": source.source_id,
        "chunk_id": source.chunk_id,
        "document_id": source.document_id,
        "filename": source.filename,
        "chunk_index": source.chunk_index,
        "score": source.score,
        "content": source.content,
    }
