"""
Author: Patrik Kiseda
File: src/app/generation/prompt_builder.py
Description: Prompt construction for a response with retrieval enabled or a no-context answer generation.
"""

from __future__ import annotations

from app.retrieval.service import RetrievedChunk


def build_grounded_answer_prompt(*, query: str, sources: list[RetrievedChunk]) -> str:
    """Create a grounded prompt from the user query and retrieved chunks.

    Args:
        query: User question.
        sources: Retrieved chunks to include.

    Returns:
        Prompt text for grounded answer generation.
    """
    source_header = _build_source_header(sources=sources)
    return (
        "You are a retrieval-grounded assistant.\n"
        "Answer only from the retrieved context when it is provided.\n"
        "Do not use outside knowledge.\n"
        "Cite supporting claims inline using source ids like [S1] and [S2].\n"
        "If the retrieved context is missing or insufficient, say that clearly.\n\n"
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{source_header}\n\n"
        "Answer:"
    )


def build_no_context_prompt(*, query: str) -> str:
    """Create prompt for retrieval mode when no hits were found.

    Args:
        query: User question.

    Returns:
        Prompt text for no-context answer generation.
    """
    return (
        "You are a helpful assistant.\n"
        "No retrieved context was available for this question, so answer without citations and make it clear there is no retrieved context to ground on.\n"
        "Still try to provide a helpful response based on your general knowledge but answer with only what you are really confident about.\n\n"
        f"Question:\n{query}\n\n"
        "Answer:"
    )

def _build_source_header(*, sources: list[RetrievedChunk]) -> str:
    """Format retrieved sources into a string for the prompt.

    Args:
        sources: Retrieved chunks to format.

    Returns:
        Source block text.
    """
    return "\n\n".join(
        (
            f"[{source.source_id}] filename={source.filename or 'unknown'} "
            f"document_id={source.document_id} chunk_index={source.chunk_index}\n"
            f"{source.content}"
        )
        for source in sources
    )
