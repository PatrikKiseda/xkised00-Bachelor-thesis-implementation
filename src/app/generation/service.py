"""
Author: Patrik Kiseda
File: src/app/generation/service.py
Description: Orchestration for grounded answer generation from retrieved chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.adapter import GenerationClient
from app.generation.prompt_builder import build_grounded_answer_prompt, build_no_context_prompt
from app.retrieval.service import RetrievedChunk

DEFAULT_GENERATION_TEMPERATURE = 0.1


@dataclass(slots=True)
class AnswerGenerationResult:
    """Generated answer plus prompt and sources used for it."""

    answer: str
    prompt: str
    sources: list[RetrievedChunk]


@dataclass(slots=True)
class AnswerGenerator:
    """Small service that turns retrieved chunks into an answer."""

    generation_client: GenerationClient
    temperature: float = DEFAULT_GENERATION_TEMPERATURE

    def generate_answer(
        self,
        *,
        query: str,
        sources: list[RetrievedChunk],
        include_context_in_prompt: bool,
    ) -> AnswerGenerationResult:
        """Select RAG, no-context, or raw-input mode and call the LLM.

        Args:
            query: User question.
            sources: Retrieved chunks to ground on.
            include_context_in_prompt: Whether to include retrieved context.

        Returns:
            Answer result with final prompt and sources.
        """
        prompt = _resolve_final_prompt(
            query=query,
            sources=sources,
            include_context_in_prompt=include_context_in_prompt,
        )

        answer = self.generation_client.generate_text(
            prompt=prompt,
            temperature=self.temperature,
        ).strip()

        return AnswerGenerationResult(answer=answer, prompt=prompt, sources=sources)


def resolve_final_prompt(
    *,
    query: str,
    sources: list[RetrievedChunk],
    include_context_in_prompt: bool,
) -> str:
    """Choose final prompt text from grounded, no-context, or raw-input modes.

    Args:
        query: User question.
        sources: Retrieved chunks.
        include_context_in_prompt: Whether context should be included.

    Returns:
        Final prompt text.
    """
    return _resolve_final_prompt(
        query=query,
        sources=sources,
        include_context_in_prompt=include_context_in_prompt,
    )

def _resolve_final_prompt(
    *,
    query: str,
    sources: list[RetrievedChunk],
    include_context_in_prompt: bool,
) -> str:
    """Pick final prompt based on sources and context flag.

    Args:
        query: User question.
        sources: Retrieved chunks.
        include_context_in_prompt: Whether context should be included.

    Returns:
        Final prompt text.
    """
    if not include_context_in_prompt:
        return query
    if not sources:
        return build_no_context_prompt(query=query)
    return build_grounded_answer_prompt(query=query, sources=sources)
