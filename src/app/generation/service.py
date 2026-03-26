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
    answer: str
    prompt: str
    sources: list[RetrievedChunk]


@dataclass(slots=True)
class AnswerGenerator:
    generation_client: GenerationClient
    temperature: float = DEFAULT_GENERATION_TEMPERATURE

    # generate_answer: select RAG, no-context, or raw-input mode and call the LLM.
    def generate_answer(
        self,
        *,
        query: str,
        sources: list[RetrievedChunk],
        include_context_in_prompt: bool,
    ) -> AnswerGenerationResult:
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


# resolve_final_prompt: choose the final prompt text from grounded, no-context, or raw-input modes.
def resolve_final_prompt(
    *,
    query: str,
    sources: list[RetrievedChunk],
    include_context_in_prompt: bool,
) -> str:
    return _resolve_final_prompt(
        query=query,
        sources=sources,
        include_context_in_prompt=include_context_in_prompt,
    )

# _resolve_final_prompt: internal helper to determine the final prompt text based on if there are retrieved sources
# and whether the context should be included in the prompt.
def _resolve_final_prompt(
    *,
    query: str,
    sources: list[RetrievedChunk],
    include_context_in_prompt: bool,
) -> str:
    if not include_context_in_prompt:
        return query
    if not sources:
        return build_no_context_prompt(query=query)
    return build_grounded_answer_prompt(query=query, sources=sources)
