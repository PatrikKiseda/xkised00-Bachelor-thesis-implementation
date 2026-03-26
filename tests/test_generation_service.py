"""
Author: Patrik Kiseda
File: tests/test_generation_service.py
Description: Unit tests for generation orchestration and LiteLLM response mapping.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.generation.providers import LiteLLMGenerationClient
from app.generation.service import AnswerGenerator, resolve_final_prompt
from app.retrieval.service import RetrievedChunk


class _RecordingGenerationClient:
    model = "openai/gpt-5.4-mini"

    def __init__(self, response_text: str = "Grounded answer [S1]") -> None:
        self.response_text = response_text
        self.calls: list[tuple[str, float]] = []

    def generate_text(self, *, prompt: str, temperature: float) -> str:
        self.calls.append((prompt, temperature))
        return self.response_text


def _sample_sources() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            source_id="S1",
            chunk_id="doc-1:000000",
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=0,
            score=0.93,
            content="alpha beta gamma",
        )
    ]


class TestGenerationService(unittest.TestCase):
    def test_no_hit_path_still_calls_generation_client_with_no_context_prompt(self) -> None:
        client = _RecordingGenerationClient()
        service = AnswerGenerator(generation_client=client)

        result = service.generate_answer(
            query="What is discussed?",
            sources=[],
            include_context_in_prompt=True,
        )

        self.assertEqual(result.answer, "Grounded answer [S1]")
        self.assertEqual(result.sources, [])
        self.assertEqual(len(client.calls), 1)
        self.assertIn("No retrieved context was available", client.calls[0][0])

    def test_resolve_final_prompt_returns_raw_query_when_context_is_disabled(self) -> None:
        prompt = resolve_final_prompt(
            query="What is discussed?",
            sources=_sample_sources(),
            include_context_in_prompt=False,
        )

        self.assertEqual(prompt, "What is discussed?")

    def test_litellm_client_maps_chat_response_content(self) -> None:
        client = LiteLLMGenerationClient(model="openai/gpt-5.4-mini", api_key="test-key")

        mocked_response = {
            "choices": [
                {
                    "message": {
                        "content": "Synthesized answer [S1]",
                    }
                }
            ]
        }

        with patch("app.generation.providers.litellm_completion", return_value=mocked_response):
            answer = client.generate_text(
                prompt="Use the retrieved context.",
                temperature=0.1,
            )

        self.assertEqual(answer, "Synthesized answer [S1]")

    def test_service_uses_generation_client_when_sources_exist(self) -> None:
        client = _RecordingGenerationClient()
        service = AnswerGenerator(generation_client=client)

        result = service.generate_answer(
            query="What is discussed?",
            sources=_sample_sources(),
            include_context_in_prompt=True,
        )

        self.assertEqual(result.answer, "Grounded answer [S1]")
        self.assertEqual(len(client.calls), 1)
        self.assertIn("alpha beta gamma", client.calls[0][0])
        self.assertEqual(client.calls[0][1], 0.1)
