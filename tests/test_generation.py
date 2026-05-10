"""
Generation and prompt-construction tests.

These tests cover prompt text formatting, answer service orchestration, and
LiteLLM response/temperature handling. LiteLLM itself is patched, so no model API
is called during the suite.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from helpers import RecordingGenerationClient, sample_sources
from app.generation.prompt_builder import build_grounded_answer_prompt, build_no_context_prompt
from app.generation.providers import LiteLLMGenerationClient
from app.generation.service import AnswerGenerator, resolve_final_prompt


class TestGeneration(unittest.TestCase):
    """Prompt formatting, answer orchestration, and LiteLLM adapter tests."""

    def test_builds_grounded_and_no_context_prompts(self) -> None:
        """Prompt builders should include sources or clear no-context wording."""
        grounded_prompt = build_grounded_answer_prompt(
            query="What is discussed?",
            sources=sample_sources(),
        )
        no_context_prompt = build_no_context_prompt(query="What is discussed?")

        self.assertIn("Question:\nWhat is discussed?", grounded_prompt)
        self.assertIn("[S1] filename=notes.txt document_id=doc-1 chunk_index=0", grounded_prompt)
        self.assertIn("alpha beta gamma", grounded_prompt)
        self.assertIn("[S2] filename=notes.txt document_id=doc-1 chunk_index=1", grounded_prompt)
        self.assertIn("delta epsilon zeta", grounded_prompt)
        self.assertIn("No retrieved context was available for this question", no_context_prompt)
        self.assertIn("answer without citations", no_context_prompt.lower())

    def test_answer_generator_uses_no_context_raw_and_grounded_paths(self) -> None:
        """AnswerGenerator should route between no-context, raw, and grounded prompts."""
        client = RecordingGenerationClient()
        service = AnswerGenerator(generation_client=client)

        no_hit_result = service.generate_answer(
            query="What is discussed?",
            sources=[],
            include_context_in_prompt=True,
        )
        raw_prompt = resolve_final_prompt(
            query="What is discussed?",
            sources=sample_sources(),
            include_context_in_prompt=False,
        )
        grounded_result = service.generate_answer(
            query="What is discussed?",
            sources=sample_sources(),
            include_context_in_prompt=True,
        )

        self.assertEqual(no_hit_result.answer, "Grounded answer [S1]")
        self.assertEqual(no_hit_result.sources, [])
        self.assertIn("No retrieved context was available", client.calls[0][0])
        self.assertEqual(raw_prompt, "What is discussed?")
        self.assertEqual(grounded_result.answer, "Grounded answer [S1]")
        self.assertIn("alpha beta gamma", client.calls[1][0])
        self.assertEqual(client.calls[1][1], 0.1)

    def test_litellm_client_maps_chat_response_content(self) -> None:
        """LiteLLM client should extract message text from chat responses."""
        client = LiteLLMGenerationClient(model="openai/gpt-5.4-mini", api_key="test-key")
        mocked_response = {"choices": [{"message": {"content": "Synthesized answer [S1]"}}]}

        with patch("app.generation.providers.litellm_completion", return_value=mocked_response):
            answer = client.generate_text(prompt="Use the retrieved context.", temperature=0.1)

        self.assertEqual(answer, "Synthesized answer [S1]")

    def test_litellm_temperature_rules_for_model_families(self) -> None:
        """Temperature handling should follow current model-family restrictions."""
        cases = [
            ("openai/gpt-5.4-mini", 1.0),
            ("openai/gpt-4o-mini", 0.1),
            ("openai/gpt-5.1", 0.1),
        ]

        for model, expected_temperature in cases:
            with self.subTest(model=model):
                client = LiteLLMGenerationClient(model=model, api_key="test-key")
                mocked_response = {"choices": [{"message": {"content": "Answer"}}]}

                with patch("app.generation.providers.litellm_completion", return_value=mocked_response) as completion:
                    answer = client.generate_text(prompt="Use the retrieved context.", temperature=0.1)

                self.assertEqual(answer, "Answer")
                self.assertEqual(completion.call_args.kwargs["temperature"], expected_temperature)
