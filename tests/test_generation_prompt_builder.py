"""
Author: Patrik Kiseda
File: tests/test_generation_prompt_builder.py
Description: Unit tests for grounded and no-context prompt construction.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.generation.prompt_builder import build_grounded_answer_prompt, build_no_context_prompt
from app.retrieval.service import RetrievedChunk


def _sample_sources() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            source_id="S1",
            chunk_id="doc-1:000000",
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=0,
            score=0.91,
            content="alpha beta gamma",
        ),
        RetrievedChunk(
            source_id="S2",
            chunk_id="doc-1:000001",
            document_id="doc-1",
            filename="notes.txt",
            chunk_index=1,
            score=0.87,
            content="delta epsilon zeta",
        ),
    ]


class TestGenerationPromptBuilder(unittest.TestCase):
    def test_builds_source_labeled_prompt_with_context(self) -> None:
        prompt = build_grounded_answer_prompt(
            query="What is discussed?",
            sources=_sample_sources(),
        )

        self.assertIn("Question:\nWhat is discussed?", prompt)
        self.assertIn("[S1] filename=notes.txt document_id=doc-1 chunk_index=0", prompt)
        self.assertIn("alpha beta gamma", prompt)
        self.assertIn("[S2] filename=notes.txt document_id=doc-1 chunk_index=1", prompt)
        self.assertIn("delta epsilon zeta", prompt)

    def test_builds_no_context_prompt_when_retrieval_returns_no_hits(self) -> None:
        prompt = build_no_context_prompt(query="What is discussed?")

        self.assertIn("No retrieved context was available for this question", prompt)
        self.assertIn("answer without citations", prompt.lower())
        self.assertIn("Question:\nWhat is discussed?", prompt)
