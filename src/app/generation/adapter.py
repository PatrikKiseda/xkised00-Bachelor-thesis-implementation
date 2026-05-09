"""
Author: Patrik Kiseda
File: src/app/generation/adapter.py
Description: Generation adapter contract for answer generation providers.
"""

from __future__ import annotations

from typing import Protocol


class GenerationClient(Protocol):
    """Minimal text generation contract used by the answer service.

    TODO - this can be extended later with more parameters for things like few-shot examples,
    system instructions, etc.
    """

    model: str

    def generate_text(self, *, prompt: str, temperature: float) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Final prompt text to send to the model.
            temperature: Sampling temperature.

        Returns:
            Generated text content.
        """
        ...
