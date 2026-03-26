"""
Author: Patrik Kiseda
File: src/app/generation/adapter.py
Description: Generation adapter contract for answer generation providers.
"""

from __future__ import annotations

from typing import Protocol


# GenerationClient: minimal text generation contract used by the answer service.
# TODO - this can be extended later with more parameters for things like few-shot examples, system instructions, etc.
class GenerationClient(Protocol):
    model: str

    def generate_text(self, *, prompt: str, temperature: float) -> str:
        ...
