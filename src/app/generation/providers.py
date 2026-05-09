"""
Author: Patrik Kiseda
File: src/app/generation/providers.py
Description: LiteLLM-backed generation provider for answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litellm import completion as litellm_completion

from app.core.settings import Settings
from app.generation.adapter import GenerationClient


@dataclass(slots=True)
class LiteLLMGenerationClient:
    """Simple GenerationClient implementation using LiteLLM for text generation."""

    model: str
    api_key: str | None

    def generate_text(self, *, prompt: str, temperature: float) -> str:
        """Run a single-prompt text completion through LiteLLM.

        Args:
            prompt: Prompt text to send.
            temperature: Sampling temperature.

        Returns:
            Generated text content.
        """
        response = litellm_completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=_resolve_temperature(model=self.model, temperature=temperature),
            api_key=self.api_key,
        )

        # Extract the generated text content from the LiteLLM response.
        content = _extract_message_text(response)
        if not content:
            raise RuntimeError("Generation API response did not include any text content.")
        return content


def build_generation_client(settings: Settings) -> GenerationClient:
    """Build default generation client for the app runtime.

    Args:
        settings: Runtime settings with model and API key.

    Returns:
        Configured generation client.
    """
    return LiteLLMGenerationClient(
        model=settings.litellm_model,
        api_key=settings.openai_api_key,
    )


def _resolve_temperature(*, model: str, temperature: float) -> float:
    """Resolve temperature, with GPT-5 variants forced to default.

    Args:
        model: Model name.
        temperature: Requested temperature.

    Returns:
        Temperature value safe for the provider.
    """
    if _requires_default_temperature(model):
        return 1.0
    return temperature

def _requires_default_temperature(model: str) -> bool:
    """Check if the model is a GPT-5 variant that needs default temperature.

    Args:
        model: Model name, maybe with provider prefix.

    Returns:
        True when the model should use default temperature.
    """
    providerless_model = model.split("/", maxsplit=1)[-1].lower()
    return providerless_model.startswith("gpt-5") and not providerless_model.startswith("gpt-5.1")


def _extract_message_text(response: Any) -> str:
    """Extract text content from a LiteLLM response.

    Args:
        response: LiteLLM completion response.

    Returns:
        Stripped generated text.
    """
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Generation API response did not contain a valid choices list.")

    message = _read_field(choices[0], "message")
    if message is None:
        raise RuntimeError("Generation API response did not include a message.")
    
    content = _read_field(message, "content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(_extract_content_part_text(part) for part in content).strip()

    raise RuntimeError("Generation API response did not include text content.")

def _extract_content_part_text(part: Any) -> str:
    """Extract text from one content part.

    Args:
        part: Message content part.

    Returns:
        Text value, or empty string for unsupported parts.
    """
    part_type = _read_field(part, "type")
    if part_type not in {None, "text"}:
        return ""

    text_value = _read_field(part, "text")
    if isinstance(text_value, str):
        return text_value
    return ""

def _read_field(item: Any, field_name: str) -> Any:
    """Read a field from dict-like or object-like items.

    Args:
        item: Item to inspect.
        field_name: Field name to read.

    Returns:
        Field value, or None when it is missing.
    """
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)
