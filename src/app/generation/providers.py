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


# LiteLLMGenerationClient: simple GenerationClient implementation using LiteLLM for text generation.
@dataclass(slots=True)
class LiteLLMGenerationClient:
    model: str
    api_key: str | None

    # generate_text: run a single-prompt text completion through LiteLLM.
    def generate_text(self, *, prompt: str, temperature: float) -> str:
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


# build_generation_client: default provider wiring for the application runtime.
def build_generation_client(settings: Settings) -> GenerationClient:
    return LiteLLMGenerationClient(
        model=settings.litellm_model,
        api_key=settings.openai_api_key,
    )


# _resolve_temperature: GPT5.x <-Current provider rejects anything else than the default temperature through LiteLLM.
def _resolve_temperature(*, model: str, temperature: float) -> float:
    if _requires_default_temperature(model):
        return 1.0
    return temperature

# _requires_default_temperature: extra check if the model is a GPT5.x variant that requires the default temperature, based on the model name.
def _requires_default_temperature(model: str) -> bool:
    providerless_model = model.split("/", maxsplit=1)[-1].lower()
    return providerless_model.startswith("gpt-5") and not providerless_model.startswith("gpt-5.1")


# Helper functions to extract text content from the LiteLLM response, which may have different structures depending on the model and response format.
def _extract_message_text(response: Any) -> str:
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

# _extract_content_part_text: helper to extract text from a part of a message content.
def _extract_content_part_text(part: Any) -> str:
    part_type = _read_field(part, "type")
    if part_type not in {None, "text"}:
        return ""

    text_value = _read_field(part, "text")
    if isinstance(text_value, str):
        return text_value
    return ""

# _read_field: helper to read a field from an object that may be a dict or have attributes, None if not found.
def _read_field(item: Any, field_name: str) -> Any:
    if isinstance(item, dict):
        return item.get(field_name)
    return getattr(item, field_name, None)
