from __future__ import annotations

import json
import re
from typing import TypeVar

import litellm
from pydantic import BaseModel, ValidationError

from open_deep_research.config import LLMConfig
from open_deep_research.models import TokenBudget

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


def _extract_json(text: str) -> str:
    # Strip thinking blocks (e.g. <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    # Find outermost braces
    start = text.find("{")
    if start == -1:
        # Try array
        start = text.find("[")
        if start == -1:
            return text
        end = text.rfind("]")
        return text[start : end + 1] if end > start else text

    end = text.rfind("}")
    return text[start : end + 1] if end > start else text


class LLMClient:
    def __init__(self, config: LLMConfig, budget: TokenBudget) -> None:
        self._model = config.model
        self._api_base = config.api_base
        self._api_key = config.api_key
        self._temperature = config.temperature
        self._budget = budget

    async def complete(self, prompt: str, response_model: type[T], system: str | None = None) -> T:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}"

        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0 and last_error:
                full_prompt += f"\n\nYour previous response had validation errors: {last_error}. Please fix and respond again."

            text = await self._call(full_prompt, system)
            json_str = _extract_json(text)

            try:
                return response_model.model_validate_json(json_str)
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e

        raise StructuredOutputError(f"Failed to parse structured output after 3 attempts: {last_error}")

    async def complete_text(self, prompt: str, system: str | None = None) -> str:
        return await self._call(prompt, system)

    async def _call(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key

        response = await litellm.acompletion(**kwargs)
        self._track_usage(response)
        return response.choices[0].message.content or ""

    def _track_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            total = getattr(usage, "total_tokens", 0)
            if not total:
                prompt_t = getattr(usage, "prompt_tokens", 0) or 0
                completion_t = getattr(usage, "completion_tokens", 0) or 0
                total = prompt_t + completion_t
            if total:
                self._budget.add(total)
