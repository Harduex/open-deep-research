from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import litellm
from pydantic import BaseModel, ValidationError
from rich.console import Console

# Suppress the noisy "Give Feedback / Get Help" banner on every exception
litellm.suppress_debug_info = True

from open_deep_research.config import LLMConfig
from open_deep_research.models import TokenBudget

T = TypeVar("T", bound=BaseModel)


@dataclass
class VerboseEvent:
    stage: str
    thinking: str
    raw_response: str
    prompt_summary: str


class StructuredOutputError(Exception):
    pass


class BudgetExhaustedError(Exception):
    pass


class LLMCallError(Exception):
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
    LLM_TIMEOUT = 120  # seconds

    def __init__(self, config: LLMConfig, budget: TokenBudget, verbose_callback: Callable[[VerboseEvent], None] | None = None) -> None:
        self._model = config.model
        self._api_base = config.api_base
        self._api_key = config.api_key
        self._temperature = config.temperature
        self._budget = budget
        self._verbose_callback = verbose_callback

    async def complete(self, prompt: str, response_model: type[T], system: str | None = None, stage: str = "") -> T:
        schema = json.dumps(response_model.model_json_schema(), indent=2)
        full_prompt = f"{prompt}\n\nRespond with valid JSON matching this schema:\n{schema}"

        last_error: Exception | None = None
        for attempt in range(3):
            if attempt > 0 and last_error:
                full_prompt += f"\n\nYour previous response had validation errors: {last_error}. Please fix and respond again."

            text = await self._call(full_prompt, system, stage=stage)
            json_str = _extract_json(text)

            try:
                return response_model.model_validate_json(json_str)
            except (ValidationError, json.JSONDecodeError) as e:
                last_error = e

        raise StructuredOutputError(f"Failed to parse structured output after 3 attempts: {last_error}")

    async def complete_text(self, prompt: str, system: str | None = None, stage: str = "") -> str:
        return await self._call(prompt, system, stage=stage)

    _STAGE_LABELS = {
        "planning": "Planning",
        "query_generation": "Generating queries",
        "plan_update": "Updating plan",
        "evaluation": "Evaluating",
        "synthesis": "Synthesizing",
        "verification": "Verifying",
    }
    # Stages that run concurrently — skip spinner to avoid console conflicts
    _BATCH_STAGES = {"finding_extraction", "summarization"}
    _console = Console()

    async def _call(self, prompt: str, system: str | None = None, stage: str = "") -> str:
        if self._budget.is_exceeded:
            raise BudgetExhaustedError("Token budget exhausted")

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

        use_spinner = stage not in self._BATCH_STAGES
        label = self._STAGE_LABELS.get(stage, stage or "Thinking")
        start = time.monotonic()

        try:
            if use_spinner:
                async def _update_status(status):
                    while True:
                        await asyncio.sleep(1)
                        elapsed = int(time.monotonic() - start)
                        status.update(f"  [bold cyan]{label}...[/] [dim]{elapsed}s[/]")

                with self._console.status(f"  [bold cyan]{label}...[/]", spinner="dots") as status:
                    updater = asyncio.create_task(_update_status(status))
                    try:
                        response = await asyncio.wait_for(
                            litellm.acompletion(**kwargs),
                            timeout=self.LLM_TIMEOUT,
                        )
                    finally:
                        updater.cancel()

                elapsed = int(time.monotonic() - start)
                self._console.print(f"  [dim]{label} done ({elapsed}s)[/]")
            else:
                response = await asyncio.wait_for(
                    litellm.acompletion(**kwargs),
                    timeout=self.LLM_TIMEOUT,
                )
        except asyncio.TimeoutError:
            elapsed = int(time.monotonic() - start)
            raise LLMCallError(f"LLM call timed out after {elapsed}s during '{label}' (model: {self._model})")
        except litellm.exceptions.APIConnectionError as e:
            raise LLMCallError(f"Cannot connect to LLM provider during '{label}': {e}")
        except litellm.exceptions.APIError as e:
            raise LLMCallError(f"LLM API error during '{label}': {e}")

        self._track_usage(response)
        raw = response.choices[0].message.content or ""

        if self._verbose_callback and stage:
            thinking = ""
            think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            if think_match:
                thinking = think_match.group(1).strip()
            self._verbose_callback(VerboseEvent(
                stage=stage,
                thinking=thinking,
                raw_response=raw,
                prompt_summary=prompt[:200],
            ))

        return raw

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
