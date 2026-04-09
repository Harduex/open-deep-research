from open_deep_research.llm.client import BudgetExhaustedError, _extract_json
from open_deep_research.models import TokenBudget

import pytest


def test_extract_json_plain():
    assert _extract_json('{"key": "value"}') == '{"key": "value"}'


def test_extract_json_with_text():
    text = 'Here is the result: {"key": "value"} some trailing text'
    assert _extract_json(text) == '{"key": "value"}'


def test_extract_json_fenced():
    text = '```json\n{"key": "value"}\n```'
    assert _extract_json(text) == '{"key": "value"}'


def test_extract_json_thinking_block():
    text = '<think>I need to think about this carefully. Let me consider the options.</think>\n{"key": "value"}'
    assert _extract_json(text) == '{"key": "value"}'


def test_extract_json_thinking_and_fenced():
    text = '<think>reasoning here</think>\n```json\n{"queries": ["a", "b"]}\n```'
    result = _extract_json(text)
    assert '"queries"' in result


def test_extract_json_array():
    text = 'result: [1, 2, 3]'
    assert _extract_json(text) == '[1, 2, 3]'


def test_extract_json_nested():
    text = '{"outer": {"inner": 1}}'
    assert _extract_json(text) == '{"outer": {"inner": 1}}'


def test_budget_exhausted_raises():
    from open_deep_research.config import LLMConfig

    budget = TokenBudget(max_tokens=100, used_tokens=100)
    client = __import__("open_deep_research.llm.client", fromlist=["LLMClient"]).LLMClient(
        LLMConfig(), budget,
    )
    with pytest.raises(BudgetExhaustedError):
        import asyncio
        asyncio.run(client.complete_text("test prompt"))
