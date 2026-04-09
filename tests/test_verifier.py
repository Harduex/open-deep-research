import pytest

from open_deep_research.core.verifier import Verifier, VerificationResult
from open_deep_research.models import Finding, Source


class _FakeClient:
    """Stub LLMClient that returns predetermined verification results."""

    def __init__(self, verdicts: list[str]):
        self._verdicts = list(verdicts)
        self._call_count = 0

    async def complete(self, prompt, model_cls):
        verdict = self._verdicts[self._call_count] if self._call_count < len(self._verdicts) else "correct"
        self._call_count += 1
        return VerificationResult(verdict=verdict, feedback="test feedback")

    async def complete_text(self, prompt, system=None):
        return "revised text"


def _make_findings_sources():
    findings = [Finding(content="fact A", source_ids=[1], confidence="high")]
    sources = [Source(id=1, url="https://ex.com", title="Ex", snippet="snip")]
    return findings, sources


@pytest.mark.asyncio
async def test_verify_correct_returns_not_flawed():
    client = _FakeClient(["correct"])
    v = Verifier(client)
    findings, sources = _make_findings_sources()
    text, was_flawed = await v.verify_and_revise("draft", findings, sources)
    assert text == "draft"
    assert was_flawed is False


@pytest.mark.asyncio
async def test_verify_fundamentally_flawed_signals_caller():
    client = _FakeClient(["fundamentally_flawed"])
    v = Verifier(client)
    findings, sources = _make_findings_sources()
    text, was_flawed = await v.verify_and_revise("draft", findings, sources)
    assert was_flawed is True


@pytest.mark.asyncio
async def test_verify_minor_fixes_revises():
    client = _FakeClient(["minor_fixes", "correct"])
    v = Verifier(client)
    findings, sources = _make_findings_sources()
    text, was_flawed = await v.verify_and_revise("draft", findings, sources)
    assert text == "revised text"
    assert was_flawed is False
