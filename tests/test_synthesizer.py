from open_deep_research.core.synthesizer import format_report_markdown
from open_deep_research.models import (
    Report,
    ReportMetadata,
    ReportSection,
    Source,
)


def test_format_report_markdown_basic():
    report = Report(
        title="Test Report",
        executive_summary="This is a summary.",
        sections=[
            ReportSection(title="Section 1", content="Content here [1].", source_ids=[1]),
        ],
        sources=[
            Source(id=1, url="https://example.com", title="Example", snippet="snip"),
        ],
    )
    md = format_report_markdown(report)
    assert "# Test Report" in md
    assert "## Executive Summary" in md
    assert "This is a summary." in md
    assert "## Section 1" in md
    assert "Content here [1]." in md
    assert "## Sources" in md
    assert "[Example](https://example.com)" in md


def test_format_report_markdown_with_contradictions():
    report = Report(
        title="Test",
        executive_summary="Sum",
        sections=[],
        contradictions=["Source 1 says X, Source 2 says Y"],
    )
    md = format_report_markdown(report)
    assert "## Contradictions" in md
    assert "Source 1 says X" in md


def test_format_report_markdown_no_contradictions():
    report = Report(
        title="Test",
        executive_summary="Sum",
        sections=[],
    )
    md = format_report_markdown(report)
    assert "Contradictions" not in md
