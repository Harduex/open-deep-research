from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from open_deep_research.cli.app import app
from open_deep_research.cli.display import (
    console,
    show_budget,
    show_complete,
    show_error,
    show_evaluation,
    show_finding,
    show_iteration,
    show_plan,
    show_sessions,
    show_synthesizing,
)
from open_deep_research.config import load_settings
from open_deep_research.core.evaluator import Evaluator
from open_deep_research.core.planner import Planner
from open_deep_research.core.reader import Reader
from open_deep_research.core.searcher import Searcher
from open_deep_research.core.synthesizer import Synthesizer, format_report_markdown
from open_deep_research.llm.client import LLMClient
from open_deep_research.models import TokenBudget
from open_deep_research.providers import create_provider
from open_deep_research.state.session import SessionManager


async def _run_research(
    query: str,
    settings_overrides: dict,
    config_path: Path | None,
) -> str:
    settings = load_settings(config_path)

    # Apply CLI overrides
    if model := settings_overrides.get("model"):
        settings.llm.model = model
    if max_sources := settings_overrides.get("max_sources"):
        settings.research.max_sources = max_sources
    if max_iterations := settings_overrides.get("max_iterations"):
        settings.research.max_iterations = max_iterations

    budget_tokens = settings_overrides.get("budget", settings.research.budget_tokens)
    budget = TokenBudget(max_tokens=budget_tokens)

    # Initialize components
    client = LLMClient(settings.llm, budget)
    provider = create_provider(settings.search)
    reader = Reader(client, settings.research.source_summary_tokens)
    planner = Planner(client)
    searcher = Searcher(provider, reader, client, settings.research.max_sources)
    evaluator = Evaluator(client)
    synthesizer = Synthesizer(client, settings.output, settings.llm.model)
    session_mgr = SessionManager(settings.sessions.storage_dir)

    # PLAN
    console.print(f"\n[bold]Researching:[/] {query}\n")
    plan = await planner.create_plan(query, settings.research.max_iterations)
    state = session_mgr.create_session(plan, budget_tokens)
    state.budget = budget
    show_plan(plan)

    # LOOP
    state.status = "investigating"
    while True:
        # Find next pending sub-question
        pending = [sq for sq in plan.sub_questions if sq.status == "pending"]
        if not pending:
            break

        sq = pending[0]
        sq.status = "investigating"  # type: ignore[assignment]
        show_iteration(plan.iteration, plan.max_iterations)
        console.print(f"  [bold]Investigating:[/] {sq.question}")

        # Search and read
        new_sources, new_findings = await searcher.search_sub_question(sq, state.sources)
        state.sources.extend(new_sources)
        state.findings.extend(new_findings)
        sq.findings.extend(new_findings)

        for f in new_findings:
            show_finding(f.content, f.confidence)

        # Update sub-question status
        sq.status = "answered" if new_findings else "unanswerable"  # type: ignore[assignment]

        # Update plan
        plan = await planner.update_plan(plan, new_findings, state.sources)
        show_budget(budget)

        # Evaluate stopping
        evaluation = await evaluator.evaluate_stopping(plan, state.findings, state.sources, budget)
        show_evaluation(evaluation)

        # Checkpoint
        session_mgr.save(state)

        if evaluation.should_stop:
            break

    # SYNTHESIZE
    show_synthesizing()
    state.status = "synthesizing"
    report = await synthesizer.synthesize(plan, state.findings, state.sources, budget)
    state.report = report
    state.status = "complete"
    session_mgr.save(state)

    markdown = format_report_markdown(report)
    show_complete(state.session_id)
    return markdown


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query"),
    model: str | None = typer.Option(None, help="LLM model override (e.g., ollama/llama3)"),
    max_sources: int | None = typer.Option(None, help="Maximum sources to read"),
    max_iterations: int | None = typer.Option(None, help="Maximum research iterations"),
    budget: int | None = typer.Option(None, help="Token budget"),
    config: Path | None = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """Run a deep research investigation on a topic."""
    overrides = {}
    if model:
        overrides["model"] = model
    if max_sources:
        overrides["max_sources"] = max_sources
    if max_iterations:
        overrides["max_iterations"] = max_iterations
    if budget:
        overrides["budget"] = budget

    try:
        result = asyncio.run(_run_research(query, overrides, config))
        console.print("\n")
        console.print(result)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted.[/]")
    except Exception as e:
        show_error(str(e))
        raise typer.Exit(1)


@app.command()
def sessions(
    config: Path | None = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """List all research sessions."""
    settings = load_settings(config)
    mgr = SessionManager(settings.sessions.storage_dir)
    show_sessions(mgr.list_sessions())


@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    output: Path | None = typer.Option(None, help="Output file path"),
    config: Path | None = typer.Option(None, help="Path to config.yaml"),
) -> None:
    """Export a completed research session as markdown."""
    settings = load_settings(config)
    mgr = SessionManager(settings.sessions.storage_dir)
    state = mgr.load(session_id)

    if not state:
        show_error(f"Session {session_id} not found.")
        raise typer.Exit(1)

    if not state.report:
        show_error(f"Session {session_id} has no report (status: {state.status}).")
        raise typer.Exit(1)

    markdown = format_report_markdown(state.report)

    if output:
        output.write_text(markdown, encoding="utf-8")
        console.print(f"Report exported to {output}")
    else:
        console.print(markdown)
