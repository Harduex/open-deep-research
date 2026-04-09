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
    show_verbose_response,
    show_verbose_step,
    show_verbose_thinking,
)
from open_deep_research.config import load_settings
from open_deep_research.core.evaluator import Evaluator
from open_deep_research.core.planner import Planner
from open_deep_research.core.reader import Reader
from open_deep_research.core.searcher import Searcher
from open_deep_research.core.synthesizer import Synthesizer, format_report_markdown
from open_deep_research.embeddings.dedup import FindingDeduplicator
from open_deep_research.llm.client import BudgetExhaustedError, LLMClient, VerboseEvent
from open_deep_research.models import IterationMetrics, SessionState, TokenBudget
from open_deep_research.providers import create_provider
from open_deep_research.state.session import SessionManager


async def _run_investigation_loop(
    state: SessionState,
    planner: Planner,
    searcher: Searcher,
    evaluator: Evaluator,
    synthesizer: Synthesizer,
    session_mgr: SessionManager,
    budget: TokenBudget,
    dedup: FindingDeduplicator | None = None,
    verbose: bool = False,
) -> str:
    """Run the investigation loop on a session state and return markdown report."""
    plan = state.plan
    show_plan(plan)

    # LOOP
    state.status = "investigating"
    try:
        while True:
            pending = [sq for sq in plan.sub_questions if sq.status == "pending"]
            if not pending:
                break

            sq = pending[0]
            sq.status = "investigating"  # type: ignore[assignment]
            show_iteration(plan.iteration, plan.max_iterations)
            console.print(f"  [bold]Investigating:[/] {sq.question}")

            if verbose:
                show_verbose_step("Searching", f"Generating queries and searching for sub-question: {sq.question}")

            new_sources, new_findings = await searcher.search_sub_question(sq, state.sources)
            state.sources.extend(new_sources)
            state.findings.extend(new_findings)
            sq.findings.extend(new_findings)

            for f in new_findings:
                show_finding(f.content, f.confidence)

            # Deduplicate findings
            deduped = 0
            if dedup:
                pre_count = len(state.findings)
                state.findings = dedup.deduplicate(state.findings)
                deduped = pre_count - len(state.findings)
                if deduped > 0:
                    console.print(f"  [dim]Dedup:[/] merged {deduped} similar findings")

            # Track iteration metrics
            state.iteration_metrics.append(IterationMetrics(
                iteration=plan.iteration,
                new_findings_count=len(new_findings),
                new_sources_count=len(new_sources),
                dedup_removed_count=deduped,
            ))

            sq.status = "answered" if new_findings else "unanswerable"  # type: ignore[assignment]

            if verbose:
                show_verbose_step("Plan Update", f"Reviewing progress and updating plan (iteration {plan.iteration})")

            plan = await planner.update_plan(plan, new_findings, state.sources)
            show_budget(budget)

            if verbose:
                show_verbose_step("Evaluation", "Checking if research has sufficient coverage to stop")

            evaluation = await evaluator.evaluate_stopping(
                plan, state.findings, state.sources, budget,
                iteration_metrics=state.iteration_metrics,
            )
            show_evaluation(evaluation)
            session_mgr.save(state)

            if evaluation.should_stop:
                break
    except BudgetExhaustedError:
        console.print("[yellow]Budget exhausted — moving to synthesis.[/]")

    # SYNTHESIZE
    show_synthesizing()
    if verbose:
        show_verbose_step("Synthesis", f"Generating report from {len(state.findings)} findings across {len(state.sources)} sources")

    state.status = "synthesizing"
    report = await synthesizer.synthesize(plan, state.findings, state.sources, budget)
    state.report = report
    state.status = "complete"
    session_mgr.save(state)

    markdown = format_report_markdown(report)
    show_complete(state.session_id)
    return markdown


def _make_verbose_callback() -> callable:
    """Create a callback that displays verbose LLM output."""
    def callback(event: VerboseEvent) -> None:
        show_verbose_thinking(event.thinking)
        show_verbose_response(event.raw_response, event.stage)
    return callback


def _build_components(settings):
    """Build all pipeline components from settings."""
    budget = TokenBudget(max_tokens=settings.research.budget_tokens)
    verbose_callback = _make_verbose_callback() if settings.output.verbose else None
    client = LLMClient(settings.llm, budget, verbose_callback=verbose_callback)
    provider = create_provider(settings.search)
    reader = Reader(client, settings.research.source_summary_tokens)
    planner = Planner(client)
    searcher = Searcher(
        provider, reader, client, settings.research.max_sources,
        follow_links=settings.research.follow_links,
        max_followed_links=settings.research.max_followed_links,
    )
    evaluator = Evaluator(client)
    synthesizer = Synthesizer(client, settings.output, settings.llm.model)
    session_mgr = SessionManager(settings.sessions.storage_dir)
    dedup = FindingDeduplicator()
    return client, provider, reader, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup


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
    if budget_override := settings_overrides.get("budget"):
        settings.research.budget_tokens = budget_override
    if settings_overrides.get("verbose") is not None:
        settings.output.verbose = settings_overrides["verbose"]

    client, provider, reader, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup = _build_components(settings)

    # PLAN
    console.print(f"\n[bold]Researching:[/] {query}\n")

    if settings.output.verbose:
        show_verbose_step("Planning", "Decomposing query into sub-questions")

    plan = await planner.create_plan(query, settings.research.max_iterations)
    state = session_mgr.create_session(plan, settings.research.budget_tokens)
    state.budget = budget

    return await _run_investigation_loop(state, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup, verbose=settings.output.verbose)


async def _run_resume(session_id: str, config_path: Path | None, verbose: bool = False) -> str:
    settings = load_settings(config_path)
    if verbose:
        settings.output.verbose = True
    client, provider, reader, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup = _build_components(settings)

    state = session_mgr.load(session_id)
    if not state:
        raise ValueError(f"Session {session_id} not found")

    if state.status == "complete":
        raise ValueError(f"Session {session_id} is already complete. Use 'follow-up' to refine.")

    # Restore budget state
    budget.used_tokens = state.budget.used_tokens
    state.budget = budget

    console.print(f"\n[bold]Resuming session:[/] {session_id}")
    console.print(f"[dim]Query:[/] {state.plan.query}\n")

    return await _run_investigation_loop(state, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup, verbose=settings.output.verbose)


async def _run_follow_up(session_id: str, follow_up_query: str, config_path: Path | None, verbose: bool = False) -> str:
    settings = load_settings(config_path)
    if verbose:
        settings.output.verbose = True
    client, provider, reader, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup = _build_components(settings)

    state = session_mgr.load(session_id)
    if not state:
        raise ValueError(f"Session {session_id} not found")

    if not state.report:
        raise ValueError(f"Session {session_id} has no report yet. Complete or resume it first.")

    # Restore budget and add more headroom
    budget.used_tokens = state.budget.used_tokens
    state.budget = budget

    console.print(f"\n[bold]Follow-up on session:[/] {session_id}")
    console.print(f"[dim]Original query:[/] {state.plan.query}")
    console.print(f"[bold]Follow-up:[/] {follow_up_query}\n")

    # Create new sub-questions for the follow-up
    plan = state.plan
    plan.iteration = 0  # Reset iteration counter
    plan.max_iterations = min(5, plan.max_iterations)  # Shorter for follow-ups

    # Use planner to generate new sub-questions for the follow-up
    follow_up_plan = await planner.create_plan(
        f"Follow-up on previous research about '{plan.query}': {follow_up_query}",
        plan.max_iterations,
    )

    # Append new sub-questions to existing plan
    next_id = len(plan.sub_questions)
    for sq in follow_up_plan.sub_questions:
        sq.id = f"sq_{next_id}"
        plan.sub_questions.append(sq)
        next_id += 1

    state.status = "investigating"
    state.report = None  # Will be regenerated

    return await _run_investigation_loop(state, planner, searcher, evaluator, synthesizer, session_mgr, budget, dedup, verbose=settings.output.verbose)


@app.command()
def research(
    query: str = typer.Argument(..., help="Research query"),
    model: str | None = typer.Option(None, "-m", "--model", help="LLM model override (e.g., ollama/llama3)"),
    max_sources: int | None = typer.Option(None, "-s", "--max-sources", help="Maximum sources to read"),
    max_iterations: int | None = typer.Option(None, "-i", "--max-iterations", help="Maximum research iterations"),
    budget: int | None = typer.Option(None, "-b", "--budget", help="Token budget"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show model thinking, raw responses, and step-by-step reasoning"),
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config.yaml"),
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
    if verbose:
        overrides["verbose"] = True

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
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config.yaml"),
) -> None:
    """List all research sessions."""
    settings = load_settings(config)
    mgr = SessionManager(settings.sessions.storage_dir)
    show_sessions(mgr.list_sessions())


@app.command()
def resume(
    session_id: str = typer.Argument(..., help="Session ID to resume"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show model thinking, raw responses, and step-by-step reasoning"),
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config.yaml"),
) -> None:
    """Resume an interrupted research session."""
    try:
        result = asyncio.run(_run_resume(session_id, config, verbose=verbose))
        console.print("\n")
        console.print(result)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted.[/]")
    except Exception as e:
        show_error(str(e))
        raise typer.Exit(1)


@app.command(name="follow-up")
def follow_up(
    session_id: str = typer.Argument(..., help="Session ID to follow up on"),
    query: str = typer.Argument(..., help="Follow-up question"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show model thinking, raw responses, and step-by-step reasoning"),
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config.yaml"),
) -> None:
    """Ask a follow-up question on a completed research session."""
    try:
        result = asyncio.run(_run_follow_up(session_id, query, config, verbose=verbose))
        console.print("\n")
        console.print(result)
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted.[/]")
    except Exception as e:
        show_error(str(e))
        raise typer.Exit(1)


@app.command()
def export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Output file path"),
    config: Path | None = typer.Option(None, "-c", "--config", help="Path to config.yaml"),
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
