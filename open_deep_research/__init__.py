from __future__ import annotations

from pathlib import Path

from open_deep_research.config import Settings, load_settings


async def research(query: str, settings: Settings | None = None, config_path: Path | None = None, **overrides) -> str:
    """Run a research session and return the markdown report."""
    from open_deep_research.core.evaluator import Evaluator
    from open_deep_research.core.planner import Planner
    from open_deep_research.core.reader import Reader
    from open_deep_research.core.searcher import Searcher
    from open_deep_research.core.synthesizer import Synthesizer, format_report_markdown
    from open_deep_research.llm.client import BudgetExhaustedError, LLMClient, VerboseEvent
    from open_deep_research.models import TokenBudget
    from open_deep_research.providers import create_provider
    from open_deep_research.state.session import SessionManager

    if settings is None:
        settings = load_settings(config_path)

    if model := overrides.get("model"):
        settings.llm.model = model
    if max_sources := overrides.get("max_sources"):
        settings.research.max_sources = max_sources
    if max_iterations := overrides.get("max_iterations"):
        settings.research.max_iterations = max_iterations
    if overrides.get("verbose") is not None:
        settings.output.verbose = overrides["verbose"]

    budget_tokens = overrides.get("budget", settings.research.budget_tokens)
    budget = TokenBudget(max_tokens=budget_tokens)

    verbose_callback = overrides.get("verbose_callback")
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

    plan = await planner.create_plan(query, settings.research.max_iterations)
    state = session_mgr.create_session(plan, budget_tokens)
    state.budget = budget
    state.status = "investigating"

    try:
        while True:
            pending = [sq for sq in plan.sub_questions if sq.status == "pending"]
            if not pending:
                break

            sq = pending[0]
            sq.status = "investigating"  # type: ignore[assignment]

            new_sources, new_findings = await searcher.search_sub_question(sq, state.sources)
            state.sources.extend(new_sources)
            state.findings.extend(new_findings)
            sq.findings.extend(new_findings)
            sq.status = "answered" if new_findings else "unanswerable"  # type: ignore[assignment]

            plan = await planner.update_plan(plan, new_findings, state.sources)

            evaluation = await evaluator.evaluate_stopping(plan, state.findings, state.sources, budget)
            session_mgr.save(state)

            if evaluation.should_stop:
                break
    except BudgetExhaustedError:
        pass  # Fall through to synthesis

    state.status = "synthesizing"
    report = await synthesizer.synthesize(plan, state.findings, state.sources, budget)
    state.report = report
    state.status = "complete"
    session_mgr.save(state)

    return format_report_markdown(report)
