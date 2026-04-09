from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from open_deep_research.models import ResearchPlan, StoppingEvaluation, TokenBudget

console = Console()

STATUS_ICONS = {
    "pending": "[yellow]○[/]",
    "investigating": "[blue]◉[/]",
    "answered": "[green]✓[/]",
    "unanswerable": "[red]✗[/]",
}


def show_plan(plan: ResearchPlan) -> None:
    lines = []
    for sq in plan.sub_questions:
        icon = STATUS_ICONS.get(sq.status, "○")
        lines.append(f"  {icon} [{sq.id}] {sq.question}")
    console.print(Panel("\n".join(lines), title=f"Research Plan — {len(plan.sub_questions)} sub-questions", border_style="cyan"))


def show_iteration(iteration: int, max_iterations: int) -> None:
    console.print(f"\n[bold cyan]━━━ Iteration {iteration + 1}/{max_iterations} ━━━[/]")


def show_searching(query: str) -> None:
    console.print(f"  [dim]Searching:[/] {query}")


def show_reading(url: str) -> None:
    console.print(f"  [dim]Reading:[/] {url[:80]}")


def show_finding(content: str, confidence: str) -> None:
    color = {"high": "green", "medium": "yellow", "low": "red"}.get(confidence, "white")
    console.print(f"  [dim]Finding[/] [{color}]({confidence})[/]: {content[:120]}")


def show_evaluation(evaluation: StoppingEvaluation) -> None:
    bar_len = 20
    filled = int(evaluation.coverage_score * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    console.print(f"\n  Coverage: [{bar}] {evaluation.coverage_score:.0%}")
    console.print(f"  Saturation: {'yes' if evaluation.saturation_detected else 'no'}")
    decision = "[green]Continuing" if not evaluation.should_stop else "[yellow]Stopping"
    console.print(f"  Decision: {decision} — {evaluation.reasoning}[/]")


def show_synthesizing() -> None:
    console.print("\n[bold magenta]Synthesizing report...[/]")


def show_budget(budget: TokenBudget) -> None:
    pct = int(budget.used_tokens / budget.max_tokens * 100) if budget.max_tokens else 0
    console.print(f"  [dim]Budget:[/] {budget.used_tokens:,}/{budget.max_tokens:,} tokens ({pct}%)")


def show_error(msg: str) -> None:
    console.print(f"[bold red]Error:[/] {msg}")


def show_sessions(sessions: list[dict]) -> None:
    if not sessions:
        console.print("[dim]No sessions found.[/]")
        return

    table = Table(title="Research Sessions")
    table.add_column("ID", style="cyan")
    table.add_column("Query")
    table.add_column("Status")
    table.add_column("Sources", justify="right")
    table.add_column("Created")

    for s in sessions:
        query = s["query"][:50] + "..." if len(s["query"]) > 50 else s["query"]
        table.add_row(s["session_id"], query, s["status"], str(s["sources"]), s["created_at"][:16])

    console.print(table)


def show_complete(session_id: str) -> None:
    console.print(f"\n[bold green]Research complete.[/] Session: {session_id}")


def show_verbose_step(stage: str, description: str) -> None:
    console.print(f"\n  [bold dim]▶ {stage}:[/] [dim]{description}[/]")


def show_verbose_thinking(thinking: str) -> None:
    if thinking:
        console.print(Panel(thinking, title="[cyan]Model Thinking[/]", border_style="dim cyan", padding=(0, 1)))


def show_verbose_response(raw: str, stage: str) -> None:
    display = raw[:2000] + "..." if len(raw) > 2000 else raw
    console.print(Panel(display, title=f"[yellow]Raw Response — {stage}[/]", border_style="dim yellow", padding=(0, 1)))
