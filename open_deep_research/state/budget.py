from __future__ import annotations

from rich.console import Console

from open_deep_research.models import TokenBudget

console = Console(stderr=True)


def check_budget(budget: TokenBudget, required: int = 0) -> bool:
    if budget.is_exceeded:
        return False
    if required > 0 and budget.remaining < required:
        return False
    if budget.should_warn:
        pct = int(budget.used_tokens / budget.max_tokens * 100)
        console.print(f"[yellow]Budget warning: {pct}% used ({budget.used_tokens}/{budget.max_tokens} tokens)[/]")
    return True
