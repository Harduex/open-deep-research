from __future__ import annotations

import io
import os
import sys

from rich.console import Console

from open_deep_research.models import TokenBudget


def _make_stderr_console() -> Console:
    if os.name == "nt" and hasattr(sys.stderr, "encoding"):
        encoding = sys.stderr.encoding or "utf-8"
        if encoding.lower().replace("-", "") not in ("utf8", "utf16"):
            wrapped = io.TextIOWrapper(
                sys.stderr.buffer, encoding=encoding, errors="replace", line_buffering=True,
            )
            return Console(file=wrapped)
    return Console(stderr=True)


console = _make_stderr_console()


def check_budget(budget: TokenBudget, required: int = 0) -> bool:
    if budget.is_exceeded:
        return False
    if required > 0 and budget.remaining < required:
        return False
    if budget.should_warn:
        pct = int(budget.used_tokens / budget.max_tokens * 100)
        console.print(f"[yellow]Budget warning: {pct}% used ({budget.used_tokens}/{budget.max_tokens} tokens)[/]")
    return True
