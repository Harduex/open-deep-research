import os
import sys

import typer

app = typer.Typer(
    name="odr",
    help="Open Deep Research — Autonomous web research agent",
    no_args_is_help=True,
)


def main() -> None:
    # Enable UTF-8 mode for the entire process (PEP 540).
    os.environ["PYTHONUTF8"] = "1"
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    import open_deep_research.cli.commands  # noqa: F401
    app()
