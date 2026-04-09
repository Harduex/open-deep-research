import io
import os
import sys

import typer

app = typer.Typer(
    name="odr",
    help="Open Deep Research — Autonomous web research agent",
    no_args_is_help=True,
)


def _ensure_utf8_streams() -> None:
    """Replace stdout/stderr with UTF-8 wrappers so every downstream
    writer (Rich Console, print, etc.) inherits safe encoding."""
    os.environ["PYTHONUTF8"] = "1"
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "buffer"):
            new = io.TextIOWrapper(
                stream.buffer, encoding="utf-8", errors="replace",
                line_buffering=stream.line_buffering,
            )
            setattr(sys, name, new)


def main() -> None:
    _ensure_utf8_streams()
    import open_deep_research.cli.commands  # noqa: F401
    app()
