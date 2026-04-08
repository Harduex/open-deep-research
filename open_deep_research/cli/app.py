import typer

app = typer.Typer(
    name="odr",
    help="Open Deep Research — Autonomous web research agent",
    no_args_is_help=True,
)


def main() -> None:
    import open_deep_research.cli.commands  # noqa: F401
    app()
