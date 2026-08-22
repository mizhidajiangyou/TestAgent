"""CLI base: shared console, help mixin, root group, session-record helper.

Importing this module defines the ``main`` click group. Subcommand modules
under :mod:`testagent.cli.commands` register themselves onto ``main`` via
``@main.command(...)``; :mod:`testagent.cli` (the package ``__init__``) imports
those modules to trigger registration.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from testagent.cli.examples import EXAMPLES_TEXT
from testagent.config.logging import setup_logging
from testagent.container import Container

#: Shared rich console used by all CLI commands for colored output.
console = Console()


def _save_session_record(
    session_id: str,
    swagger: str | None,
    requirements: str | None,
    output: str,
    settings: Any,
    resumed_from: str | None,
) -> None:
    """Persist a run's inputs + config so it can be resumed by ``--resume <id>``.

    Stored at ``output/sessions/<session_id>.json``. The record captures enough
    to re-run the exact same generation (input paths + model/Token config) after
    an interruption or a partial failure.
    """
    out_dir = Path("output/sessions")
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "session_id": session_id,
        "created_at": datetime.now(UTC).isoformat(),
        "resumed_from": resumed_from,
        "swagger": swagger,
        "requirements": requirements,
        "output": output,
        "model": settings.llm.models,
        "max_output_tokens": settings.llm.max_output_tokens,
        "max_concurrency": settings.llm.max_concurrency,
        "verify_model": settings.llm.verify_model,
        "stream": settings.llm.stream,
        "json_mode": settings.llm.json_mode,
        "review_enabled": settings.review_enabled,
    }
    (out_dir / f"{session_id}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class _ExamplesHelpMixin:
    """Mixin that appends a pre-formatted examples block after standard help.

    Click's built-in epilog wraps long lines and collapses single newlines,
    which mangles multi-line example blocks. We override ``format_help`` to
    let click write the standard help into the formatter, then append the
    examples block verbatim via ``formatter.write`` (which does NOT wrap),
    so line breaks are preserved and the block appears after the standard
    help (correct ordering).
    """

    examples_text: str = ""

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """Write standard help, then append examples block verbatim."""
        super().format_help(ctx, formatter)  # type: ignore[misc]
        if self.examples_text:
            # ``formatter.write`` appends raw text without re-wrapping,
            # preserving the line breaks in the examples block.
            formatter.write(self.examples_text)


class TestAgentGroup(_ExamplesHelpMixin, click.Group):
    """Root command group with appended examples block."""

    def __init__(self, *args: Any, examples_text: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.examples_text = examples_text


class TestAgentCommand(_ExamplesHelpMixin, click.Command):
    """Subcommand with appended examples block."""

    def __init__(self, *args: Any, examples_text: str = "", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.examples_text = examples_text


@click.group(cls=TestAgentGroup, examples_text=EXAMPLES_TEXT)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """TestAgent - AI-powered test case and performance script generator."""
    ctx.ensure_object(dict)
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)
    ctx.obj["container"] = Container()
