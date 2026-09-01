"""Dynamic CLI command registration for task packages (plan-c B4.7).

Conflict policy (review #2): click 8.4.2 silently OVERWRITES same-named
commands, so the LEGACY commands always win a collision — a task (or alias)
whose name collides with a still-registered legacy command is skipped with a
warning until the legacy command is deleted in B7. Cross-task conflicts were
already rejected at Registry construction (B3.5); TASKS_DISABLE masks
individual packages for rollback.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

import click

from testagent.cli.base import TestAgentCommand

if TYPE_CHECKING:
    from click import Group

    from testagent.pipeline.registry import Registry, TaskPackage

logger = logging.getLogger(__name__)


def _to_click_option(spec: Any) -> click.Option:
    names = spec.option or [f"--{spec.name.replace('_', '-')}"]
    if spec.kind == "bool":
        return click.Option(
            [n for n in names],
            is_flag=True,
            default=None,
            help=spec.help,
        )
    if spec.kind == "choice":
        return click.Option(
            [n for n in names],
            type=click.Choice(spec.choices),
            default=None,
            help=spec.help,
        )
    if spec.kind == "int":
        return click.Option([n for n in names], type=int, default=None, help=spec.help)
    return click.Option([n for n in names], default=None, help=spec.help)


def build_click_command(
    task: TaskPackage,
    *,
    name_override: str | None = None,
    hidden: bool = False,
) -> click.Command:
    """Build one click command from a task package."""
    name = name_override or task.name
    manifest = task.manifest
    params: list[click.Parameter] = [_to_click_option(i) for i in manifest.inputs]
    formats = list(manifest.output.formats) or ["json"]
    params += [
        click.Option(
            ["--output", "-o"],
            default=manifest.output.default_path or f"./output/{name}",
            help="Output file path",
        ),
        click.Option(
            ["--format", "-f"],
            "output_format",
            type=click.Choice(formats),
            default=formats[0],
            help="Output format",
        ),
        click.Option(
            ["--session"],
            default=None,
            help="Reuse a session id (snapshot recovery key)",
        ),
    ]

    def callback(**kwargs: Any) -> None:
        ctx = click.get_current_context()
        _run_task(task, kwargs, ctx)

    return TestAgentCommand(
        name=name,
        params=params,
        callback=callback,
        help=manifest.description or manifest.display_name or name,
        examples_text=manifest.examples,
        hidden=hidden,
    )


def _run_task(task: TaskPackage, kwargs: Any, click_ctx: click.Context) -> None:
    """Execute one task package via the DI container's executor."""
    from testagent.cli.base import console
    from testagent.pipeline.inputs import parse_inputs
    from testagent.pipeline.writers import write_artifact

    container = click_ctx.obj.get("container")
    if container is None:
        from testagent.container import Container

        container = Container()
    settings = container.settings()
    output = kwargs.pop("output", None) or f"./output/{task.name}"
    output_format = kwargs.pop("output_format", None) or "json"
    session_id = kwargs.pop("session", None)

    ctx = parse_inputs(task.manifest, {k: v for k, v in kwargs.items()}, settings)
    executor = container.pipeline_executor()
    console.print(f"[cyan]Task[/] {task.name} — session {session_id or '(new)'}")
    result = asyncio.run(executor.arun(task, ctx, session_id=session_id))

    from pathlib import Path

    path = write_artifact(
        task.manifest,
        result.artifact,
        Path(output),
        output_format,
        ctx=ctx.parsed,
        review_meta=result.review_meta,
    )
    count = len(result.artifact) if isinstance(result.artifact, list) else 1
    console.print(
        f"[green]Done:[/] {count} item(s) -> {path} "
        f"(session {result.session_id}, failed units {result.units_failed})"
    )


def register_tasks(main: Group, registry: Registry) -> None:
    """Register task commands; LEGACY commands always win name conflicts.

    Underscore-prefixed packages (``_example`` etc.) register as HIDDEN
    click commands: runnable by explicit name, absent from ``--help``.
    """
    disabled = {n.strip() for n in os.getenv("TASKS_DISABLE", "").split(",") if n.strip()}
    for conflict in registry.conflicts:
        logger.warning("Registry conflict: %s", conflict)
    for task in registry.tasks(include_hidden=True):
        if task.name in disabled:
            logger.info("Task package '%s' disabled via TASKS_DISABLE.", task.name)
            continue
        if task.name in main.commands:
            logger.warning(
                "Task '%s' collides with a legacy command; skipping until the "
                "legacy command is removed (B7).",
                task.name,
            )
            continue
        main.add_command(build_click_command(task, hidden=task.name.startswith("_")))
        for alias in task.manifest.aliases:
            if alias in main.commands:
                logger.warning(
                    "Alias '%s' of task '%s' collides with a legacy command; "
                    "keeping the legacy command.",
                    alias,
                    task.name,
                )
                continue
            main.add_command(build_click_command(task, name_override=alias, hidden=True))
