"""``testagent tasks`` — inspect and validate task packages (plan-c B4.4)."""

import click

from testagent.cli.base import console, main


@main.group()
def tasks() -> None:
    """Inspect and validate task packages."""


@tasks.command("list")
@click.option("--all", "include_hidden", is_flag=True, help="Include hidden (underscore) packages.")
@click.pass_context
def tasks_list(ctx: click.Context, include_hidden: bool) -> None:
    """List discovered task packages (hidden ones with --all)."""
    registry = ctx.obj["container"].task_registry()
    found = registry.tasks(include_hidden=include_hidden)
    if not found:
        console.print("[yellow]No task packages found (check TASKS_DIR).[/]")
        return
    for pkg in found:
        aliases = f" [aliases: {', '.join(pkg.manifest.aliases)}]" if pkg.manifest.aliases else ""
        console.print(
            f"[cyan]{pkg.name}[/]{aliases} — {pkg.manifest.description or pkg.manifest.display_name}"
        )


@tasks.command("validate")
@click.option("--strict", is_flag=True, help="Exit 1 on any problem (CI mode).")
@click.pass_context
def validate(ctx: click.Context, strict: bool) -> None:
    """Load every task package and report manifest/template problems."""
    container = ctx.obj["container"]
    registry = container.task_registry()
    exit_code = 0
    for conflict in registry.conflicts:
        console.print(f"[red]conflict: {conflict}[/]")
        exit_code = 1
    for pkg in registry.tasks(include_hidden=True):
        problems = pkg.validate_renderable() + pkg.validate_references()
        if problems:
            exit_code = 1
            console.print(f"[red]✗ {pkg.name}[/]")
            for p in problems:
                console.print(f"    {p}")
        else:
            console.print(f"[green]✓ {pkg.name}[/]")
    if exit_code and not strict:
        # Non-strict mode: problems are reported but do not fail the run.
        exit_code = 0
    # SystemExit (not ctx.exit): click 8.4 does not propagate ctx.exit's
    # code through CliRunner's standalone_mode=False path.
    raise SystemExit(exit_code)
