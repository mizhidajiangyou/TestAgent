"""config command: show the current effective configuration."""

import click

from testagent.cli.base import console, main
from testagent.container import Container


@main.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show current configuration."""
    container: Container = ctx.obj["container"]
    settings = container.settings()

    console.print("[bold]Current Configuration:[/]")
    console.print()
    console.print("[bold]LLM Settings:[/]")
    console.print(f"  Provider: {'Azure OpenAI' if settings.azure_llm.enabled else 'OpenAI'}")
    if settings.azure_llm.enabled:
        console.print(f"  Deployment: {settings.azure_llm.deployment}")
        console.print(f"  Endpoint: {settings.azure_llm.endpoint}")
    else:
        models = settings.llm.models
        primary = models[0] if models else "(none)"
        fallbacks = models[1:] if len(models) > 1 else []
        console.print(f"  Primary model: {primary}")
        if fallbacks:
            console.print(f"  Fallback models (in order): {', '.join(fallbacks)}")
        else:
            console.print("  Fallback models: (none configured)")
        console.print(f"  Base URL: {settings.llm.base_url}")
    console.print()
    console.print("[bold]Performance Settings:[/]")
    console.print(f"  Base URL: {settings.perf.base_url}")
    console.print(f"  Virtual Users: {settings.perf.virtual_users}")
    console.print(f"  Duration: {settings.perf.duration_seconds}s")
    console.print(f"  Ramp-Up: {settings.perf.ramp_up_seconds}s")
    console.print(f"  Think Time: {settings.perf.think_time_ms}ms")
    console.print()
    console.print("[bold]Output Settings:[/]")
    console.print(f"  Output Dir: {settings.output_dir}")
    console.print(f"  Script Format: {settings.script_format}")
    console.print()
    console.print("[bold]Pipeline Settings:[/]")
    console.print(f"  Review Enabled: {'yes' if settings.review_enabled else 'no'}")
    console.print(f"  Review Max Rounds: {settings.review_max_rounds}")
    console.print(f"  Output Language: {settings.output_language}")
