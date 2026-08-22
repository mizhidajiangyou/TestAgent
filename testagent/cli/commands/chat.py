"""chat command: interactive conversational refinement of generated artifacts."""

from pathlib import Path
from typing import Any

import click
from rich.panel import Panel
from rich.prompt import Prompt

from testagent.cli.base import TestAgentCommand, console, main
from testagent.cli.examples import CHAT_EXAMPLES
from testagent.container import Container


@main.command(cls=TestAgentCommand, examples_text=CHAT_EXAMPLES)
@click.option(
    "--requirements",
    "-r",
    default=None,
    help="Path to requirements document (optional, for context)",
)
@click.option(
    "--swagger",
    "-s",
    default=None,
    help="Path or URL to Swagger/OpenAPI spec (optional, for context)",
)
@click.option(
    "--session",
    default=None,
    help="Session ID to resume (default: new auto-generated session)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=None,
    help="Max refine iterations per message (default: from config)",
)
@click.option(
    "--url",
    default=None,
    help="Target web application URL for GUI test generation during chat",
)
@click.pass_context
def chat(
    ctx: click.Context,
    requirements: str | None,
    swagger: str | None,
    session: str | None,
    max_iterations: int | None,
    url: str | None,
) -> None:
    """Interactive conversational refinement of generated artifacts.

    Supports continuous dialogue to generate, validate, and refine
    test cases, performance scripts, and GUI test scripts. Inspired
    by langgraph's StateGraph + checkpoint pattern: conversation state
    persists across turns within the same session.
    """
    container: Container = ctx.obj["container"]
    settings = container.settings()

    # Parse context if provided
    context: dict[str, Any] = {}
    if requirements:
        console.print("[bold blue]Parsing requirements...[/]")
        req_items = container.requirement_parser().parse(requirements)
        context["requirements"] = (
            container.requirement_parser().requirements_to_text(req_items)
            if hasattr(container.requirement_parser(), "requirements_to_text")
            else str(req_items)
        )
        console.print(f"  Found [green]{len(req_items)}[/] requirements")

    if swagger:
        console.print("[bold blue]Parsing Swagger spec...[/]")
        endpoints = container.swagger_parser().parse(swagger)
        from testagent.parsers.swagger_parser import SwaggerParser

        context["endpoints"] = SwaggerParser.endpoints_to_text(endpoints)
        console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    # Provide a target URL for in-chat GUI script generation. The conversation
    # engine routes "gui_script" artifacts through build_gui_test_prompt, which
    # requires a URL; it falls back to the generator default when absent.
    if url:
        context["gui_url"] = url
        console.print(f"  [dim]GUI target URL: {url}[/]")

    # Get or create conversation session
    manager = container.conversation_manager()
    existing = manager.get_session(session) if session else None
    if existing is not None:
        conv_session = existing
        console.print(f"[green]Resumed session: {session}[/]")
    else:
        conv_session = manager.create_session(session)
        console.print(f"[green]New session: {conv_session.session_id}[/]")

    console.print()
    console.print(
        Panel(
            "[bold]TestAgent Chat[/]\n"
            "Type natural language to generate, validate, or refine artifacts.\n"
            "Commands: [cyan]generate[/], [cyan]refine[/], [cyan]validate[/], "
            "[cyan]save[/], [cyan]history[/], [cyan]exit[/]",
            border_style="blue",
        )
    )
    console.print()

    # When the caller overrides max_iterations, apply it to this session so
    # the refine loop respects the per-turn ceiling.
    if max_iterations:
        conv_session.set_max_iterations(max(1, max_iterations))

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/]")
        except EOFError, KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/]")
            break

        if not user_input.strip():
            continue

        lower = user_input.strip().lower()
        if lower in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye![/]")
            break

        # Handle special commands
        if lower == "history":
            history = conv_session.get_history()
            for msg in history:
                color = "cyan" if msg.role == "user" else "green"
                console.print(f"[{color}]{msg.role}[/]: {msg.content[:200]}")
            continue

        if lower == "save":
            artifact = conv_session.get_latest_artifact()
            if artifact:
                output_path = Path(settings.output_dir) / f"artifact_{artifact.id}.txt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(artifact.content, encoding="utf-8")
                console.print(f"[green]Saved artifact v{artifact.version} to {output_path}[/]")
            else:
                console.print("[yellow]No artifact to save.[/]")
            continue

        # Send message to conversation session
        console.print("[bold blue]Assistant:[/]")
        try:
            response = conv_session.send(user_input, context=context)
            console.print(response)
        except Exception as e:
            console.print(f"[red]Error: {e}[/]")

        console.print()
        console.print(
            f"[dim]Iteration: {conv_session.iteration} | "
            f"Artifacts: {len(conv_session.get_artifacts())} | "
            f"Token usage: {container.llm_client().usage.summary()}[/]"
        )
        console.print()
