"""serve command: start the FastAPI web GUI."""

import os

import click
from rich.panel import Panel

from testagent.cli.base import TestAgentCommand, console, main
from testagent.cli.examples import SERVE_EXAMPLES
from testagent.container import Container


@main.command(cls=TestAgentCommand, examples_text=SERVE_EXAMPLES)
@click.option(
    "--host",
    default="0.0.0.0",
    help="Host to bind the web server to (default: 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=8000,
    help="Port to bind the web server to (default: 8000)",
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload (development only)",
)
@click.pass_context
def serve(
    ctx: click.Context,
    host: str,
    port: int,
    reload: bool,
) -> None:
    """Start the web GUI (FastAPI) for browser-based test case generation.

    The page is embeddable in other platforms via ``<iframe>``. By default
    any origin may embed it; restrict origins with the WEB_FRAME_ANCESTORS
    env var (space-separated).
    """
    try:
        import uvicorn
    except ImportError as exc:
        console.print(
            "[red]FastAPI/uvicorn are not installed. Install the web extra:[/]\n"
            '  [cyan]pip install -e ".[web]"[/]'
        )
        raise SystemExit(2) from exc

    from testagent.web.app import create_app

    container: Container = ctx.obj["container"]
    app = create_app(container)

    console.print(
        Panel(
            f"[bold]TestAgent Web GUI[/]\n"
            f"Open: [cyan]http://{host}:{port}/[/]\n"
            f'Embed: <iframe src="http://{host}:{port}/"></iframe>\n'
            f"frame-ancestors: [dim]{os.environ.get('WEB_FRAME_ANCESTORS', '*')}[/]",
            border_style="blue",
        )
    )
    uvicorn.run(app, host=host, port=port, reload=reload)
