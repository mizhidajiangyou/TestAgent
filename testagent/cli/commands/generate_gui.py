"""generate-gui command: produce Playwright Python GUI test scripts."""

from pathlib import Path

import click

from testagent.cli.base import TestAgentCommand, console, main
from testagent.cli.examples import GENERATE_GUI_EXAMPLES
from testagent.config.models import GUITestGenInput
from testagent.container import Container


@main.command(cls=TestAgentCommand, examples_text=GENERATE_GUI_EXAMPLES)
@click.option(
    "--requirements",
    "-r",
    required=True,
    help="Path to requirements document (supports .md, .txt, .json, .pdf, .docx)",
)
@click.option(
    "--swagger",
    "-s",
    default=None,
    help="Path or URL to Swagger/OpenAPI spec (optional, for API context)",
)
@click.option(
    "--url",
    default=None,
    help="Target web application URL for GUI testing",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path (default: ./output/gui_test.py)",
)
@click.option(
    "--review/--no-review",
    "review",
    default=None,
    help="Override REVIEW_ENABLED: cross-validate the generated script",
)
@click.pass_context
def generate_gui(
    ctx: click.Context,
    requirements: str,
    swagger: str | None,
    url: str | None,
    output: str | None,
    review: bool | None,
) -> None:
    """Generate Playwright Python test script for web/GUI testing."""
    container: Container = ctx.obj["container"]
    settings = container.settings()

    console.print("[bold blue]Parsing requirements...[/]")
    req_items = container.requirement_parser().parse(requirements)
    console.print(f"  Found [green]{len(req_items)}[/] requirements")

    endpoints = []
    if swagger:
        console.print("[bold blue]Parsing Swagger spec...[/]")
        endpoints = container.swagger_parser().parse(swagger)
        console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    if not url:
        url = settings.perf.base_url
        console.print(f"  [dim]Using default base URL: {url}[/]")

    console.print("[bold blue]Generating Playwright GUI test script via LLM...[/]")
    generator = container.gui_generator()
    if review is not None:
        generator.set_review_enabled(review)
    script = generator.generate(
        GUITestGenInput(
            requirements=req_items,
            url=url,
            endpoints=endpoints,
            output_language=settings.output_language,
        )
    )

    output_path = Path(output) if output else Path(settings.output_dir) / "gui_test.py"
    generator.save(script, output_path)
    console.print(f"[bold green]Done![/] Script saved to [cyan]{output_path}[/]")
    console.print(f"  Token usage: [dim]{container.llm_client().usage.summary()}[/]")
    console.print(f"  [dim]Run with: pytest {output_path} --browser chromium[/]")
