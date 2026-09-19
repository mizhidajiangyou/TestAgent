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
@click.option(
    "--with-testcases",
    "with_testcases",
    default=None,
    help="Import existing test cases as supplementary GUI reference (strict validation; default off).",
)
@click.pass_context
def generate_gui(
    ctx: click.Context,
    requirements: str,
    swagger: str | None,
    url: str | None,
    output: str | None,
    review: bool | None,
    with_testcases: str | None,
) -> None:
    """Generate Playwright Python test script for web/GUI testing."""
    container: Container = ctx.obj["container"]
    settings = container.settings()

    console.print("[bold blue]Parsing requirements...[/]")
    req_items = container.requirement_parser().parse(requirements)
    console.print(f"  Found [green]{len(req_items)}[/] requirements")

    selection = None
    gui_input = None
    if with_testcases:
        from testagent.orchestration.gui_with_cases import build_gui_input_with_cases

        console.print(f"[bold blue]Importing case references from {with_testcases}...[/]")
        endpoints = []
        if swagger:
            endpoints = container.swagger_parser().parse(swagger)
        gui_input, selection = build_gui_input_with_cases(
            requirements=req_items,
            reference_path=with_testcases,
            url=url,
            endpoints=endpoints,
            output_language=settings.output_language,
            generator=container.gui_generator(),
        )
        console.print(
            f"  Reference: selected [green]{selection.selected_count}[/]/"
            f"{selection.original_count} cases "
            f"({len(selection.omitted)} omitted: "
            f"{ {reason for _, reason in selection.omitted} })"
        )

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
    if gui_input is not None:
        script = generator.generate(gui_input)
    else:
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
