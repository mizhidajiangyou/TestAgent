"""generate-tests command: produce test cases from requirements and/or API spec."""

import asyncio
import json
import uuid
from pathlib import Path

import click

from testagent.cli.base import TestAgentCommand, _save_session_record, console, main
from testagent.cli.examples import GENERATE_TESTS_EXAMPLES
from testagent.config.models import TestCaseGenInput, TestCaseReportInput
from testagent.container import Container
from testagent.engine.llm_client import ModelUnavailableError
from testagent.generators.testcase_generator import TestCaseGenerator


@main.command(cls=TestAgentCommand, examples_text=GENERATE_TESTS_EXAMPLES)
@click.option(
    "--swagger",
    "-s",
    default=None,
    help="Path or URL to Swagger/OpenAPI spec (optional)",
)
@click.option(
    "--requirements",
    "-r",
    default=None,
    help="Path to requirements document",
)
@click.option(
    "--output",
    "-o",
    default="./output/testcases.json",
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    "output_format",
    type=click.Choice(["json", "csv", "markdown"]),
    default="json",
    help="Output format",
)
@click.option(
    "--historical-cases",
    "-H",
    default=None,
    help="Path to historical test cases JSON file (baseline for incremental generation)",
)
@click.option(
    "--resume",
    default=None,
    help="Resume a previous run by its session id (re-runs with the same inputs/config).",
)
@click.pass_context
def generate_tests(
    ctx: click.Context,
    swagger: str | None,
    requirements: str | None,
    output: str,
    output_format: str,
    historical_cases: str | None,
    resume: str | None,
) -> None:
    """Generate test cases from requirements and/or API spec."""
    container: Container = ctx.obj["container"]

    # --- Session id + optional resume ---------------------------------
    session_id = uuid.uuid4().hex[:12]
    resumed_from: str | None = None
    if resume:
        record_path = Path("output/sessions") / f"{resume}.json"
        if not record_path.exists():
            console.print(f"[red]Resume session not found:[/] {record_path}")
            ctx.exit(1)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        resumed_from = record.get("session_id")
        session_id = resumed_from or session_id
        # Re-run with the exact same inputs that produced the original record.
        swagger = record.get("swagger") or swagger
        requirements = record.get("requirements") or requirements
        console.print(f"[cyan]Resuming session {session_id}[/] from record {record_path}")

    if not swagger and not requirements:
        console.print("[red]Error: at least one of --swagger or --requirements is required.[/]")
        ctx.exit(1)

    endpoints = []
    if swagger:
        console.print("[bold blue]Parsing Swagger spec...[/]")
        endpoints = container.swagger_parser().parse(swagger)
        console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    req_items = []
    if requirements:
        console.print("[bold blue]Parsing requirements...[/]")
        req_items = container.requirement_parser().parse(requirements)
        console.print(f"  Found [green]{len(req_items)}[/] requirements")

    historical = []
    if historical_cases:
        historical = TestCaseGenerator.load_historical_cases(historical_cases)
        console.print(
            f"  Loaded [green]{len(historical)}[/] historical cases as baseline "
            f"from [cyan]{historical_cases}[/]"
        )

    console.print(f"[bold]Session:[/] [cyan]{session_id}[/]")
    console.print("[bold blue]Generating test cases via LLM...[/]")
    try:
        test_cases = asyncio.run(
            container.testcase_generator().agenerate(
                TestCaseGenInput(
                    endpoints=endpoints,
                    requirements=req_items,
                    historical_cases=historical,
                ),
                session_id=session_id,
            )
        )
    except ModelUnavailableError as exc:
        # Zero-token pre-flight failed: bad key / base_url / model name.
        console.print(f"[red]Model unavailable:[/] {exc}")
        console.print(
            "[dim]Check OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_MODEL "
            "(or set OPENAI_VERIFY_MODEL=false if your provider lacks the "
            "/models API).[/]"
        )
        ctx.exit(1)

    # Persist a session record so this exact run can be resumed by id later.
    _save_session_record(
        session_id, swagger, requirements, output, container.settings(), resumed_from
    )

    if not test_cases:
        console.print(
            "[red]Generation produced 0 test cases.[/] The LLM calls failed or "
            "returned no usable content — most often because responses were "
            "truncated beyond the model's output limit. Try lowering "
            "OPENAI_MAX_OUTPUT_TOKENS to the model's real output cap, or reduce "
            "the requested scope (fewer / more compact test cases)."
        )
        ctx.exit(1)
    if historical:
        console.print(
            f"  Merged: [green]{len(historical)}[/] historical + net-new = "
            f"[green]{len(test_cases)}[/] total"
        )
    else:
        console.print(f"  Generated [green]{len(test_cases)}[/] test cases")

    output_path = Path(output)
    if output_format == "markdown":
        report = container.testcase_report().generate(
            TestCaseReportInput(
                test_cases=test_cases,
                output_format="markdown",
                output_language=container.settings().output_language,
            )
        )
        output_path = output_path.with_suffix(".md")
        container.testcase_report().save(report, output_path)
    elif output_format == "csv":
        output_path = output_path.with_suffix(".csv")
        container.testcase_generator().save_csv(test_cases, output_path)
    else:
        container.testcase_generator().save(test_cases, output_path)

    console.print(f"  Token usage: [dim]{container.llm_client().usage.summary()}[/]")
    console.print(f"[bold green]Done![/] Output saved to [cyan]{output_path}[/]")
