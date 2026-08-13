"""
CLI entry point for TestAgent.

Provides commands for generating test cases and performance scripts.
"""

from pathlib import Path
from typing import Any

import click
from rich.console import Console

from testagent.config.logging import setup_logging
from testagent.config.models import (
    PerfGenInput,
    PerformanceConfig,
    PerfReportInput,
    TestCaseGenInput,
    TestCaseReportInput,
)
from testagent.container import Container

console = Console()


EXAMPLES_TEXT = """
EXAMPLES:
  # Show current configuration
  testagent config

  # Generate test cases (JSON, default)
  testagent generate-tests -s swagger.json -r requirements.md -o ./output/testcases.json

  # Generate Markdown report
  testagent generate-tests -s swagger.json -r requirements.md -f markdown -o ./output/testcases.md

  # Generate CSV (Excel-friendly, UTF-8 BOM)
  testagent generate-tests -s swagger.json -r requirements.md -f csv -o ./output/testcases.csv

  # Parse Swagger from URL
  testagent generate-tests -s https://petstore3.swagger.io/api/v3/openapi.json

  # Enable multi-model fallback + cross-validation review
  OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true REVIEW_MAX_ROUNDS=2 \\
      testagent generate-tests -s swagger.json -r requirements.md -f markdown

  # Generate k6 performance script (default format)
  testagent generate-perf -s swagger.json --base-url https://api.example.com

  # Generate JMeter JMX script
  testagent generate-perf -s swagger.json -f jmeter -o ./output/perf.jmx

  # Run generated scripts
  k6 run output/perf_test.js
  jmeter -n -t output/perf_test.jmx -l results.jtl

ENVIRONMENT:
  All settings can be overridden via env vars or .env file (see .env.example).
  Priority: env vars > .env > defaults.

  OPENAI_MODEL          comma-separated model list, first is primary (default: gpt-4o-mini)
  OPENAI_API_KEY        OpenAI API key
  OPENAI_BASE_URL       OpenAI-compatible endpoint (default: https://api.openai.com/v1)
  REVIEW_ENABLED        run cross-validation review after generation (default: false)
  REVIEW_MAX_ROUNDS     review rounds, odd=secondary model, even=primary (default: 2)
  OUTPUT_LANGUAGE       chinese | english (default: chinese)
  OUTPUT_DIR            output directory (default: ./output)
  SCRIPT_FORMAT         k6 | jmeter (default: k6)
"""

GENERATE_TESTS_EXAMPLES = """
EXAMPLES:
  # JSON output (default)
  testagent generate-tests -s swagger.json -r requirements.md

  # Markdown report
  testagent generate-tests -s swagger.json -r requirements.md -f markdown -o report.md

  # CSV for Excel
  testagent generate-tests -s swagger.json -f csv -o cases.csv

  # Swagger from URL
  testagent generate-tests -s https://petstore3.swagger.io/api/v3/openapi.json

  # With multi-model cross-validation review
  OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true \\
      testagent generate-tests -s swagger.json -r requirements.md

  # Requirements only (no Swagger)
  testagent generate-tests -r requirements.md -o cases.json
"""

GENERATE_PERF_EXAMPLES = """
EXAMPLES:
  # k6 script (default format from config)
  testagent generate-perf -s swagger.json --base-url https://api.example.com

  # JMeter JMX script
  testagent generate-perf -s swagger.json -f jmeter -o ./output/perf.jmx

  # Custom load profile
  testagent generate-perf -s swagger.json --virtual-users 50 --duration 120 --base-url https://api.example.com

  # Run generated scripts
  k6 run output/perf_test.js
  jmeter -n -t output/perf_test.jmx -l results.jtl
"""


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
@click.pass_context
def generate_tests(
    ctx: click.Context,
    swagger: str | None,
    requirements: str | None,
    output: str,
    output_format: str,
) -> None:
    """Generate test cases from requirements and/or API spec."""
    container: Container = ctx.obj["container"]

    if not swagger and not requirements:
        console.print("[red]Error: at least one of --swagger or --requirements is required.[/]")
        ctx.exit(1)

    endpoints = []
    if swagger:
        console.print("[bold blue]Parsing Swagger spec...[/]")
        endpoints = container.swagger_parser.parse(swagger)
        console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    req_items = []
    if requirements:
        console.print("[bold blue]Parsing requirements...[/]")
        req_items = container.requirement_parser.parse(requirements)
        console.print(f"  Found [green]{len(req_items)}[/] requirements")

    console.print("[bold blue]Generating test cases via LLM...[/]")
    test_cases = container.testcase_generator.generate(
        TestCaseGenInput(endpoints=endpoints, requirements=req_items)
    )
    console.print(f"  Generated [green]{len(test_cases)}[/] test cases")

    output_path = Path(output)
    if output_format == "markdown":
        report = container.testcase_report.generate(
            TestCaseReportInput(
                test_cases=test_cases,
                output_format="markdown",
                output_language=container.settings.output_language,
            )
        )
        output_path = output_path.with_suffix(".md")
        container.testcase_report.save(report, output_path)
    elif output_format == "csv":
        output_path = output_path.with_suffix(".csv")
        container.testcase_generator.save_csv(test_cases, output_path)
    else:
        container.testcase_generator.save(test_cases, output_path)

    console.print(f"  Token usage: [dim]{container.llm_client.usage.summary()}[/]")
    console.print(f"[bold green]Done![/] Output saved to [cyan]{output_path}[/]")


@main.command(cls=TestAgentCommand, examples_text=GENERATE_PERF_EXAMPLES)
@click.option(
    "--swagger",
    "-s",
    required=True,
    help="Path or URL to Swagger/OpenAPI spec",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output file path",
)
@click.option(
    "--format",
    "-f",
    "script_format",
    type=click.Choice(["k6", "jmeter"]),
    default=None,
    help="Script format (default from config)",
)
@click.option("--base-url", default=None, help="Base URL for the API")
@click.option("--virtual-users", type=int, default=None, help="Number of virtual users")
@click.option("--duration", type=int, default=None, help="Test duration in seconds")
@click.pass_context
def generate_perf(
    ctx: click.Context,
    swagger: str,
    output: str | None,
    script_format: str | None,
    base_url: str | None,
    virtual_users: int | None,
    duration: int | None,
) -> None:
    """Generate performance test script from API spec."""
    container: Container = ctx.obj["container"]
    settings = container.settings

    console.print("[bold blue]Parsing Swagger spec...[/]")
    endpoints = container.swagger_parser.parse(swagger)
    console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    # Build performance config
    perf_config = PerformanceConfig(
        base_url=base_url or settings.perf.base_url,
        virtual_users=virtual_users or settings.perf.virtual_users,
        duration_seconds=duration or settings.perf.duration_seconds,
        ramp_up_seconds=settings.perf.ramp_up_seconds,
        think_time_ms=settings.perf.think_time_ms,
    )

    fmt = script_format or settings.script_format

    console.print(f"[bold blue]Generating {fmt} performance script via LLM...[/]")
    from testagent.generators.performance_generator import PerformanceGenerator

    generator = PerformanceGenerator(
        llm_client=container.llm_client,
        prompt_builder=container.prompt_builder,
        script_format=fmt,
        output_language=container.settings.output_language,
    )
    script = generator.generate(PerfGenInput(endpoints=endpoints, config=perf_config))

    # Determine output path
    if output:
        output_path = Path(output)
    else:
        ext = ".jmx" if fmt == "jmeter" else ".js"
        output_path = Path(settings.output_dir) / f"perf_test{ext}"

    generator.save(script, output_path)
    console.print(f"[bold green]Done![/] Script saved to [cyan]{output_path}[/]")

    # Generate report
    console.print("[bold blue]Generating report...[/]")
    report = container.performance_report.generate(
        PerfReportInput(
            script_path=str(output_path),
            config=perf_config,
            output_language=container.settings.output_language,
        )
    )
    report_path = output_path.with_suffix(".md")
    container.performance_report.save(report, report_path)
    console.print(f"  Report saved to [cyan]{report_path}[/]")
    console.print(f"  Token usage: [dim]{container.llm_client.usage.summary()}[/]")


@main.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Show current configuration."""
    container: Container = ctx.obj["container"]
    settings = container.settings

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


if __name__ == "__main__":
    main()
