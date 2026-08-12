"""
CLI entry point for TestAgent.

Provides commands for generating test cases and performance scripts.
"""

from pathlib import Path

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


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """TestAgent - AI-powered test case and performance script generator."""
    ctx.ensure_object(dict)
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level)
    ctx.obj["container"] = Container()


@main.command()
@click.option(
    "--swagger",
    "-s",
    required=True,
    help="Path or URL to Swagger/OpenAPI spec",
)
@click.option(
    "--requirements",
    "-r",
    default=None,
    help="Path to requirements document (optional)",
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
    type=click.Choice(["json", "markdown"]),
    default="json",
    help="Output format",
)
@click.pass_context
def generate_tests(
    ctx: click.Context,
    swagger: str,
    requirements: str | None,
    output: str,
    output_format: str,
) -> None:
    """Generate test cases from API spec and requirements."""
    container: Container = ctx.obj["container"]

    console.print("[bold blue]Parsing Swagger spec...[/]")
    endpoints = container.swagger_parser.parse(swagger)
    console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    req_items = None
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
            TestCaseReportInput(test_cases=test_cases, output_format="markdown")
        )
        output_path = output_path.with_suffix(".md")
        container.testcase_report.save(report, output_path)
    else:
        container.testcase_generator.save(test_cases, output_path)

    console.print(f"  Token usage: [dim]{container.llm_client.usage.summary()}[/]")
    console.print(f"[bold green]Done![/] Output saved to [cyan]{output_path}[/]")


@main.command()
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
        PerfReportInput(script_path=str(output_path), config=perf_config)
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
        console.print(f"  Model: {settings.llm.model}")
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


if __name__ == "__main__":
    main()
