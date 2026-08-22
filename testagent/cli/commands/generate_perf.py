"""generate-perf command: produce k6/JMeter performance scripts from an API spec."""

from pathlib import Path

import click

from testagent.cli.base import TestAgentCommand, console, main
from testagent.cli.examples import GENERATE_PERF_EXAMPLES
from testagent.config.models import PerfGenInput, PerformanceConfig, PerfReportInput
from testagent.container import Container


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
@click.option("--ramp-up", type=int, default=None, help="Ramp-up period in seconds")
@click.option("--think-time", type=int, default=None, help="Think time between requests in ms")
@click.option(
    "--auth-type",
    type=click.Choice(["none", "bearer", "basic"]),
    default=None,
    help="Authentication type to embed in the script",
)
@click.option(
    "--review/--no-review",
    "review",
    default=None,
    help="Override REVIEW_ENABLED: cross-validate the generated script",
)
@click.pass_context
def generate_perf(
    ctx: click.Context,
    swagger: str,
    output: str | None,
    script_format: str | None,
    base_url: str | None,
    virtual_users: int | None,
    duration: int | None,
    ramp_up: int | None,
    think_time: int | None,
    auth_type: str | None,
    review: bool | None,
) -> None:
    """Generate performance test script from API spec."""
    container: Container = ctx.obj["container"]
    settings = container.settings()

    console.print("[bold blue]Parsing Swagger spec...[/]")
    endpoints = container.swagger_parser().parse(swagger)
    console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    # Build performance config
    perf_config = PerformanceConfig(
        base_url=base_url or settings.perf.base_url,
        virtual_users=virtual_users or settings.perf.virtual_users,
        duration_seconds=duration or settings.perf.duration_seconds,
        ramp_up_seconds=ramp_up or settings.perf.ramp_up_seconds,
        think_time_ms=think_time or settings.perf.think_time_ms,
        auth_type=auth_type or settings.perf.auth_type,
    )

    fmt = script_format or settings.script_format

    console.print(f"[bold blue]Generating {fmt} performance script via LLM...[/]")
    # Resolve through DI so the generator keeps its review wiring (review
    # client / rounds) instead of silently losing it to a hand-built instance.
    generator = container.performance_generator()
    generator.set_script_format(fmt)
    if review is not None:
        generator.set_review_enabled(review)
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
    report = container.performance_report().generate(
        PerfReportInput(
            script_path=str(output_path),
            config=perf_config,
            output_language=container.settings().output_language,
        )
    )
    report_path = output_path.with_suffix(".md")
    container.performance_report().save(report, report_path)
    console.print(f"  Report saved to [cyan]{report_path}[/]")
    console.print(f"  Token usage: [dim]{container.llm_client().usage.summary()}[/]")
