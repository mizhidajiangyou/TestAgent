"""
CLI entry point for TestAgent.

Provides commands for generating test cases, performance scripts, GUI test
scripts, and interactive conversational refinement.
"""

import os
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from testagent.config.logging import setup_logging
from testagent.config.models import (
    GUITestGenInput,
    PerfGenInput,
    PerformanceConfig,
    PerfReportInput,
    TestCaseGenInput,
    TestCaseReportInput,
)
from testagent.container import Container
from testagent.generators.testcase_generator import TestCaseGenerator

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

  # Incremental generation: reuse previous cases as baseline, only add net-new cases
  testagent generate-tests -s swagger.json -r new_requirements.md -H ./output/testcases.json

  # Parse Swagger from URL
  testagent generate-tests -s https://petstore3.swagger.io/api/v3/openapi.json

  # Enable multi-model fallback + cross-validation review
  OPENAI_MODEL="gpt-4o-mini,gpt-4o" REVIEW_ENABLED=true REVIEW_MAX_ROUNDS=2 \\
      testagent generate-tests -s swagger.json -r requirements.md -f markdown

  # Generate k6 performance script (default format)
  testagent generate-perf -s swagger.json --base-url https://api.example.com

  # Generate JMeter JMX script
  testagent generate-perf -s swagger.json -f jmeter -o ./output/perf.jmx

  # Generate Playwright GUI test script
  testagent generate-gui -r requirements.md --url https://example.com -o ./output/gui_test.py

  # Interactive conversational refinement (generate then refine via dialogue)
  testagent chat -r requirements.md -s swagger.json

  # Start the web GUI (embeddable via iframe in other platforms)
  testagent serve --port 8000

  # Run generated scripts
  k6 run output/perf_test.js
  jmeter -n -t output/perf_test.jmx -l results.jtl
  pytest output/gui_test.py --browser chromium

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

  # Incremental: reuse a previous test case baseline, only generate net-new cases
  testagent generate-tests -s swagger.json -r new_requirements.md -H ./output/testcases.json
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
@click.option(
    "--historical-cases",
    "-H",
    default=None,
    help="Path to historical test cases JSON file (baseline for incremental generation)",
)
@click.pass_context
def generate_tests(
    ctx: click.Context,
    swagger: str | None,
    requirements: str | None,
    output: str,
    output_format: str,
    historical_cases: str | None,
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

    historical = []
    if historical_cases:
        historical = TestCaseGenerator.load_historical_cases(historical_cases)
        console.print(
            f"  Loaded [green]{len(historical)}[/] historical cases as baseline "
            f"from [cyan]{historical_cases}[/]"
        )

    console.print("[bold blue]Generating test cases via LLM...[/]")
    test_cases = container.testcase_generator.generate(
        TestCaseGenInput(
            endpoints=endpoints,
            requirements=req_items,
            historical_cases=historical,
        )
    )
    if historical:
        console.print(
            f"  Merged: [green]{len(historical)}[/] historical + net-new = "
            f"[green]{len(test_cases)}[/] total"
        )
    else:
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


GENERATE_GUI_EXAMPLES = """
EXAMPLES:
  # Generate Playwright test from requirements + target URL
  testagent generate-gui -r requirements.md --url https://example.com

  # With Swagger context for API-aware GUI tests
  testagent generate-gui -r requirements.md -s swagger.json --url https://app.example.com

  # Save to custom path
  testagent generate-gui -r requirements.md --url https://example.com -o tests/test_login.py

  # Run the generated script
  pytest output/gui_test.py --browser chromium
"""


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
@click.pass_context
def generate_gui(
    ctx: click.Context,
    requirements: str,
    swagger: str | None,
    url: str | None,
    output: str | None,
) -> None:
    """Generate Playwright Python test script for web/GUI testing."""
    container: Container = ctx.obj["container"]
    settings = container.settings

    console.print("[bold blue]Parsing requirements...[/]")
    req_items = container.requirement_parser.parse(requirements)
    console.print(f"  Found [green]{len(req_items)}[/] requirements")

    endpoints = []
    if swagger:
        console.print("[bold blue]Parsing Swagger spec...[/]")
        endpoints = container.swagger_parser.parse(swagger)
        console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    if not url:
        url = settings.perf.base_url
        console.print(f"  [dim]Using default base URL: {url}[/]")

    console.print("[bold blue]Generating Playwright GUI test script via LLM...[/]")
    from testagent.generators.gui_test_generator import GUITestGenerator

    generator = GUITestGenerator(
        llm_client=container.llm_client,
        prompt_builder=container.prompt_builder,
        output_language=settings.output_language,
    )
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
    console.print(f"  Token usage: [dim]{container.llm_client.usage.summary()}[/]")
    console.print(f"  [dim]Run with: pytest {output_path} --browser chromium[/]")


CHAT_EXAMPLES = """
EXAMPLES:
  # Start interactive chat with requirements + swagger context
  testagent chat -r requirements.md -s swagger.json

  # Chat with just requirements
  testagent chat -r requirements.md

  # Chat with a specific session ID (resume previous session)
  testagent chat -r requirements.md --session my-session-1

  # In the chat, type natural language:
  #   > Generate test cases for the user registration module
  #   > Add more boundary test cases for the password field
  #   > Validate the current test cases
  #   > Refine the test cases to be more concise
  #   > exit
"""


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
@click.pass_context
def chat(
    ctx: click.Context,
    requirements: str | None,
    swagger: str | None,
    session: str | None,
    max_iterations: int | None,
) -> None:
    """Interactive conversational refinement of generated artifacts.

    Supports continuous dialogue to generate, validate, and refine
    test cases, performance scripts, and GUI test scripts. Inspired
    by langgraph's StateGraph + checkpoint pattern: conversation state
    persists across turns within the same session.
    """
    container: Container = ctx.obj["container"]
    settings = container.settings

    # Parse context if provided
    context: dict[str, Any] = {}
    if requirements:
        console.print("[bold blue]Parsing requirements...[/]")
        req_items = container.requirement_parser.parse(requirements)
        context["requirements"] = (
            container.requirement_parser.requirements_to_text(req_items)
            if hasattr(container.requirement_parser, "requirements_to_text")
            else str(req_items)
        )
        console.print(f"  Found [green]{len(req_items)}[/] requirements")

    if swagger:
        console.print("[bold blue]Parsing Swagger spec...[/]")
        endpoints = container.swagger_parser.parse(swagger)
        from testagent.parsers.swagger_parser import SwaggerParser

        context["endpoints"] = SwaggerParser.endpoints_to_text(endpoints)
        console.print(f"  Found [green]{len(endpoints)}[/] endpoints")

    # Get or create conversation session
    manager = container.conversation_manager
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
        except (EOFError, KeyboardInterrupt):
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
            f"Token usage: {container.llm_client.usage.summary()}[/]"
        )
        console.print()


SERVE_EXAMPLES = """
EXAMPLES:
  # Start the web GUI on the default port (8000)
  testagent serve

  # Custom host/port
  testagent serve --host 0.0.0.0 --port 8080

  # Restrict iframe embedding to specific origins (default: allow all)
  WEB_FRAME_ANCESTORS="https://app.example.com https://portal.example.com" testagent serve

  # Embed in another platform via iframe
  #   <iframe src="http://localhost:8000/" width="100%" height="800"></iframe>

  # Install the web extra first if not already installed:
  #   pip install -e ".[web]"
"""


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
