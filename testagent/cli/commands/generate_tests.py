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
from testagent.engine.truncation import TruncationPolicy, chars_per_token_for
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.orchestration.input_adapter import resolve_requirements
from testagent.orchestration.settings_override import (
    apply_capabilities_override,
    parse_capability_options,
    resolve_capabilities,
)


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
@click.option(
    "--review/--no-review",
    "review",
    default=None,
    help="Override REVIEW_ENABLED: cross-validate generated test cases",
)
@click.option(
    "--split/--no-split",
    "split",
    default=None,
    help="Requirement input mode: split by chapters (auto) or feed the whole document as one unit (single). Default: SPLIT_MODE/auto.",
)
@click.option(
    "--concurrency",
    default=None,
    type=click.IntRange(min=1),
    help="Override OPENAI_MAX_CONCURRENCY for this run (integer >= 1).",
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
    review: bool | None,
    split: bool | None,
    concurrency: int | None,
) -> None:
    """Generate test cases from requirements and/or API spec."""
    container: Container = ctx.obj["container"]

    # --- capability resolution (v4 §3.3): resume saved > settings > CLI ---
    saved_capabilities = None
    if resume:
        record_path = Path("output/sessions") / f"{resume}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            saved_capabilities = parse_capability_options(record)

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

    # --- capabilities override BEFORE resolving llm/generators (§3.3) ---
    cli_split = {True: "single", False: "auto", None: None}.get(split)
    resolved = resolve_capabilities(
        cli_split=cli_split,
        cli_concurrency=concurrency,
        saved=saved_capabilities,
        settings=container.settings(),
    )
    apply_capabilities_override(container, resolved)
    console.print(
        f"[dim]Effective capabilities: split_mode={resolved.split_mode} "
        f"({resolved.split_mode_source}), concurrency={resolved.concurrency} "
        f"({resolved.concurrency_source})[/]"
    )
    if resolved.concurrency == 1:
        console.print("[dim]serial mode: concurrency=1 (ordered single-worker path)[/]")

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
        console.print(f"[bold blue]Parsing requirements (mode={resolved.split_mode})...[/]")
        req_items = resolve_requirements(
            requirements,
            resolved.split_mode,
            document_parser=container.document_parser(),
        )
        console.print(f"  Found [green]{len(req_items)}[/] requirement unit(s)")
        if resolved.split_mode == "single":
            if endpoints:
                console.print(
                    "[yellow]WARN: single-doc mode with a Swagger spec — Phase 2 "
                    "still batches endpoints; the single unit only covers Phase 1.[/]"
                )
            policy = TruncationPolicy()
            body_chars = len(req_items[0].description)
            est_tokens = body_chars / max(1, chars_per_token_for(req_items[0].description, policy))
            warn_tokens = container.settings().single_doc_warn_tokens
            if est_tokens > warn_tokens:
                console.print(
                    f"[yellow]WARN: single document ≈{int(est_tokens)} estimated tokens "
                    f"exceeds SINGLE_DOC_WARN_TOKENS={warn_tokens} — truncation "
                    "recovery stays active, coverage accounting is limited "
                    "(no per-AC obligations in single mode).[/]"
                )

    historical = []
    if historical_cases:
        historical = TestCaseGenerator.load_historical_cases(historical_cases)
        console.print(
            f"  Loaded [green]{len(historical)}[/] historical cases as baseline "
            f"from [cyan]{historical_cases}[/]"
        )

    console.print(f"[bold]Session:[/] [cyan]{session_id}[/]")
    console.print("[bold blue]Generating test cases via LLM...[/]")
    generator = container.testcase_generator()
    if review is not None:
        generator.set_review_enabled(review)
    try:
        test_cases = asyncio.run(
            generator.agenerate(
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
        session_id,
        swagger,
        requirements,
        output,
        container.settings(),
        resumed_from,
        capability_options=resolved.as_saved(),
    )

    if not test_cases:
        console.print(
            "[red]Generation produced 0 test cases.[/] The LLM returned empty "
            "responses (no usable content) for every requirement. This is "
            "usually an empty/aborted response from the provider — NOT a token "
            "overflow. Recovery steps: set [cyan]OPENAI_STREAM=false[/] to use the "
            "blocking endpoint, configure a secondary model via "
            "[cyan]OPENAI_MODEL[/] (comma-separated), or verify OPENAI_BASE_URL "
            "actually serves the configured model. (Lowering "
            "OPENAI_MAX_OUTPUT_TOKENS will NOT fix an empty response.)"
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
