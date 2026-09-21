"""links-check command: offline Gate replay over an existing artifact (LINK-S7).

Read-only by contract (v15 §8.3): it reports and never rewrites the artifact.
Exit codes carry meaning and the command does not mask them: ``0`` clean,
``1`` violations found / candidates left unfulfilled, ``2`` the inputs, the
sidecar or the recorded hashes do not add up.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from testagent.cli.base import TestAgentCommand, console, main
from testagent.parsers.swagger_parser import SwaggerParser
from testagent.pipeline.links_replay import ReportError, load_document, observe, replay

_LINKS_CHECK_EXAMPLES = """
\b
testagent links-check --input cases.json --spec api.json
testagent links-check --input cases.json --spec api.json --run-metadata run.links.json -o report.json
"""


@main.command(name="links-check", cls=TestAgentCommand, examples_text=_LINKS_CHECK_EXAMPLES)
@click.option("--input", "input_path", required=True, help="Test case artifact (.json list)")
@click.option("--spec", "spec_path", required=True, help="Swagger/OpenAPI spec the cases target")
@click.option(
    "--run-metadata",
    default=None,
    help="The run's <session>.links.json sidecar; omit for observation mode (selection unknown)",
)
@click.option("--output", "-o", default=None, help="Write the check report here (JSON)")
@click.option("--quiet", is_flag=True, default=False, help="Only print the summary line")
def links_check(
    input_path: str, spec_path: str, run_metadata: str | None, output: str | None, quiet: bool
) -> None:
    """Check an artifact against path contracts (Gate 1/2/3) without any LLM."""
    try:
        cases = json.loads(Path(input_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]✗[/red] cannot read artifact {input_path}: {exc}")
        raise SystemExit(2) from exc
    if isinstance(cases, dict):
        for key in ("test_cases", "cases", "data"):
            if isinstance(cases.get(key), list):
                cases = cases[key]
                break
    if not isinstance(cases, list):
        console.print("[red]✗[/red] artifact is not a list of test cases")
        raise SystemExit(2)
    try:
        endpoints = SwaggerParser().parse(spec_path)
    except Exception as exc:  # parser raises on unreadable specs
        console.print(f"[red]✗[/red] cannot read spec {spec_path}: {exc}")
        raise SystemExit(2) from exc

    try:
        if run_metadata:
            document = load_document(Path(run_metadata).read_text(encoding="utf-8"))
            report = replay(document, cases, endpoints)
        else:
            report = observe(cases, endpoints)
    except (ReportError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[red]✗[/red] {exc}")
        raise SystemExit(2) from exc

    payload = {
        "mode": report.mode,
        "run_id": report.run_id,
        "errors": report.errors,
        "findings": report.findings,
        "metrics": report.metrics,
        "cases": report.per_case,
        "exit_code": report.exit_code,
    }
    if output:
        target = Path(output)
        if target.exists():
            console.print(f"[red]✗[/red] refusing to overwrite existing {target}")
            raise SystemExit(2)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not quiet:
        # click.echo, not console.print: the report JSON is full of ``[...]``
        # fragments that rich would consume as markup.
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    for error in report.errors:
        click.echo(f"ERROR: {error}", err=True)
    # Plain echo: rich reads ``[replay]`` as a markup tag and swallows the one
    # token this line exists to show.
    click.echo(
        f"links-check [{report.mode}] cases={len(cases)} findings={len(report.findings)} "
        f"errors={len(report.errors)} exit={report.exit_code}"
    )
    raise SystemExit(report.exit_code)
