"""review command: independent review of an existing artifact (v4 §3.2).

Mode A (requirement-grounded, -r given) or mode B (quality-review). Paths,
no-clobber, backup and report ordering follow publication.py; exit codes:
0 ok (REVIEWED/REJECTED), 1 failure/publication error, 2 usage errors.
"""

from __future__ import annotations

from pathlib import Path

import click

from testagent.artifact import (
    ChunkPlan,
    load_artifact,
    plan_chunks,
    reassemble_text,
    serialize_json,
)
from testagent.artifact.loader import ArtifactLoadError
from testagent.artifact.models import DocumentStatus, LoadedArtifact
from testagent.artifact.publication import publish_outcome, resolve_publication_paths
from testagent.cli.base import TestAgentCommand, console, main
from testagent.cli.examples import GENERATE_TESTS_EXAMPLES
from testagent.container import Container
from testagent.review import DocumentReviewService

_REVIEW_EXAMPLES = """
\b
testagent review --doc cases.json -r req.md --rounds 2 -o cases_final.json
testagent review --doc notes.md --rounds 2
"""


@main.command(cls=TestAgentCommand, examples_text=_REVIEW_EXAMPLES or GENERATE_TESTS_EXAMPLES)
@click.option(
    "--doc", "doc_path", required=True, help="Artifact to review (.json/.md/.markdown/.txt)"
)
@click.option(
    "--requirement",
    "-r",
    default=None,
    help="Requirement reference (mode A). Missing/invalid = FAILED, never mode B fallback.",
)
@click.option(
    "--rounds",
    default=None,
    type=click.IntRange(min=0),
    help="Review rounds (0 = explicitly disabled). Default: 2.",
)
@click.option(
    "--output", "-o", default=None, help="Reviewed output path (exclusive with --in-place)"
)
@click.option(
    "--in-place",
    is_flag=True,
    default=False,
    help="Review the source file in place (creates .bak backup)",
)
@click.pass_context
def review(
    ctx: click.Context,
    doc_path: str,
    requirement: str | None,
    rounds: int | None,
    output: str | None,
    in_place: bool,
) -> None:
    """Review an existing test-case artifact (with or without requirement reference).

    Quality-review without reference does NOT verify requirement coverage and
    is not guaranteed to improve the document.
    """
    container: Container = ctx.obj["container"]
    if output and in_place:
        console.print("[red]Error: --output and --in-place are mutually exclusive.[/]")
        raise SystemExit(2)
    effective_rounds = 2 if rounds is None else rounds

    source = Path(doc_path)
    try:
        artifact: LoadedArtifact = load_artifact(source)
    except ArtifactLoadError as exc:
        console.print(f"[red]Artifact load failed:[/] {exc}")
        raise SystemExit(1) from None

    try:
        target, report = resolve_publication_paths(source, output, in_place)
    except ValueError as exc:
        console.print(f"[red]Path error:[/] {exc}")
        raise SystemExit(2) from None

    # Reference load (mode A only): any failure is FAILED, not mode B.
    reference_text = ""
    grounded = False
    if requirement is not None:
        grounded = True
        ref_path = Path(requirement)
        try:
            if not ref_path.exists():
                raise ArtifactLoadError(f"reference not found: {ref_path}")
            reference_text = ref_path.read_text(encoding="utf-8")
            if not reference_text.strip():
                raise ArtifactLoadError("reference has no usable content")
        except (ArtifactLoadError, UnicodeDecodeError) as exc:
            console.print(f"[red]Reference load failed (mode A does not fall back):[/] {exc}")
            _publish_failure_report(
                container,
                artifact,
                source,
                target,
                report,
                in_place,
                f"reference-unavailable:{exc}",
            )
            raise SystemExit(1) from None

    settings = container.settings()
    service = DocumentReviewService(
        primary_llm=container.llm_client(),
        review_llm=container.review_client(),
        prompt_builder=container.prompt_builder(),
        chunk_size=settings.review_chunk_size,
        max_prompt_chars=settings.review_max_prompt_chars,
        rounds=effective_rounds,
        max_chunk_failure_ratio=settings.review_max_chunk_failure_ratio,
    )

    plan: ChunkPlan = plan_chunks(
        artifact,
        chunk_size=settings.review_chunk_size,
        char_budget=settings.review_max_prompt_chars,
        system_prompt_chars=len(service._system_prompt(grounded)),
        reference_chars=len(reference_text) if grounded else 0,
    )
    oversized = len(plan.oversized_atomic)

    outcome = service.review_document(
        artifact,
        plan.chunks,
        grounded=grounded,
        reference_text=reference_text,
        enabled=True,
    )
    if oversized:
        outcome.diagnostics.append(
            f"oversized_atomic_items={oversized} (zero-call FAILED chunks, counted in denominators)"
        )

    # Build final artifact bytes for REVIEWED outcomes.
    reviewed_bytes: bytes | None = None
    if outcome.status.value == "REVIEWED":
        if artifact.format.value == "json":
            per_item: dict[int, object] = {}
            for chunk_result in outcome.chunk_results:
                chunk = next((c for c in plan.chunks if c.index == chunk_result.chunk_index), None)
                if chunk is None or chunk_result.status.value != "REVIEWED":
                    continue
                for offset, item in enumerate(chunk_result.adopted_items):
                    per_item[chunk.item_start + offset] = item
            reviewed_bytes = serialize_json(artifact, per_item)
        else:
            rewritten = {
                r.chunk_index: ("\n\n".join(str(item) for item in r.adopted_items))
                for r in outcome.chunk_results
                if r.status.value == "REVIEWED" and r.adopted_items
            }
            reviewed_bytes = artifact.bom + reassemble_text(artifact, rewritten).encode("utf-8")

    result, _report_data = publish_outcome(
        outcome,
        artifact,
        source,
        target,
        report,
        in_place=in_place,
        reviewed_bytes=reviewed_bytes,
    )
    console.print(
        f"[bold]Review status:[/] [cyan]{outcome.status.value}[/] | run_id={result.run_id} | "
        f"report={report}"
    )
    if result.error:
        console.print(f"[red]Publication error:[/] {result.error}")
        raise SystemExit(1)
    if outcome.status.value in ("REVIEW_FAILED",):
        raise SystemExit(1)
    if outcome.status.value == "REVIEW_DISABLED":
        raise SystemExit(2)
    # REVIEWED / REVIEW_REJECTED -> 0


def _publish_failure_report(
    container: Container,
    artifact: LoadedArtifact,
    source: Path,
    target: Path,
    report: Path,
    in_place: bool,
    reason: str,
) -> None:
    """Best-effort FAILED report (v4 §6.3: content failures may form a report)."""
    from testagent.artifact.models import DocumentReviewOutcome
    from testagent.artifact.publication import publish_outcome

    outcome = DocumentReviewOutcome(artifact=artifact)
    outcome.status = _failed_status()
    outcome.diagnostics.append(reason)
    publish_outcome(
        outcome, artifact, source, target, report, in_place=in_place, reviewed_bytes=None
    )


def _failed_status() -> DocumentStatus:
    return DocumentStatus.REVIEW_FAILED
