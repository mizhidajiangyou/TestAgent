"""LINK-S5b / S6b / T12b: the links pass wired onto the task-package chain.

v15 §8.2 pins what the switch must do, and both directions are gates:

- ``off`` (the default): the ORIGINAL prompts, Phase 2 still batched in pairs,
  no L3b units, no identity columns, no report — i.e. "as if this code were
  never added", which is what keeps the FH2.4/FH2.6/FH2.7 recordings valid;
- ``on``: L0/L1 batching material, L3a clusters instead of blind pairs, one
  unit per selected path within ``LINKS_L3B_MAX_CALLS``, program-stamped
  provenance, and a Gate report with per-stage denominators.

The 13-endpoint ecommerce spec is the shared measuring stick (the same fixture
``scripts/measure_baseline.py`` asserts): 7 pairs off, 3 clusters on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from testagent.engine.llm_client import LLMClient, LLMResponse
from testagent.parsers.swagger_parser import endpoints_to_rich_signature
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import build_engine_generate_unit, build_generate_unit
from tests.parity_harness import bare_settings

REPO = Path(__file__).parents[1]
_SPEC = (
    str(REPO / "examples" / "ecommerce_requirements.md"),
    str(REPO / "examples" / "ecommerce_swagger.json"),
)
_ENDPOINT_LINE = re.compile(r"^- (GET|POST|PUT|PATCH|DELETE) (\S+)", re.MULTILINE)


class _LinksLLM(LLMClient):
    """Answers with one case for the first endpoint the prompt shows, and
    FORGES the identity columns so the overwrite-and-record path is exercised
    on every unit rather than only in a synthetic test."""

    intent_capable = False

    def __init__(self) -> None:
        self.prompts: list[tuple[str, str]] = []
        self.usage = None  # type: ignore[assignment]

    def _respond(self, user: str) -> str:
        match = _ENDPOINT_LINE.search(user)
        endpoint = f"{match.group(1)} {match.group(2)}" if match else "N/A"
        return json.dumps(
            [
                {
                    "id": "TC-XXX",
                    "title": f"case for {endpoint}",
                    "description": "one case",
                    "endpoint": endpoint,
                    "test_type": "functional",
                    "priority": "high",
                    "preconditions": [],
                    "steps": [f"call {endpoint}"],
                    "expected_results": ["200 and a payload"],
                    "tags": ["smoke"],
                    "path_id": "P-HACKED",
                    "source_stage": "l3b",
                }
            ]
        )

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):
        self.prompts.append((system, user))
        return LLMResponse(text=self._respond(user), finish_reason="stop", completion_tokens=20)

    async def achat(self, system, user, response_format=None, max_tokens=None):
        return self._respond(user)


@contextlib.contextmanager
def _in_dir(tmp: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(previous)


def _run(tmp: Path, *, links: bool, historical: str = "") -> tuple[Any, list[Any]]:
    tmp.mkdir(parents=True, exist_ok=True)
    settings = bare_settings(
        output_dir=str(tmp / "output"),
        links_enabled=links,
        links_prose_enabled=True,
    )
    llm = _LinksLLM()
    task = get_registry(REPO / "tasks").get("testcase")
    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_engine_generate_unit(
            llm,  # type: ignore[arg-type]
            output_token_cap=settings.llm.max_output_tokens,
        ),
        links_signature_fn=endpoints_to_rich_signature,
    )
    raw: dict[str, Any] = {"requirements": _SPEC[0], "swagger": _SPEC[1]}
    if links:
        raw["links"] = True
    if historical:
        (tmp / "hist.json").write_text(historical, encoding="utf-8")
        raw["historical_cases"] = "hist.json"
    with _in_dir(tmp):
        ctx = parse_inputs(task.manifest, raw, settings)
        result = asyncio.run(executor.arun(task, ctx, session_id=f"links-{links}"))
    return result, llm.prompts


def test_links_off_is_the_pre_links_run(tmp_path: Path) -> None:
    """Batching, prompts and artifact must look like links was never added."""
    result, prompts = _run(tmp_path / "off", links=False)
    units = {stat.name: stat.units_total for stat in result.stage_stats}
    assert units == {"phase1": 4, "phase2_api": 7}, f"13 endpoints must pair, got {units}"
    assert "l3b" not in units
    joined = "\n".join(f"{s}\n{u}" for s, u in prompts)
    assert "[L0 endpoint index]" not in joined, "links material leaked into a links-off prompt"
    assert result.links_report is None, "a links-off run must not publish a gate report"
    artifact = result.artifact if isinstance(result.artifact, list) else []
    assert artifact, "the run produced nothing, so the column checks would be vacuous"
    assert not [c for c in artifact if c.get("path_id") or c.get("source_stage")], (
        "identity columns moved without the links switch"
    )


def test_links_on_clusters_and_runs_l3b_within_budget(tmp_path: Path) -> None:
    result, prompts = _run(tmp_path / "on", links=True)
    units = {stat.name: stat.units_total for stat in result.stage_stats}
    assert units["phase2_api"] == 3, f"L3a clusters replace the 7 pairs, got {units}"
    assert units["l3b"] <= 3, f"LINKS_L3B_MAX_CALLS=3, ran {units['l3b']} units"
    assert units["l3b"] >= 1, "no path unit ran, so the L3b wiring is unproven"
    assert all("[L0 endpoint index]" in user for _, user in prompts), "L0 must reach every unit"
    artifact = result.artifact if isinstance(result.artifact, list) else []
    stages = {str(c.get("source_stage", "")) for c in artifact}
    assert "l3b" in stages and "phase2" in stages, f"provenance missing: {stages}"
    for case in artifact:
        if case.get("source_stage") == "l3b":
            assert case.get("path_id"), "an L3b case without path_id cannot be fulfilled"
        else:
            assert not case.get("path_id"), "only L3b units may carry a path_id"


def test_links_on_overwrites_and_records_model_identity(tmp_path: Path) -> None:
    """Every scripted answer forges ``path_id``/``source_stage``."""
    result, _ = _run(tmp_path / "forgery", links=True)
    report = result.links_report or {}
    assert report.get("model_forged_identity", 0) >= 1, (
        "the fake forged identity on every unit and nothing was recorded"
    )
    artifact = result.artifact if isinstance(result.artifact, list) else []
    assert "P-HACKED" not in {str(c.get("path_id", "")) for c in artifact}


def test_gate_report_carries_per_stage_denominators(tmp_path: Path) -> None:
    result, _ = _run(tmp_path / "report", links=True)
    report = result.links_report or {}
    for key in (
        "planned_paths",
        "selected_paths",
        "candidate_paths",
        "pair_rate",
        "path_rate",
        "grades",
        "grades_by_source_stage",
    ):
        assert key in report, f"report lacks {key!r}: {sorted(report)}"
    l3b = report["grades_by_source_stage"].get("l3b")
    assert l3b and l3b["total"] >= 1 and 0.0 <= l3b["rejected_rate"] <= 1.0
    # A trivial scripted case cannot fulfil a path contract; a report that
    # graded them INTEGRATION here would be a gate that passes on nothing.
    assert l3b.get("REJECTED", 0) >= 1, f"the gates accepted non-fulfilling cases: {l3b}"


def test_external_history_cannot_borrow_this_runs_provenance(tmp_path: Path) -> None:
    """v15 §7.3: identity columns from another run are audit history only."""
    history = json.dumps(
        [
            {
                "id": "TC-900",
                "title": "historical case with borrowed provenance",
                "endpoint": "GET /products",
                "test_type": "functional",
                "priority": "high",
                "steps": ["call GET /products"],
                "expected_results": ["200"],
                "path_id": "P-FROM-OLD-RUN",
                "source_stage": "l3b",
            }
        ]
    )
    result, _ = _run(tmp_path / "history", links=True, historical=history)
    artifact = result.artifact if isinstance(result.artifact, list) else []
    baseline = next(c for c in artifact if c.get("id") == "TC-001")
    assert not baseline.get("path_id"), "the historical item kept another run's path_id"
    assert (result.links_report or {}).get("history_identity_cleared", 0) >= 1


def test_two_concurrent_runs_do_not_share_links_state(tmp_path: Path) -> None:
    """v15 §7.4: the builder, the planning result and the ledgers are per-run.

    Two runs over the same executor would otherwise share one ``LinksPass``,
    and a path stamped by run A would be counted as fulfilled by run B.
    """
    settings = bare_settings(
        output_dir=str(tmp_path / "output"), links_enabled=True, links_prose_enabled=True
    )
    task = get_registry(REPO / "tasks").get("testcase")
    runs: list[PipelineExecutor] = []
    for _ in range(2):
        llm = _LinksLLM()
        runs.append(
            PipelineExecutor(
                llm,  # type: ignore[arg-type]
                settings,
                generate_unit=build_engine_generate_unit(
                    llm,  # type: ignore[arg-type]
                    output_token_cap=settings.llm.max_output_tokens,
                ),
                links_signature_fn=endpoints_to_rich_signature,
            )
        )
    raw = {"requirements": _SPEC[0], "swagger": _SPEC[1], "links": True}

    async def both() -> list[Any]:
        return list(
            await asyncio.gather(
                *(
                    executor.arun(task, ctx, session_id=f"concurrent-{i}")
                    for i, (executor, ctx) in enumerate(zip(runs, contexts, strict=True))
                )
            )
        )

    with _in_dir(tmp_path):
        contexts = [parse_inputs(task.manifest, dict(raw), settings) for _ in runs]
        results = asyncio.run(both())
    assert results[0].links_report and results[1].links_report
    assert results[0].session_id != results[1].session_id
    assert contexts[0].links is not contexts[1].links, "two runs shared one LinksPass"
    for index, (result, context) in enumerate(zip(results, contexts, strict=True)):
        artifact = result.artifact if isinstance(result.artifact, list) else []
        assert artifact, f"run {index} produced nothing"
        stamped = {str(c.get("path_id", "")) for c in artifact if c.get("path_id")}
        own = {bundle.path_id for bundle in context.links.bundles}
        assert stamped <= own, f"run {index} carries paths from another run: {stamped - own}"


def test_packages_without_the_declaration_stay_off(tmp_path: Path) -> None:
    """v15 §10: the adapter is default-off, and a package that never declared
    ``pipeline.links`` cannot be switched on by configuration alone."""
    tmp = tmp_path / "perf"
    tmp.mkdir()
    settings = bare_settings(output_dir=str(tmp / "output"), links_enabled=True)
    llm = _LinksLLM()
    task = get_registry(REPO / "tasks").get("perf")
    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_generate_unit(llm),  # type: ignore[arg-type]
        links_signature_fn=endpoints_to_rich_signature,
    )
    with _in_dir(tmp):
        ctx = parse_inputs(
            task.manifest, {"swagger": _SPEC[1], "base_url": "https://x.test"}, settings
        )
        result = asyncio.run(executor.arun(task, ctx, session_id="perf-links-off"))
    assert ctx.links is None, "a package without pipeline.links opted in anyway"
    assert result.links_report is None


def test_cli_replays_the_run_it_shipped(tmp_path: Path) -> None:
    """LINK-S7: ``links-check`` reads the sidecar and refuses to re-plan.

    Round trip on a real run: the pipeline writes artifact + sidecar, the CLI
    replays the recorded contracts, and a tampered document exits 2 rather than
    quietly measuring a different denominator.
    """
    from click.testing import CliRunner

    from testagent.cli import main

    result, _ = _run(tmp_path / "cli", links=True)
    out = tmp_path / "cli"
    artifact_path = out / "artifact.json"
    artifact_path.write_text(json.dumps(result.artifact, ensure_ascii=False), encoding="utf-8")
    sidecar = out / "output" / f"{result.session_id}.links.json"
    assert sidecar.exists(), "the run did not persist its links sidecar"

    runner = CliRunner()
    checked = runner.invoke(
        main,
        [
            "links-check",
            "--input",
            str(artifact_path),
            "--spec",
            _SPEC[1],
            "--run-metadata",
            str(sidecar),
            "--quiet",
        ],
        catch_exceptions=False,
    )
    assert checked.exit_code in (0, 1), checked.output
    assert f"links-check [replay] cases={len(result.artifact)}" in checked.output, checked.output

    tampered = json.loads(sidecar.read_text(encoding="utf-8"))
    tampered["candidate_contracts"][0]["payload"]["endpoints"] = ["GET /nowhere"]
    bad_path = out / "tampered.links.json"
    bad_path.write_text(json.dumps(tampered), encoding="utf-8")
    rejected = runner.invoke(
        main,
        [
            "links-check",
            "--input",
            str(artifact_path),
            "--spec",
            _SPEC[1],
            "--run-metadata",
            str(bad_path),
            "--quiet",
        ],
        catch_exceptions=False,
    )
    assert rejected.exit_code == 2, f"an edited contract was not refused: {rejected.output}"

    wrong_artifact = out / "other.json"
    wrong_artifact.write_text(
        json.dumps([dict(result.artifact[0], title="changed after the run")]), encoding="utf-8"
    )
    mismatch = runner.invoke(
        main,
        [
            "links-check",
            "--input",
            str(wrong_artifact),
            "--spec",
            _SPEC[1],
            "--run-metadata",
            str(sidecar),
            "--quiet",
        ],
        catch_exceptions=False,
    )
    assert mismatch.exit_code == 2, "a different artifact passed the hash check"


def test_cli_observation_mode_never_claims_coverage(tmp_path: Path) -> None:
    """No sidecar: text-level facts only, and the n/a denominators stay n/a."""
    from click.testing import CliRunner

    from testagent.cli import main

    artifact = tmp_path / "cases.json"
    artifact.write_text(
        json.dumps(
            [
                {
                    "id": "TC-001",
                    "title": "orphan placeholder use",
                    "endpoint": "GET /products/{id}",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": ["GET /products/{id} params.id=<PROD_ID>"],
                    "expected_results": ["200"],
                    "path_id": "P-MADE-UP-BY-THE-MODEL",
                }
            ]
        ),
        encoding="utf-8",
    )
    checked = CliRunner().invoke(
        main,
        ["links-check", "--input", str(artifact), "--spec", _SPEC[1]],
        catch_exceptions=False,
    )
    assert "links-check [observe]" in checked.output
    assert '"selection": "unknown"' in checked.output, checked.output
    assert '"contract_coverage": "n/a"' in checked.output, checked.output
    # A placeholder consumed but never produced is a fact in the text: it must
    # be reported even though nothing is claimed about path coverage.
    assert checked.exit_code == 1, checked.output
