"""QL-1 evidence: legacy chain vs pipeline chain, field by field.

The migration's risk is not "does the pipeline run" but "does it produce the
same artifact semantics". FH2.4 recorded the new chain against itself (there was
no legacy baseline to re-record), so it could not catch a capability only the
legacy chain had — which is exactly how the quality line (T1~T13) stayed out of
the new chain.

This test closes that hole: the same scripted LLM output goes through
(a) ``TestCaseGenerator.generate`` (the legacy host adapter that carried the
quality line) and (b) ``PipelineExecutor.arun`` on ``tasks/testcase`` with the
wired ``QualityPass``. The two artifact lists are then compared field by field,
and the LLM call counts are compared for fan-out equality.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from testagent.config.models import TestCaseGenInput
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import build_engine_generate_unit

# aliased: pytest collects any module-level name starting with ``test``, and the
# adapter's serializer is one (R01 applies to imported callables too)
from testagent.pipeline.testcase_adapter import dict_to_testcase
from testagent.pipeline.testcase_adapter import testcase_to_full_dict as case_row
from tests.parity_harness import bare_settings
from tests.web_fakes import ScriptedLLM

REPO = Path(__file__).parents[1]

_REQUIREMENTS_MD = (
    "# User Management\n"
    "Users can be listed and created.\n\n"
    "Acceptance Criteria:\n"
    "- User can register\n"
    "- User can login\n"
)
_SPEC = {
    "openapi": "3.0.0",
    "paths": {
        "/users": {
            "get": {"tags": ["users"], "responses": {"200": {"description": "OK"}}},
            "post": {"tags": ["users"], "responses": {"201": {"description": "OK"}}},
        }
    },
}


def _case(title: str, endpoint: str) -> dict[str, Any]:
    return {
        "id": "TC-XXX",
        "title": title,
        "description": f"{title} ({endpoint})",
        "endpoint": endpoint,
        "test_type": "functional",
        "priority": "high",
        "preconditions": ["user exists"],
        "steps": [f"call {endpoint}"],
        "expected_results": ["200 and a matching payload"],
        "tags": ["smoke"],
    }


#: Distinct titles per call, so a fan-out difference between the chains cannot
#: hide behind de-duplication.
_RESPONSES = [
    json.dumps([_case("list users ok", "GET /users")]),
    json.dumps([_case("login ok", "GET /users")]),
    json.dumps([_case("register ok", "GET /users")]),
    json.dumps([_case("create user ok", "POST /users")]),
    json.dumps([_case("delete user ok", "GET /users/{id}")]),
    json.dumps([_case("extra call", "GET /users")]),
    json.dumps([_case("extra call 2", "GET /users")]),
]

_KEY_FIELDS = (
    "id",
    "title",
    "endpoint",
    "test_type",
    "priority",
    "steps",
    "expected_results",
    "tags",
)


class _CountingLLM(ScriptedLLM):
    """Counts every LLM call so a fan-out difference is visible."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.call_count = 0

    def _next(self) -> str:
        self.call_count += 1
        return super()._next()


@contextlib.contextmanager
def _in_dir(tmp: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(previous)


def _write_inputs(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "req.md").write_text(_REQUIREMENTS_MD, encoding="utf-8")
    (tmp / "spec.json").write_text(json.dumps(_SPEC), encoding="utf-8")


def _legacy_run(tmp: Path, responses: list[str]) -> tuple[list[dict[str, Any]], int]:
    """The legacy host adapter: same inputs, same scripted answers."""
    _write_inputs(tmp)
    llm = _CountingLLM(responses)
    generator = TestCaseGenerator(llm_client=llm, prompt_builder=PromptBuilder())
    cases = generator.generate(
        TestCaseGenInput(
            endpoints=SwaggerParser().parse(str(tmp / "spec.json")),
            requirements=RequirementParser().parse(str(tmp / "req.md")),
            historical_cases=[],
        ),
        session_id="legacy-eq",
    )
    return [case_row(tc) for tc in cases], llm.call_count


def _pipeline_run(
    tmp: Path, settings: Any, responses: list[str]
) -> tuple[list[dict[str, Any]], int]:
    """The new chain: tasks/testcase + PipelineExecutor + wired QualityPass."""
    _write_inputs(tmp)
    llm = _CountingLLM(responses)
    task = get_registry(REPO / "tasks").get("testcase")
    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_engine_generate_unit(
            llm,  # type: ignore[arg-type]
            output_token_cap=settings.llm.max_output_tokens,
        ),
    )
    with _in_dir(tmp):
        ctx = parse_inputs(
            task.manifest, {"requirements": "req.md", "swagger": "spec.json"}, settings
        )
        result = asyncio.run(executor.arun(task, ctx, session_id="pipeline-eq"))
    artifact = result.artifact if isinstance(result.artifact, list) else []
    cases = [tc for tc in (dict_to_testcase(item) for item in artifact) if tc]
    return [case_row(tc) for tc in cases], llm.call_count


def test_pipeline_matches_legacy_field_by_field(tmp_path: Path) -> None:
    settings = bare_settings(output_dir=str(tmp_path / "output"))
    legacy, legacy_calls = _legacy_run(tmp_path / "legacy", list(_RESPONSES))
    pipeline, pipeline_calls = _pipeline_run(tmp_path / "pipeline", settings, list(_RESPONSES))

    assert legacy_calls == pipeline_calls, (
        f"fan-out differs: legacy made {legacy_calls} LLM calls, pipeline {pipeline_calls}"
    )
    assert legacy, "the legacy chain produced nothing — the comparison would be vacuous"
    diffs = [
        f"{field}: legacy={[row[field] for row in legacy]} pipeline={[row[field] for row in pipeline]}"
        for field in _KEY_FIELDS
        if [row[field] for row in legacy] != [row[field] for row in pipeline]
    ]
    assert len(legacy) == len(pipeline), (
        f"count differs ({len(legacy)} vs {len(pipeline)}): {diffs}"
    )
    assert not diffs, "field-by-field drift:\n" + "\n".join(diffs)


def test_quality_columns_are_populated_on_both_chains(tmp_path: Path) -> None:
    """T10 grades and the deterministic columns must reach the NEW chain's
    artifact — an empty ``executability`` was the symptom that started QL-1."""
    settings = bare_settings(output_dir=str(tmp_path / "output"))
    legacy, _ = _legacy_run(tmp_path / "legacy", list(_RESPONSES))
    pipeline, _ = _pipeline_run(tmp_path / "pipeline", settings, list(_RESPONSES))
    for row, label in ((legacy, "legacy"), (pipeline, "pipeline")):
        assert all(str(case["executability"].get("grade")) for case in row), label
        assert all(case["expected_results"] for case in row), label


def test_adjudicated_difference_generated_dedup_without_baseline(tmp_path: Path) -> None:
    """The one accepted divergence, kept visible instead of hidden.

    Legacy de-duplicated generated cases against each other ONLY when a
    historical baseline existed (the dedup lived inside
    ``_merge_historical_cases``, called under ``if historical_cases:``); the
    pipeline applies the ratified H1 dimension ``duplicate_generated: dropped
    within new_cases`` unconditionally. Identical responses therefore yield
    duplicates on the legacy chain and one copy on the new one — and the
    web contract cell ``ok-with-swagger`` records that accepted change.
    """
    repeated = [json.dumps([_case("same title", "GET /users")])] * 6
    settings = bare_settings(output_dir=str(tmp_path / "output"))
    legacy, _ = _legacy_run(tmp_path / "legacy", list(repeated))
    pipeline, _ = _pipeline_run(tmp_path / "pipeline", settings, list(repeated))
    assert len(legacy) > len(pipeline), (
        "expected the legacy chain to keep generated duplicates without a "
        "historical baseline; if it changed, re-check the adjudication above"
    )
    assert len(pipeline) == 1


def test_new_chain_reconciles_against_raw_calls(tmp_path: Path) -> None:
    """T1 on the new chain: the session audit must be reconcilable.

    ``reconciliation.json`` with ``raw_calls: 0`` beside a non-empty artifact is
    not a pass — it is the audit being dead. The engine's ``raw_sink`` has to be
    wired through ``QualityPass.emit_raw_record`` for the artifact count to mean
    anything next to the raw rows.
    """
    settings = bare_settings(
        output_dir=str(tmp_path / "audit-output"),
        audit_dump_enabled=True,
    )
    _pipeline_run(tmp_path / "audit", settings, list(_RESPONSES))
    sessions = list((tmp_path / "audit-output" / "sessions").glob("*/reconciliation.json"))
    assert sessions, "the run wrote no reconciliation report"
    document = json.loads(sessions[0].read_text(encoding="utf-8"))
    assert document["raw_calls"] > 0, f"raw audit is dead: {document}"
    assert document["rows"], "no raw rows landed"
    assert document["artifact_count"] == len(document["rows"]) or document["match"] is False
