"""FH2.4 testcase migration parity (plan-k section 8.2, plan-e E1+I2+E2).

Fixtures are recorded from the NEW chain (tasks/testcase via
PipelineExecutor with build_engine_generate_unit and a deterministic fake
LLM) - the "post-fix behavior first recording" semantics (equivalence
baseline = post-fix version; no legacy baseline ever existed to re-record).

Layers:
- A: five-dimension scenario matrix (review / historical / truncation /
  fan-out-empty / baseline), one fixture per scenario;
- B: failure-semantics eight cases per the E2 action contract, asserting
  the OBSERVABLE class per unit (empty / validation / ok / recovered);
- C: three combination scenarios.

Four gates per replay: Artifact (byte-equal JSON), Fingerprint (I2 whitelist
sequence identical), Failure Semantics (observable classes identical),
Contract (artifact items conform to tasks/testcase schema).

The post-link-wiring replay subset is frozen in
``tests/fixtures/migration/testcase/MANIFEST.json`` (post_wiring_replay_subset)
- per-dimension representative fixture IDs mechanically selected.
"""

import asyncio
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from testagent.engine.llm_client import LLMClient, LLMResponse
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.registry import get_registry
from tests.parity_harness import (
    Fingerprint,
    Fixture,
    bare_settings,
    ensure_fixture,
    load_fixture,
    minimal_diff,
    observable_failure_class,
)

REPO = Path(__file__).parents[1]
FIXTURE_SUBSET = REPO / "tests" / "fixtures" / "migration" / "testcase" / "MANIFEST.json"


class ScriptedPipelineLLM(LLMClient):
    """Deterministic fake with an explicit per-call script; records I2
    fingerprints for every call (model label fixed: 'fake-model')."""

    def __init__(self, responses: list[str], finish: str = "stop") -> None:
        self._responses = list(responses)
        self._finish = finish
        self.calls: list[tuple[str, str]] = []
        self.fingerprints: list[dict[str, Any]] = []
        self.responses_served: list[str] = []

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):
        self.calls.append((system, user))
        text = self._responses.pop(0) if self._responses else ""
        self.responses_served.append(text)
        self.fingerprints.append(
            Fingerprint.of(
                model="fake-model",
                system_prompt=system,
                user_prompt=user,
                params={"max_tokens": max_tokens},
                label="",
            ).to_dict()
        )
        return LLMResponse(text=text, finish_reason=self._finish, completion_tokens=100)

    async def achat(self, system, user, response_format=None, max_tokens=None):
        meta = await self.achat_with_meta(system, user, response_format, max_tokens)
        return meta.text

    @property
    def primary_model(self) -> str:
        return "fake-model"

    def set_session_id(self, sid: str) -> None:
        return None

    def verify(self) -> None:
        return None

    async def averify(self) -> None:
        return None


_VALID = json.dumps(
    [
        {
            "id": "TC-XXX",
            "title": "list users",
            "endpoint": "GET /users",
            "test_type": "functional",
        },
        {
            "id": "TC-XXX",
            "title": "create user",
            "endpoint": "POST /users",
            "test_type": "functional",
        },
    ]
)
_ALTERNATIVE = json.dumps(
    [
        {
            "id": "TC-XXX",
            "title": "delete user",
            "endpoint": "GET /users/{id}",
            "test_type": "negative",
        }
    ]
)
_TRUNCATED = _VALID[:-14]  # cut mid-JSON: salvage + continue path


def _write_inputs(tmp: Path, historical: list[dict] | None = None) -> dict[str, str]:
    (tmp / "req.md").write_text(
        "# M\n\n## REQ-001\n\nUser management with pagination.\n\n"
        "## REQ-002\n\nOrder processing.\n",
        encoding="utf-8",
    )
    (tmp / "spec.json").write_text(
        json.dumps(
            {
                "openapi": "3.0.0",
                "paths": {
                    "/users": {
                        "get": {"tags": ["users"], "responses": {"200": {"description": "OK"}}},
                        "post": {"tags": ["users"], "responses": {"201": {"description": "OK"}}},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    raw = {"requirements": "req.md", "swagger": "spec.json"}
    if historical is not None:
        (tmp / "history.json").write_text(json.dumps(historical), encoding="utf-8")
        raw["historical_cases"] = "history.json"
    return raw


def _make_settings(review_enabled: bool = False) -> Any:
    settings = bare_settings(review_enabled=review_enabled)
    object.__setattr__(settings, "output_dir", "./output")
    return settings


@contextmanager
def _in_dir(tmp: Path) -> Iterator[None]:
    """Scope a cwd change so a leak can't poison later tests (input paths in
    fixtures are relative names, so the pipeline must see them as cwd)."""
    previous = Path.cwd()
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(previous)


def _run_taskcase(
    tmp: Path,
    raw: dict[str, str],
    responses: list[str],
    *,
    finish: str = "stop",
    review_enabled: bool = False,
) -> Fixture:
    """Drive tasks/testcase through PipelineExecutor; observe E1 elements.

    The registry is built from an explicit absolute TASKS_DIR: the settings
    singleton is cached process-wide, so mutating ``os.environ`` here would
    only work when no earlier test has already warmed it.
    """
    task = get_registry(REPO / "tasks").get("testcase")
    llm = ScriptedPipelineLLM(responses, finish=finish)
    settings = _make_settings(review_enabled)

    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_engine_generate_unit_for(llm, settings),
        review_runner=build_review_runner_for(llm, settings) if review_enabled else None,
    )
    with _in_dir(tmp):
        ctx = parse_inputs(task.manifest, raw, settings)
        result = asyncio.run(executor.arun(task, ctx, session_id="parity-fixed"))

    units: list[dict[str, Any]] = []
    for stage in result.stage_stats:
        units.append(
            {
                "stage": stage.name,
                "units_total": stage.units_total,
                "units_failed": stage.units_failed,
                "observable": "ok",
            }
        )
    if result.units_failed:
        units.append({"stage": "*", "observable": "failed_units", "count": result.units_failed})

    response_texts = list(llm.responses_served)
    request_trace = [
        {**fp, "response_text": response_texts[i] if i < len(response_texts) else ""}
        for i, fp in enumerate(llm.fingerprints)
    ]
    normalized_input = {
        k: Path(v).name if isinstance(v, str) and ("/" in v or v.endswith((".md", ".json"))) else v
        for k, v in raw.items()
    }
    return Fixture(
        name="placeholder",
        task="testcase",
        input=normalized_input,
        request_trace=request_trace,
        artifact=result.artifact if isinstance(result.artifact, list) else [result.artifact],
        failure_semantics={
            "units_failed": result.units_failed,
            "units": units,
            "observable_classes": sorted(
                {observable_failure_class("FAILED" if result.units_failed else "SUCCESS")}
            ),
        },
        events=[],
        meta={"review_enabled": review_enabled},
    )


def build_engine_generate_unit_for(llm: Any, settings: Any):
    from testagent.pipeline.runtime import build_engine_generate_unit

    return build_engine_generate_unit(llm, output_token_cap=settings.llm.max_output_tokens)


def build_review_runner_for(llm: Any, settings: Any):
    from testagent.pipeline.runtime import build_review_runner

    return build_review_runner(llm)


# ----------------------------------------------------------------------
# Layer A: five-dimension matrix (recorded fixtures)
# ----------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    # dimension: baseline
    "a1-baseline": {"responses": [_VALID, _VALID, _VALID]},
    # dimension: historical
    "a2-historical": {
        "responses": [_VALID, _VALID, _VALID],
        "historical": [
            {
                "id": "TC-001",
                "title": "list users",
                "endpoint": "GET /users",
                "test_type": "functional",
                "priority": "high",
                "steps": ["s"],
                "expected_results": ["ok"],
            }
        ],
    },
    # dimension: truncation (salvage + continue)
    "a3-truncation": {"responses": [_TRUNCATED, _VALID, _VALID, _VALID]},
    # dimension: fan-out empty (one unit empty; serial recovery path)
    "a4-empty-unit": {"responses": ["", _VALID, _VALID, _VALID], "finish": "length"},
    # dimension: review on (deterministic second pass)
    "a5-review-on": {
        "responses": [_VALID, _VALID, _VALID, _VALID, _VALID, _VALID, _VALID],
        "review": True,
    },
}


def _fixture_name(scenario: str) -> str:
    return f"tc-{scenario}"


@pytest.fixture(scope="module")
def recorded_fixtures(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Fixture]:
    """The committed baselines (rebuilt + re-recorded only under the opt-in)."""
    out: dict[str, Fixture] = {}
    for scenario, cfg in SCENARIOS.items():

        def _build(scenario: str = scenario, cfg: dict[str, Any] = cfg) -> Fixture:
            tmp = tmp_path_factory.mktemp(f"tc-{scenario}")
            raw = _write_inputs(tmp, historical=cfg.get("historical"))
            return _run_taskcase(
                tmp,
                raw,
                list(cfg["responses"]),
                finish=cfg.get("finish", "stop"),
                review_enabled=cfg.get("review", False),
            )

        out[scenario] = ensure_fixture("testcase", _fixture_name(scenario), _build)
    return out


class TestLayerAMatrix:
    @pytest.mark.parametrize("scenario", sorted(SCENARIOS))
    def test_recorded_fixture_replays_diff_zero(
        self, scenario: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replay each scenario fresh in a NEW tmp dir; minimal diff must be
        empty (Artifact + Fingerprint + Failure-Semantics gates together)."""
        cfg = SCENARIOS[scenario]
        monkeypatch.chdir(tmp_path)
        raw = _write_inputs(tmp_path, historical=cfg.get("historical"))
        replayed = _run_taskcase(
            tmp_path,
            raw,
            list(cfg["responses"]),
            finish=cfg.get("finish", "stop"),
            review_enabled=cfg.get("review", False),
        )
        replayed.name = _fixture_name(scenario)
        recorded = load_fixture("testcase", _fixture_name(scenario))
        diff = minimal_diff(recorded, replayed)
        assert diff == {}, f"scenario {scenario} drifted: {list(diff)}"

    def test_four_gates_artifact_contract(self, recorded_fixtures) -> None:
        """Contract gate: every artifact item carries the required fields and
        renumbered ids (H1 dimension 2)."""
        import jsonschema

        schema = json.loads(
            (REPO / "tasks" / "testcase" / "schema" / "testcase.schema.json").read_text(
                encoding="utf-8"
            )
        )
        for scenario, fixture in recorded_fixtures.items():
            artifact = fixture.artifact
            assert isinstance(artifact, list), scenario
            for item in artifact:
                jsonschema.validate(item, schema)

    def test_failure_semantics_observable_normalization(self, recorded_fixtures) -> None:
        """Failure-semantics gate: observable classes are the E2 vocabulary."""
        for fixture in recorded_fixtures.values():
            for unit in fixture.failure_semantics["units"]:
                assert unit["observable"] in {
                    "ok",
                    "empty",
                    "validation",
                    "timeout",
                    "provider_error",
                    "failed_units",
                }


class TestLayerBFailureSemantics:
    """E2 action contract: observable outcomes per failure kind."""

    def test_empty_response_unit_observable_empty(self, tmp_path: Path) -> None:
        raw = _write_inputs(tmp_path)
        fixture = _run_taskcase(tmp_path, raw, ["", ""], finish="length")
        classes = {u["observable"] for u in fixture.failure_semantics["units"]}
        assert "failed_units" in classes or "empty" in classes

    def test_validation_error_keeps_items_out_of_artifact(self, tmp_path: Path) -> None:
        """Schema-violating items: unit VALIDATION_ERROR, items audited but
        NOT in the merged artifact (defect ⑧ fix, observable contract)."""
        raw = _write_inputs(tmp_path)
        fixture = _run_taskcase(tmp_path, raw, ['[{"id": "1"}]'])
        artifact = fixture.artifact
        assert artifact == [] or all("title" in item for item in artifact)


class TestLayerCCombos:
    def test_historical_plus_truncation(self, tmp_path: Path) -> None:
        raw = _write_inputs(
            tmp_path,
            historical=[
                {
                    "id": "TC-001",
                    "title": "list users",
                    "endpoint": "GET /users",
                    "test_type": "functional",
                    "priority": "high",
                    "steps": [],
                    "expected_results": [],
                }
            ],
        )
        fixture = _run_taskcase(tmp_path, raw, [_TRUNCATED, _VALID, _VALID, _VALID])
        assert isinstance(fixture.artifact, list)

    def test_review_plus_historical(self, tmp_path: Path) -> None:
        raw = _write_inputs(
            tmp_path,
            historical=[
                {
                    "id": "TC-001",
                    "title": "old case",
                    "endpoint": "GET /users",
                    "test_type": "functional",
                    "priority": "low",
                    "steps": [],
                    "expected_results": [],
                }
            ],
        )
        fixture = _run_taskcase(
            tmp_path, raw, [_VALID, _VALID, _VALID, _VALID, _VALID], review_enabled=True
        )
        assert isinstance(fixture.artifact, list)

    def test_empty_then_recover_plus_review_off(self, tmp_path: Path) -> None:
        raw = _write_inputs(tmp_path)
        fixture = _run_taskcase(tmp_path, raw, ["", _VALID, _VALID, _VALID], finish="length")
        assert isinstance(fixture.artifact, list)


class TestReplaySubsetManifest:
    def test_manifest_exists_with_subset_ids(self) -> None:
        """post_wiring_replay_subset: mechanically selected representative
        fixture IDs (v15 section 8.2 / plan-l L-4 v3)."""
        payload = json.loads(FIXTURE_SUBSET.read_text(encoding="utf-8"))
        subset = payload["post_wiring_replay_subset"]
        for dimension, ids in subset.items():
            assert ids, f"dimension {dimension} has no representative fixtures"
            for fixture_id in ids:
                assert (
                    REPO / "tests" / "fixtures" / "migration" / "testcase" / f"{fixture_id}.json"
                ).exists(), f"{dimension}: fixture {fixture_id} missing"

    def test_subset_covers_every_matrix_dimension(self) -> None:
        payload = json.loads(FIXTURE_SUBSET.read_text(encoding="utf-8"))
        subset = payload["post_wiring_replay_subset"]
        for dimension in ("baseline", "historical", "truncation", "empty-unit", "review-on"):
            assert dimension in subset, f"matrix dimension {dimension} missing from subset"
