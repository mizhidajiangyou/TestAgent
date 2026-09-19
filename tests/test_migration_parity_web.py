"""FH2.6 (plan-k B7.1) web Contract Parity.

The web layer switched its internals from the legacy ``TestCaseGenerator`` to
the task-package pipeline; the EXTERNAL HTTP contract must not change. The
baselines were recorded from the OLD routes BEFORE the switch (git 75680a4
carries that pre-switch state, and ``meta.git_sha`` in each fixture proves it),
so a replay diff is a contract break, not noise.

Twelve cells: success x output format x historical x swagger, plus every error
branch (empty requirements / bad format / missing spec / nothing parseable /
model unavailable / call failure / empty artifact).

E4 normalization (plan-e §E4): only runtime noise is normalized — ``session_id``,
machine-local path prefixes, and the report timestamp. HTTP status, response
schema, error codes, error messages and business fields compare verbatim.

``request_trace`` is empty at this layer on purpose: the point of B7.1 is that
the internal prompt sequence changes (pipeline templates instead of the legacy
builder), so prompt fingerprints are not a contract element.

Adjudicated difference (one cell): ``ok-with-swagger`` returns 2 cases where the
legacy chain returned 4. Legacy only de-duplicated generated cases against
themselves when a historical baseline existed (the dedup lived inside
``_merge_historical_cases``, called only ``if historical_cases:``); the pipeline
applies the ratified H1 dimension ``duplicate_generated: dropped within
new_cases, first kept`` unconditionally. Accepted as post-fix behavior — the
duplicate rows were exactly what the quality gates target.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from testagent.config.models import TestCase
from testagent.engine.llm_client import LLMClient
from testagent.web.app import create_app
from tests.parity_harness import Fixture, bare_settings, ensure_fixture, minimal_diff
from tests.web_fakes import (
    CASES,
    REQUIREMENTS,
    UNPARSEABLE,
    ScriptedLLM,
    make_container,
)

TASK = "web_contract"
_HISTORICAL = [dict(CASES[0], id="TC-900", title="legacy case kept"), {"title": "no endpoint"}]

#: session id + machine-local paths + report timestamp.
_VOLATILE_KEYS = ("session_id",)
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+\d{2}:\d{2})?")
_REPO = Path(__file__).parents[1]


@dataclass
class Scenario:
    """One contract cell: request body + how the fake must behave."""

    name: str
    body: dict[str, Any]
    responses: list[str] | None = None
    fail_verify: bool = False
    fail_call: bool = False
    #: stage a swagger file in tmp and pass its path as swagger_url
    spec_file: bool = False


_SCENARIOS: list[Scenario] = [
    Scenario("ok-json", {"requirements": REQUIREMENTS}),
    Scenario("ok-csv", {"requirements": REQUIREMENTS, "output_format": "csv"}),
    Scenario("ok-markdown", {"requirements": REQUIREMENTS, "output_format": "markdown"}),
    Scenario("ok-with-swagger", {"requirements": REQUIREMENTS}, spec_file=True),
    Scenario("ok-historical", {"requirements": REQUIREMENTS, "historical_cases": _HISTORICAL}),
    Scenario("err-empty-requirements", {"requirements": "   "}),
    Scenario("err-bad-format", {"requirements": REQUIREMENTS, "output_format": "xml"}),
    Scenario("err-swagger-missing", {"requirements": REQUIREMENTS, "swagger_url": "./nope.json"}),
    Scenario("err-nothing-parseable", {"requirements": UNPARSEABLE}),
    Scenario("err-model-unavailable", {"requirements": REQUIREMENTS}, fail_verify=True),
    Scenario("err-call-failed", {"requirements": REQUIREMENTS}, fail_call=True),
    Scenario("err-empty-artifact", {"requirements": REQUIREMENTS}, responses=["[]"] * 12),
]


def _normalize(payload: Any) -> Any:
    """Placeholder-normalize runtime noise only (E4)."""
    if isinstance(payload, dict):
        return {
            key: ("<normalized>" if key in _VOLATILE_KEYS else _normalize(value))
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_normalize(item) for item in payload]
    if isinstance(payload, str):
        return _TIMESTAMP_RE.sub("<timestamp>", payload.replace(str(_REPO), "<repo>"))
    return payload


def _observe(scn: Scenario, tmp: Path) -> Fixture:
    """Drive one request through ``POST /api/generate`` and record what an
    outside observer can see: status + body."""
    body = dict(scn.body)
    recorded_input = dict(scn.body)
    if scn.spec_file:
        from tests.web_fakes import SPEC

        spec = tmp / "spec.json"
        spec.write_text(SPEC, encoding="utf-8")
        body["swagger_url"] = str(spec)
        # the staged path carries the pytest run number; keep it out of the
        # fixture so `input` stays comparable across runs
        recorded_input["swagger_url"] = "<tmp>/spec.json"
    settings = bare_settings(output_dir=str(tmp / "output"))
    llm: LLMClient = ScriptedLLM(
        scn.responses, fail_verify=scn.fail_verify, fail_call=scn.fail_call
    )
    client = TestClient(create_app(container=make_container(llm, settings)))
    response = client.post("/api/generate", json=body)
    is_json = response.headers.get("content-type", "").startswith("application/json")
    observable = {
        "status": response.status_code,
        "body": _normalize(response.json() if is_json else response.text),
    }
    return Fixture(
        name=scn.name,
        task=TASK,
        input=_normalize(recorded_input),
        request_trace=[],
        artifact=observable,
        failure_semantics={
            "status_class": f"{response.status_code // 100}xx",
            "observable": "ok" if response.status_code == 200 else "error",
        },
        events=[],
        meta={"layer": "http-contract", "normalized": list(_VOLATILE_KEYS)},
    )


@pytest.fixture(scope="module")
def baselines(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Fixture]:
    """The committed HTTP contract baselines (recorded only under the opt-in)."""
    out: dict[str, Fixture] = {}
    for scn in _SCENARIOS:

        def _build(scn: Scenario = scn) -> Fixture:
            return _observe(scn, tmp_path_factory.mktemp(f"web-{scn.name}"))

        out[scn.name] = ensure_fixture(TASK, scn.name, _build)
    return out


@pytest.mark.parametrize("scn", _SCENARIOS, ids=[s.name for s in _SCENARIOS])
def test_contract_cell_has_baseline(scn: Scenario, baselines: dict[str, Fixture]) -> None:
    """Every cell must have a baseline — a silently skipped cell is how a
    contract break hides."""
    assert baselines[scn.name].name == scn.name


@pytest.mark.parametrize("scn", _SCENARIOS, ids=[s.name for s in _SCENARIOS])
def test_route_contract_replays_diff_zero(
    scn: Scenario, baselines: dict[str, Fixture], tmp_path: Path
) -> None:
    recorded = baselines[scn.name]
    replayed = _observe(scn, tmp_path)
    replayed.name = scn.name
    diff = minimal_diff(recorded, replayed)
    assert diff == {}, f"{scn.name} drifted: {json.dumps(diff, ensure_ascii=False)[:700]}"


def test_historical_parse_keeps_legacy_tolerance() -> None:
    """``historical_count`` semantics: the converter stays as tolerant as the
    legacy one — ``{"title": ...}`` still yields a case with a placeholder
    endpoint instead of being dropped (verified against
    ``TestCaseGenerator._dict_to_testcase`` while recording the baseline)."""
    from testagent.pipeline.testcase_adapter import dict_to_testcase

    parsed = [tc for tc in (dict_to_testcase(d) for d in _HISTORICAL) if tc]
    assert len(parsed) == len(_HISTORICAL)
    assert isinstance(parsed[0], TestCase)
    assert parsed[1].endpoint.method == "N/A" and parsed[1].endpoint.path == "N/A"
