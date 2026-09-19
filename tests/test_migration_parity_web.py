"""FH2.6 (plan-k B7.1) web Contract Parity.

The web layer switches its internals from the legacy ``TestCaseGenerator`` to
the task-package pipeline; the EXTERNAL HTTP contract must not change. The
baselines below were recorded from the OLD routes BEFORE the switch (golden
discipline: recording predates the change under test, and ``meta.git_sha``
proves it), and every run replays the same twelve cells — success x output
format x historical x swagger, plus every error branch — comparing what an
outside observer can see.

E4 normalization (plan-e §E4): only runtime noise may be normalized — here
``session_id`` and machine-local path prefixes. HTTP status, response schema,
error codes, error messages and business fields are compared verbatim; a diff
in any of them is a contract break, not noise.

``request_trace`` is intentionally empty at this layer: the point of B7.1 is
that the internal prompt sequence changes (pipeline templates instead of the
legacy builder), so prompt fingerprints are not a contract element.
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
from testagent.engine.llm_client import LLMClient, LLMResponse
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import build_engine_generate_unit
from testagent.reports.testcase_report import TestCaseReport
from testagent.web.app import create_app
from tests.parity_harness import Fixture, bare_settings, ensure_fixture, minimal_diff

REPO = Path(__file__).parents[1]
FIXTURE_DIR = Path(__file__).parent / "fixtures" / "migration"
TASK = "web_contract"

#: Requirement text the parser accepts (one requirement, two criteria).
REQUIREMENTS = (
    "# User Management\n"
    "Users can be created, listed, and deleted.\n\n"
    "Acceptance Criteria:\n"
    "- User can register\n"
    "- User can login\n"
)
#: A document that yields no requirements at all (mapped to HTTP 400).
UNPARSEABLE = "　\n---\n"

_CASES = [
    {
        "id": "TC-001",
        "title": "List users returns 200",
        "description": "GET /users with a valid token",
        "endpoint": "GET /users",
        "test_type": "functional",
        "priority": "high",
        "preconditions": ["User is authenticated"],
        "steps": ["Send GET request to /users"],
        "expected_results": ["Status code is 200"],
        "tags": ["smoke"],
    },
    {
        "id": "TC-002",
        "title": "Create user with valid data",
        "description": "POST /users accepts a valid body",
        "endpoint": "POST /users",
        "test_type": "functional",
        "priority": "medium",
        "preconditions": [],
        "steps": ["Send POST request with valid body"],
        "expected_results": ["Status code is 201"],
        "tags": ["crud"],
    },
]
_CASES_JSON = json.dumps(_CASES, ensure_ascii=False)
_HISTORICAL = [dict(_CASES[0], id="TC-900", title="legacy case kept"), {"title": "no endpoint"}]
_SPEC = json.dumps(
    {
        "openapi": "3.0.0",
        "info": {"title": "users", "version": "1"},
        "paths": {
            "/users": {
                "summary": "users",
                "get": {"summary": "List users", "responses": {"200": {"description": "OK"}}},
                "post": {"summary": "Create user", "responses": {"201": {"description": "OK"}}},
            }
        },
    }
)


class _Usage:
    """Deterministic token ledger (the real client's is cumulative, which is
    not reproducible across runs)."""

    def summary(self) -> str:
        return "prompt=10 completion=20 total=30"


class ScriptedLLM(LLMClient):
    """One fake serves both chains: legacy generator (chat/achat) and the
    pipeline (achat_with_meta)."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        fail_verify: bool = False,
        fail_call: bool = False,
    ) -> None:
        self._responses = list(responses) if responses is not None else [_CASES_JSON] * 12
        self._fail_verify = fail_verify
        self._fail_call = fail_call
        self.usage = _Usage()

    def _next(self) -> str:
        if self._fail_call:
            raise RuntimeError("upstream exploded")
        return self._responses.pop(0) if self._responses else "[]"

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):
        return LLMResponse(text=self._next(), finish_reason="stop", completion_tokens=20)

    async def achat(self, system, user, response_format=None, max_tokens=None):
        meta = await self.achat_with_meta(system, user, response_format, max_tokens)
        return meta.text

    def chat(self, system, user, response_format=None, max_tokens=None) -> str:  # type: ignore[override]
        return self._next()

    def chat_with_meta(  # type: ignore[override]
        self, system, user, response_format=None, max_tokens=None
    ) -> LLMResponse:
        return LLMResponse(text=self._next(), finish_reason="stop", completion_tokens=20)

    @property
    def primary_model(self) -> str:
        return "fake-model"

    def set_session_id(self, sid: str) -> None:
        return None

    def verify(self) -> None:
        if self._fail_verify:
            from testagent.engine.llm_client import ModelUnavailableError

            raise ModelUnavailableError("mock: model not found")

    async def averify(self) -> None:
        self.verify()


def make_container(llm: LLMClient, settings: Any) -> Any:
    """Fake container exposing every provider either chain may ask for.

    The legacy provider is built lazily: after B6b.5 deletes the legacy
    generator this module must still import and run (contract fixtures are
    permanent regressions), and by then nothing asks for it.
    """
    from types import SimpleNamespace

    from testagent.parsers.requirement_parser import RequirementParser
    from testagent.parsers.swagger_parser import SwaggerParser

    def legacy_generator() -> Any:
        from testagent.engine.prompt_builder import PromptBuilder
        from testagent.generators.testcase_generator import TestCaseGenerator

        return TestCaseGenerator(llm_client=llm, prompt_builder=PromptBuilder())

    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_engine_generate_unit(
            llm,  # type: ignore[arg-type]
            output_token_cap=settings.llm.max_output_tokens,
        ),
    )
    return SimpleNamespace(
        testcase_generator=legacy_generator,
        pipeline_executor=lambda: executor,
        task_registry=lambda: get_registry(REPO / "tasks"),
        requirement_parser=lambda: RequirementParser(),
        swagger_parser=lambda: SwaggerParser(),
        testcase_report=lambda: TestCaseReport(),
        llm_client=lambda: llm,
        settings=lambda: settings,
    )


@dataclass
class Scenario:
    """One contract cell: request body + how the fake must behave."""

    name: str
    body: dict[str, Any]
    responses: list[str] | None = None
    fail_verify: bool = False
    fail_call: bool = False
    #: "spec" → stage a swagger file in tmp and pass its path as swagger_url.
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

#: E4 whitelist in force here: session id, machine-local paths, timestamps
#: (the markdown report stamps its generation date).
_VOLATILE_KEYS = ("session_id",)
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\+\d{2}:\d{2})?")


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
        return _TIMESTAMP_RE.sub("<timestamp>", payload.replace(str(REPO), "<repo>"))
    return payload


def _observe(scn: Scenario, tmp: Path) -> Fixture:
    """Drive one request through ``POST /api/generate`` and record what an
    outside observer can see: status + body."""
    body = dict(scn.body)
    recorded_input = dict(scn.body)
    if scn.spec_file:
        spec = tmp / "spec.json"
        spec.write_text(_SPEC, encoding="utf-8")
        body["swagger_url"] = str(spec)
        # The staged path carries the pytest run number; keep the real value
        # out of the fixture so `input` is comparable across runs.
        recorded_input["swagger_url"] = "<tmp>/spec.json"
    settings = bare_settings(output_dir=str(tmp / "output"))
    llm = ScriptedLLM(scn.responses, fail_verify=scn.fail_verify, fail_call=scn.fail_call)
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
    assert (FIXTURE_DIR / TASK / f"{scn.name}.json").exists(), scn.name


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
    """``historical_count`` semantics: the converter must stay as tolerant as
    the legacy one — ``{"title": ...}`` still yields a case with a placeholder
    endpoint instead of being dropped (verified against
    ``TestCaseGenerator._dict_to_testcase`` while recording the baseline)."""
    from testagent.pipeline.testcase_adapter import dict_to_testcase

    parsed = [tc for tc in (dict_to_testcase(d) for d in _HISTORICAL) if tc]
    assert len(parsed) == len(_HISTORICAL)
    assert isinstance(parsed[0], TestCase)
    assert parsed[1].endpoint.method == "N/A" and parsed[1].endpoint.path == "N/A"
