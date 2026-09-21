"""Shared web-layer test doubles.

Both web test suites need the same thing: an app wired to a container whose LLM
never leaves the process. ``tests/test_migration_parity_web.py`` (HTTP contract
fixtures) and ``tests/test_web_app.py`` (route behavior) must not maintain two
copies of that double — the parity cells would then prove a chain the other
suite is not exercising.

The container exposes providers for BOTH chains on purpose: the contract
baselines were recorded against the legacy generator (that is what
``meta.git_sha`` in the fixtures proves), while the routes now run on the task
pipeline. After B6b.5 deletes the legacy generator nothing asks for that
provider, and the lazy factory simply never fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from testagent.engine.llm_client import LLMClient, LLMResponse
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import build_engine_generate_unit
from testagent.reports.testcase_report import TestCaseReport

REPO = Path(__file__).parents[1]

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

CASES = [
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
CASES_JSON = json.dumps(CASES, ensure_ascii=False)

SPEC = json.dumps(
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
    pipeline (achat_with_meta).

    ``fail_verify`` / ``fail_call`` exist because the web error branches are
    part of the recorded contract — a test must be able to drive them without
    reaching into private attributes of a wrapped object.
    """

    #: Both chains pre-flight the model; the base default would hide the 400
    #: branch (the real client sets this too).
    intent_capable = True

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        fail_verify: bool = False,
        fail_call: bool = False,
    ) -> None:
        self._responses = list(responses) if responses is not None else [CASES_JSON] * 12
        self._fail_verify = fail_verify
        self._fail_call = fail_call
        self.usage = _Usage()
        self.prompts: list[tuple[str, str]] = []

    def _next(self) -> str:
        if self._fail_call:
            raise RuntimeError("upstream exploded")
        return self._responses.pop(0) if self._responses else "[]"

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):
        self.prompts.append((system, user))
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


def make_container(llm: LLMClient, settings: Any, *, tasks_dir: Path = REPO / "tasks") -> Any:
    """Fake container with every provider either chain may ask for."""
    from testagent.parsers.requirement_parser import RequirementParser
    from testagent.parsers.swagger_parser import SwaggerParser

    def legacy_generator() -> Any:
        from testagent.engine.prompt_builder import PromptBuilder
        from testagent.generators.testcase_generator import TestCaseGenerator

        return TestCaseGenerator(llm_client=llm, prompt_builder=PromptBuilder())

    from testagent.parsers.swagger_parser import endpoints_to_rich_signature

    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_engine_generate_unit(
            llm,  # type: ignore[arg-type]
            output_token_cap=settings.llm.max_output_tokens,
        ),
        # Same wiring as the production container, so a links run here measures
        # the same L0 index the CLI run would.
        links_signature_fn=endpoints_to_rich_signature,
    )
    return SimpleNamespace(
        testcase_generator=legacy_generator,
        pipeline_executor=lambda: executor,
        task_registry=lambda: get_registry(tasks_dir),
        requirement_parser=lambda: RequirementParser(),
        swagger_parser=lambda: SwaggerParser(),
        testcase_report=lambda: TestCaseReport(),
        llm_client=lambda: llm,
        settings=lambda: settings,
    )
