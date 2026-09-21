"""Prompt capture shared by the two prompt-fidelity gates.

Deliberately free of any legacy-chain import: `test_prompt_oracle.py` replays
the recorded oracle against the NEW chain only, and that is the gate which has
to survive plan-k B6b.5/B7.3 deleting the legacy generators. The live
legacy-vs-new comparison lives in
``tests/test_migration_parity_pipeline_prompts.py`` and dies with the legacy
code it measures.

The fake answers *from the prompt* rather than from a script because a
comparison of prompt sequences only works when both chains make the same number
of calls: a canned list runs out, the truncation engine re-asks, and the call
lists no longer line up unit for unit.
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

from testagent.engine.llm_client import LLMResponse
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.registry import get_registry
from testagent.pipeline.runtime import (
    build_engine_generate_unit,
    build_generate_unit,
    build_review_runner,
)
from tests.parity_harness import bare_settings
from tests.web_fakes import ScriptedLLM

REPO = Path(__file__).parents[1]

REQUIREMENTS_MD = (
    "# User Management\n"
    "Users can be listed and created.\n\n"
    "Acceptance Criteria:\n"
    "- User can register\n"
    "- User can login\n"
)
SPEC = {
    "openapi": "3.0.0",
    "paths": {
        "/users": {
            "get": {"tags": ["users"], "responses": {"200": {"description": "OK"}}},
            "post": {"tags": ["users"], "responses": {"201": {"description": "OK"}}},
        },
        "/users/{id}": {"delete": {"tags": ["users"], "responses": {"204": {"description": "OK"}}}},
    },
}
#: Playwright-style script used as the answer for the text tasks.
SCRIPT = "import pytest\n\n\ndef test_flow(page: Page) -> None:\n    assert page\n"

#: Every scenario the oracle pins: the four testcase surfaces plus the two text
#: packages in both languages and with/without the review pass.
CASE_SCENARIOS: list[dict[str, Any]] = [
    {"name": "testcase-plain", "language": "english", "json_mode": False, "historical": False},
    {"name": "testcase-chinese", "language": "chinese", "json_mode": False, "historical": False},
    {"name": "testcase-json-mode", "language": "english", "json_mode": True, "historical": False},
    {"name": "testcase-historical", "language": "english", "json_mode": False, "historical": True},
]
TEXT_SCENARIOS: list[dict[str, Any]] = [
    {"name": "gui-plain", "module": "gui", "language": "english", "review": False},
    {"name": "gui-chinese", "module": "gui", "language": "chinese", "review": False},
    {"name": "gui-review", "module": "gui", "language": "english", "review": True},
    {"name": "perf-plain", "module": "perf", "language": "english", "review": False},
    {"name": "perf-chinese", "module": "perf", "language": "chinese", "review": False},
    {"name": "perf-review", "module": "perf", "language": "english", "review": True},
]


def case(title: str, endpoint: str) -> dict[str, Any]:
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


class RecordingLLM(ScriptedLLM):
    """Records the ``(system, user)`` pair of every call on every entry point."""

    _ENDPOINT_LINE = re.compile(r"^- (GET|POST|PUT|PATCH|DELETE) (\S+)", re.MULTILINE)

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(responses or [])
        #: With a script, every call answers that script verbatim (text tasks);
        #: without one, the answer is derived from the prompt (case tasks).
        self._fixed = list(responses or [])

    def _respond(self, user: str) -> str:
        if self._fixed:
            return self._fixed.pop(0)
        match = self._ENDPOINT_LINE.search(user)
        endpoint = f"{match.group(1)} {match.group(2)}" if match else "N/A"
        return json.dumps([case(f"case for {endpoint}", endpoint)])

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):  # type: ignore[override]
        self.prompts.append((system, user))
        return LLMResponse(text=self._respond(user), finish_reason="stop", completion_tokens=20)

    async def achat(self, system, user, response_format=None, max_tokens=None):  # type: ignore[override]
        meta = await self.achat_with_meta(system, user, response_format, max_tokens)
        return meta.text

    def chat(self, system, user, response_format=None, max_tokens=None) -> str:  # type: ignore[override]
        self.prompts.append((system, user))
        return self._respond(user)

    def chat_with_meta(  # type: ignore[override]
        self, system, user, response_format=None, max_tokens=None
    ) -> LLMResponse:
        self.prompts.append((system, user))
        return LLMResponse(text=self._respond(user), finish_reason="stop", completion_tokens=20)


@contextlib.contextmanager
def in_dir(tmp: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(previous)


def write_inputs(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "req.md").write_text(REQUIREMENTS_MD, encoding="utf-8")
    (tmp / "spec.json").write_text(json.dumps(SPEC), encoding="utf-8")


def raw_for(module: str, *, language: str, review: bool) -> dict[str, Any]:
    """Task-package input values for one text scenario (paths are staged by
    :func:`write_inputs`, mirroring how the CLI hands files to the parsers)."""
    del review
    if module == "gui":
        return {
            "requirements": "req.md",
            "swagger": "spec.json",
            "url": "https://x.test",
            "output_language": language,
        }
    return {"swagger": "spec.json", "base_url": "https://x.test", "output_language": language}


def new_case_prompts(
    tmp: Path, *, language: str, json_mode: bool, historical: bool, links: bool = False
) -> list[tuple[str, str]]:
    """Run ``tasks/testcase`` on the new chain and return its prompt pairs."""
    write_inputs(tmp)
    settings = bare_settings(
        output_language=language,
        output_dir=str(tmp / "output"),
        links_enabled=links,
        llm=bare_settings().llm.model_copy(update={"json_mode": json_mode, "max_concurrency": 1}),
    )
    llm = RecordingLLM()
    task = get_registry(REPO / "tasks").get("testcase")
    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_engine_generate_unit(
            llm,  # type: ignore[arg-type]
            output_token_cap=settings.llm.max_output_tokens,
        ),
    )
    raw: dict[str, Any] = {"requirements": "req.md", "swagger": "spec.json"}
    if links:
        raw["links"] = True
    if historical:
        (tmp / "hist.json").write_text(
            json.dumps([case("old case", "GET /users")]), encoding="utf-8"
        )
        raw["historical_cases"] = "hist.json"
    with in_dir(tmp):
        ctx = parse_inputs(task.manifest, raw, settings)
        asyncio.run(executor.arun(task, ctx, session_id="oracle-prompts"))
    return llm.prompts


def new_text_prompts(
    tmp: Path,
    *,
    module: str,
    language: str,
    review: bool,
    raw: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    """Run ``tasks/gui`` / ``tasks/perf`` (single-call path, text contract)."""
    write_inputs(tmp)
    settings = bare_settings(
        output_language=language,
        output_dir=str(tmp / "output"),
        review_enabled=review,
    )
    llm = RecordingLLM([SCRIPT] * 6)
    task = get_registry(REPO / "tasks").get(module)
    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_generate_unit(llm),  # type: ignore[arg-type]
        review_runner=build_review_runner(llm) if review else None,
    )
    with in_dir(tmp):
        ctx = parse_inputs(
            task.manifest,
            raw if raw is not None else raw_for(module, language=language, review=review),
            settings,
        )
        asyncio.run(executor.arun(task, ctx, session_id=f"oracle-{module}"))
    return llm.prompts
