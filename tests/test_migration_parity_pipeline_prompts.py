"""FH2.8 prerequisite gate: the new chain must send the SAME prompts as legacy.

Why this file exists: FH2.4 replayed the new chain against a scripted LLM, and
a scripted LLM answers whatever it is asked. Artifact-level equivalence
(``test_quality_chain_equivalence.py``) has the same blind spot. So a prompt
that stopped carrying its material could stay green forever - which is exactly
what happened: ``{{ endpoints }}`` / ``{{ requirements }}`` are STRINGS in the
legacy builders and were rendered from OBJECTS on the task-package chain, and
Phase 2 received the run-wide endpoint list in every batch instead of its own.

The gate therefore compares the actual ``(system, user)`` byte pairs each chain
sends, per unit and in order, for the four surfaces that can drift:
plain, output language, JSON mode and the historical baseline.
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

import pytest

from testagent.config.models import TestCaseGenInput
from testagent.engine.llm_client import LLMResponse
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.parsers.swagger_parser import SwaggerParser
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
        },
        "/users/{id}": {"delete": {"tags": ["users"], "responses": {"204": {"description": "OK"}}}},
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


class _RecordingLLM(ScriptedLLM):
    """One fake for both chains: records every prompt on every entry point and
    answers with one case for the FIRST endpoint the prompt shows.

    Prompt-aware (rather than scripted) answers are what makes this a prompt
    comparison at all: a canned list runs out, the truncation engine then
    re-asks, and the call sequences no longer line up unit-for-unit.
    """

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
        return json.dumps([_case(f"case for {endpoint}", endpoint)])

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


def _legacy_prompts(tmp: Path, *, language: str, json_mode: bool, historical: bool) -> list[Any]:
    _write_inputs(tmp)
    llm = _RecordingLLM()
    generator = TestCaseGenerator(
        llm_client=llm,
        prompt_builder=PromptBuilder(),
        output_language=language,
        json_mode=json_mode,
        max_concurrency=1,
        verify_model=False,
    )
    historical_cases = []
    if historical:
        from testagent.pipeline.testcase_adapter import dict_to_testcase

        tc = dict_to_testcase(_case("old case", "GET /users"))
        historical_cases = [tc] if tc else []
    generator.generate(
        TestCaseGenInput(
            endpoints=SwaggerParser().parse(str(tmp / "spec.json")),
            requirements=RequirementParser().parse(str(tmp / "req.md")),
            historical_cases=historical_cases,
        ),
        session_id="legacy-prompts",
    )
    return llm.prompts


def _pipeline_prompts(tmp: Path, *, language: str, json_mode: bool, historical: bool) -> list[Any]:
    _write_inputs(tmp)
    settings = bare_settings(
        output_language=language,
        output_dir=str(tmp / "output"),
        llm=bare_settings().llm.model_copy(update={"json_mode": json_mode, "max_concurrency": 1}),
    )
    llm = _RecordingLLM()
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
    if historical:
        (tmp / "hist.json").write_text(
            json.dumps([_case("old case", "GET /users")]), encoding="utf-8"
        )
        raw["historical_cases"] = "hist.json"
    with _in_dir(tmp):
        ctx = parse_inputs(task.manifest, raw, settings)
        asyncio.run(executor.arun(task, ctx, session_id="pipeline-prompts"))
    return llm.prompts


def _diff(legacy: list[Any], pipeline: list[Any]) -> str:
    lines: list[str] = []
    if len(legacy) != len(pipeline):
        lines.append(f"call count: legacy {len(legacy)} pipeline {len(pipeline)}")
    for i, (sys_l, usr_l) in enumerate(legacy):
        if i >= len(pipeline):
            lines.append(f"#{i}: pipeline sent no call")
            continue
        sys_p, usr_p = pipeline[i]
        if sys_l != sys_p:
            lines.append(f"#{i} SYSTEM differs\n  legacy  : {sys_l!r}\n  pipeline: {sys_p!r}")
        if usr_l != usr_p:
            start = next(
                (j for j in range(min(len(usr_l), len(usr_p))) if usr_l[j] != usr_p[j]),
                min(len(usr_l), len(usr_p)),
            )
            lines.append(
                f"#{i} USER differs at char {start}\n"
                f"  legacy  : {usr_l[start : start + 220]!r}\n"
                f"  pipeline: {usr_p[start : start + 220]!r}"
            )
    return "\n".join(lines)


SCENARIOS: list[dict[str, Any]] = [
    {"name": "plain", "language": "english", "json_mode": False, "historical": False},
    {"name": "chinese", "language": "chinese", "json_mode": False, "historical": False},
    {"name": "json-mode", "language": "english", "json_mode": True, "historical": False},
    {"name": "historical", "language": "english", "json_mode": False, "historical": True},
]


def test_new_chain_sends_legacy_prompts(tmp_path: Path) -> None:
    failures: list[str] = []
    for scn in SCENARIOS:
        kwargs = {k: v for k, v in scn.items() if k != "name"}
        legacy = _legacy_prompts(tmp_path / f"legacy-{scn['name']}", **kwargs)
        pipeline = _pipeline_prompts(tmp_path / f"pipeline-{scn['name']}", **kwargs)
        diffs = _diff(legacy, pipeline)
        if diffs:
            failures.append(f"[{scn['name']}]\n{diffs}")
    assert not failures, "prompt drift between the legacy chain and tasks/testcase:\n" + (
        "\n".join(failures)
    )


def test_phase2_batches_are_scoped_to_their_own_endpoints(tmp_path: Path) -> None:
    """The defect this gate was written for: every Phase 2 unit must see ONLY
    its own endpoints, and they must be a signature - not a dataclass repr."""
    prompts = _pipeline_prompts(
        tmp_path / "scoped", language="english", json_mode=False, historical=False
    )
    phase2 = [
        user for _, user in prompts if "This is PHASE 2" in user or "Generate API-specific" in user
    ]
    assert len(phase2) > 1, f"expected several Phase 2 batches, got {len(phase2)}"
    assert len({user for user in phase2}) == len(phase2), "all Phase 2 batches rendered alike"
    for user in phase2:
        assert "APIEndpoint(" not in user, "endpoint list reached the prompt as a Python repr"
        assert "RequirementItem(" not in user, "requirements reached the prompt as a repr"


# ----------------------------------------------------------------------
# The other two packages: their templates spell the same variables, but the
# legacy builders fed them the PLAIN endpoint text, so the mapping is per
# package (manifest.prompt_views) rather than one global guess.
# ----------------------------------------------------------------------

_SCRIPT = "import pytest\n\n\ndef test_flow(page) -> None:\n    assert page\n"


def _text_legacy_prompts(
    module: str, tmp: Path, raw: dict[str, Any], *, review: bool = False
) -> list[Any]:
    _write_inputs(tmp)
    language = raw.get("output_language", "english")
    llm = _RecordingLLM([_SCRIPT] * 6)
    builder = PromptBuilder()
    if module == "gui":
        from testagent.config.models import GUITestGenInput
        from testagent.generators.gui_test_generator import GUITestGenerator

        generator = GUITestGenerator(
            llm_client=llm,
            prompt_builder=builder,
            output_language=language,
            review_enabled=review,
        )
        generator.generate(
            GUITestGenInput(
                requirements=RequirementParser().parse(str(tmp / "req.md")),
                endpoints=SwaggerParser().parse(str(tmp / "spec.json")),
                url="https://x.test",
                # The GUI chain takes the language per request (the CLI flag),
                # which overrides the constructor default — mirror that, or the
                # comparison silently checks two different runs.
                output_language=language,
            )
        )
    else:
        from testagent.config.models import PerfGenInput, PerformanceConfig
        from testagent.generators.performance_generator import PerformanceGenerator

        generator = PerformanceGenerator(
            llm_client=llm,
            prompt_builder=builder,
            output_language=language,
            review_enabled=review,
        )
        generator.generate(
            PerfGenInput(
                endpoints=SwaggerParser().parse(str(tmp / "spec.json")),
                config=PerformanceConfig(base_url="https://x.test"),
            )
        )
    return llm.prompts


def _text_pipeline_prompts(
    module: str, tmp: Path, raw: dict[str, Any], *, review: bool = False
) -> list[Any]:
    _write_inputs(tmp)
    settings = bare_settings(
        output_language=raw.get("output_language", "english"),
        output_dir=str(tmp / "output"),
        review_enabled=review,
    )
    llm = _RecordingLLM([_SCRIPT] * 6)
    task = get_registry(REPO / "tasks").get(module)
    executor = PipelineExecutor(
        llm,  # type: ignore[arg-type]
        settings,
        generate_unit=build_generate_unit(llm),  # type: ignore[arg-type]
        review_runner=build_review_runner(llm) if review else None,
    )
    with _in_dir(tmp):
        ctx = parse_inputs(task.manifest, raw, settings)
        asyncio.run(executor.arun(task, ctx, session_id=f"{module}-prompts"))
    return llm.prompts


def test_requirements_only_prompt_carries_no_validation_sample(tmp_path: Path) -> None:
    """A render must never fall back to the ``tasks validate`` sample.

    ``template_context`` exists so validation can smoke-render every template;
    if a production run reaches for it, the prompt claims an API the user never
    supplied (here: ``GET /users - list`` inside the "API Context" block), and
    the model then writes cases for that phantom spec.
    """
    prompts = _text_pipeline_prompts(
        "gui", tmp_path / "no-spec", {"requirements": "req.md", "url": "https://x.test"}
    )
    assert prompts, "the run produced no calls at all, so the check would be vacuous"
    joined = "\n".join(f"{system}\n{user}" for system, user in prompts)
    assert "GET /users - list" not in joined, "validation sample reached a production prompt"
    assert "## API Context" not in joined, "an endpoint block rendered for a run with no spec"


@pytest.mark.parametrize(
    ("module", "raw"),
    [
        (
            "gui",
            {
                "requirements": "req.md",
                "swagger": "spec.json",
                "url": "https://x.test",
                "output_language": "english",
            },
        ),
        ("perf", {"swagger": "spec.json", "base_url": "https://x.test"}),
    ],
)
def test_text_packages_send_legacy_prompts(
    module: str, raw: dict[str, Any], tmp_path: Path
) -> None:
    legacy = _text_legacy_prompts(module, tmp_path / f"legacy-{module}", raw)
    pipeline = _text_pipeline_prompts(module, tmp_path / f"pipeline-{module}", raw)
    diffs = _diff(legacy, pipeline)
    assert not diffs, f"{module} prompt drift:\n{diffs}"


@pytest.mark.parametrize(
    ("module", "raw"),
    [
        (
            "gui",
            {
                "requirements": "req.md",
                "swagger": "spec.json",
                "url": "https://x.test",
                "output_language": "chinese",
            },
        ),
        (
            "perf",
            {"swagger": "spec.json", "base_url": "https://x.test", "output_language": "chinese"},
        ),
    ],
)
def test_script_packages_send_legacy_prompts_in_chinese(
    module: str, raw: dict[str, Any], tmp_path: Path
) -> None:
    """The language hint is part of the prompt contract, not a nicety."""
    legacy = _text_legacy_prompts(module, tmp_path / f"legacy-{module}", raw)
    pipeline = _text_pipeline_prompts(module, tmp_path / f"pipeline-{module}", raw)
    assert any("Simplified Chinese" in system for system, _ in pipeline), (
        "the package prompt lost the output-language hint"
    )
    diffs = _diff(legacy, pipeline)
    assert not diffs, f"{module} chinese prompt drift:\n{diffs}"


@pytest.mark.parametrize("module", ["gui", "perf"])
def test_script_review_prompts_match_the_legacy_builder(module: str, tmp_path: Path) -> None:
    """The script review pass is the last prompt the legacy chain owns for the
    text packages; ``tasks/gui`` gained its review stage from this reading."""
    raw = (
        {"requirements": "req.md", "swagger": "spec.json", "url": "https://x.test"}
        if module == "gui"
        else {"swagger": "spec.json", "base_url": "https://x.test"}
    )
    legacy = _text_legacy_prompts(module, tmp_path / f"legacy-{module}-rev", raw, review=True)
    pipeline = _text_pipeline_prompts(module, tmp_path / f"pipeline-{module}-rev", raw, review=True)
    assert len(legacy) >= 2, f"the legacy {module} chain ran no review call"
    assert len(pipeline) >= 2, f"the {module} package ran no review call"
    diffs = _diff(legacy, pipeline)
    assert not diffs, f"{module} review prompt drift:\n{diffs}"
