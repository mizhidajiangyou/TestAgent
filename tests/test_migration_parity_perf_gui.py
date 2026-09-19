"""FH2.5 perf/gui migration parity.

Perf task package has been on the pipeline since B5.1 (fingerprint parity
against the legacy generator proven there); gui joined at FH1.1. This
module records parity fixtures for both (gui: text contract; perf: text
contract with script_format variance), replays them with minimal diff=0,
and asserts the four gates at the observable level.
"""

import asyncio
import json
import os
from collections.abc import Callable, Iterator
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
)

REPO = Path(__file__).parents[1]


@contextmanager
def _in_dir(tmp: Path) -> Iterator[None]:
    """Scope a cwd change so a leak can't poison later tests."""
    previous = Path.cwd()
    os.chdir(tmp)
    try:
        yield
    finally:
        os.chdir(previous)


class ScriptedTextLLM(LLMClient):
    """Deterministic fake emitting a fixed script."""

    def __init__(self, script: str) -> None:
        self._script = script
        self.calls: list[tuple[str, str]] = []
        self.fingerprints: list[dict[str, Any]] = []
        self.responses_served: list[str] = []

    async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):
        self.calls.append((system, user))
        self.responses_served.append(self._script)
        self.fingerprints.append(
            Fingerprint.of(
                model="fake-model",
                system_prompt=system,
                user_prompt=user,
                params={},
                label="",
            ).to_dict()
        )
        return LLMResponse(text=self._script, finish_reason="stop", completion_tokens=50)

    async def achat(self, system, user, response_format=None, max_tokens=None):
        meta = await self.achat_with_meta(system, user)
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


_GUI_SCRIPT = (
    "import pytest\nfrom playwright.sync_api import Page, expect\n\n\n"
    "def test_home(page: Page) -> None:\n    page.goto('https://x.test')\n"
    "    expect(page.get_by_role('heading')).to_be_visible()\n"
)
_K6_SCRIPT = (
    "import http from 'k6/http';\nexport const options = {vus: 1, duration: '1s'};\n"
    "export default function () { http.get('https://x.test/users'); }\n"
)


def _run_task(task_name: str, raw: dict[str, str], script: str, tmp: Path) -> Fixture:
    """Deterministic by construction: absolute registry path + settings that
    never read the developer's ``.env`` (a baseline that varies with local env
    is not a baseline, and CI would drift from the recording machine)."""
    task = get_registry(REPO / "tasks").get(task_name)
    llm = ScriptedTextLLM(script)
    settings = bare_settings()
    executor = PipelineExecutor(llm, settings, generate_unit=build_unit_for(llm))
    with _in_dir(tmp):
        ctx = parse_inputs(task.manifest, raw, settings)
        result = asyncio.run(executor.arun(task, ctx, session_id="parity-fixed"))
    response_texts = list(llm.responses_served)
    return Fixture(
        name="placeholder",
        task=task_name,
        input={
            k: Path(v).name
            if isinstance(v, str) and ("/" in v or v.endswith((".md", ".json")))
            else v
            for k, v in raw.items()
        },
        request_trace=[
            {**fp, "response_text": response_texts[i] if i < len(response_texts) else ""}
            for i, fp in enumerate(llm.fingerprints)
        ],
        artifact=result.artifact,
        failure_semantics={
            "units_failed": result.units_failed,
            "observable": "ok" if result.units_failed == 0 else "failed_units",
        },
        events=[],
        meta={"task": task_name},
    )


def build_unit_for(llm: Any):
    from testagent.pipeline.runtime import build_generate_unit

    return build_generate_unit(llm)


def _gui_inputs(tmp: Path) -> dict[str, str]:
    (tmp / "req.md").write_text("# GUI\n\nLogin and logout flows.\n", encoding="utf-8")
    return {"requirements": "req.md", "url": "https://x.test"}


def _perf_inputs(tmp: Path) -> dict[str, str]:
    (tmp / "spec.json").write_text(
        json.dumps(
            {
                "openapi": "3.0.0",
                "paths": {
                    "/users": {
                        "get": {"tags": ["users"], "responses": {"200": {"description": "OK"}}}
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return {"swagger": "spec.json", "base_url": "https://x.test", "script_format": "k6"}


#: fixture name -> (task package, scripted LLM output, input writer). The same
#: builder runs for recording and for replay, so a "drift" can only come from
#: the code under test.
SCENARIOS: dict[str, tuple[str, str, Callable[[Path], dict[str, str]]]] = {
    "gui-baseline": ("gui", _GUI_SCRIPT, _gui_inputs),
    "perf-baseline-k6": ("perf", _K6_SCRIPT, _perf_inputs),
}


def _run_scenario(name: str, tmp: Path) -> Fixture:
    task_name, script, write_inputs = SCENARIOS[name]
    fixture = _run_task(task_name, write_inputs(tmp), script, tmp)
    fixture.name = name
    return fixture


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Fixture]:
    out: dict[str, Fixture] = {}
    for name, (task_name, _, _) in SCENARIOS.items():

        def _build(name: str = name) -> Fixture:
            return _run_scenario(name, tmp_path_factory.mktemp(f"{name}-parity"))

        out[name] = ensure_fixture(task_name, name, _build)
    return out


class TestPerfGuiParity:
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    def test_replay_diff_zero(self, name: str, tmp_path: Path) -> None:
        recorded = load_fixture(SCENARIOS[name][0], name)
        diff = minimal_diff(recorded, _run_scenario(name, tmp_path))
        assert diff == {}, f"{name} drifted: {list(diff)}"

    def test_gui_contract_compilable(self, recorded) -> None:
        artifact = recorded["gui-baseline"].artifact
        script = artifact if isinstance(artifact, str) else artifact[0]
        compile(script, "<gui-baseline>", "exec")

    def test_failure_semantics_observed(self, recorded) -> None:
        for name, fixture in recorded.items():
            assert fixture.failure_semantics["observable"] == "ok", name
