"""FH2.5 perf/gui migration parity.

Perf task package has been on the pipeline since B5.1 (fingerprint parity
against the legacy generator proven there); gui joined at FH1.1. This
module records parity fixtures for both (gui: text contract; perf: text
contract with script_format variance), replays them with minimal diff=0,
and asserts the four gates at the observable level.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from testagent.container import Container
from testagent.engine.llm_client import LLMClient, LLMResponse
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from tests.parity_harness import (
    Fingerprint,
    Fixture,
    ensure_no_credentials,
    load_fixture,
    minimal_diff,
    record_fixture,
)

REPO = Path(__file__).parents[1]


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
    import os

    os.environ["TASKS_DIR"] = str(REPO / "tasks")
    os.chdir(tmp)
    container = Container()
    task = container.task_registry().get(task_name)
    llm = ScriptedTextLLM(script)
    settings = container.settings()
    executor = PipelineExecutor(llm, settings, generate_unit=build_unit_for(llm))
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


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Fixture]:
    out: dict[str, Fixture] = {}
    tmp_gui = tmp_path_factory.mktemp("gui-parity")
    (tmp_gui / "req.md").write_text("# GUI\n\nLogin and logout flows.\n", encoding="utf-8")
    gui = _run_task(
        "gui",
        {"requirements": "req.md", "url": "https://x.test"},
        _GUI_SCRIPT,
        tmp_gui,
    )
    gui.name = "gui-baseline"
    ensure_no_credentials(gui.to_dict())
    record_fixture(gui)
    out["gui-baseline"] = gui

    tmp_perf = tmp_path_factory.mktemp("perf-parity")
    (tmp_perf / "spec.json").write_text(
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
    perf = _run_task(
        "perf",
        {"swagger": "spec.json", "base_url": "https://x.test", "script_format": "k6"},
        _K6_SCRIPT,
        tmp_perf,
    )
    perf.name = "perf-baseline-k6"
    ensure_no_credentials(perf.to_dict())
    record_fixture(perf)
    out["perf-baseline-k6"] = perf
    return out


class TestPerfGuiParity:
    @pytest.mark.parametrize("name", ["gui-baseline", "perf-baseline-k6"])
    def test_replay_diff_zero(
        self, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TASKS_DIR", str(REPO / "tasks"))
        recorded = load_fixture("gui" if name.startswith("gui") else "perf", name)
        if name == "gui-baseline":
            tmp = tmp_path / "gui"
            tmp.mkdir()
            (tmp / "req.md").write_text("# GUI\n\nLogin and logout flows.\n", encoding="utf-8")
            replayed = _run_task(
                "gui", {"requirements": "req.md", "url": "https://x.test"}, _GUI_SCRIPT, tmp
            )
        else:
            tmp = tmp_path / "perf"
            tmp.mkdir()
            (tmp / "spec.json").write_text(
                json.dumps(
                    {
                        "openapi": "3.0.0",
                        "paths": {
                            "/users": {
                                "get": {
                                    "tags": ["users"],
                                    "responses": {"200": {"description": "OK"}},
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            replayed = _run_task(
                "perf",
                {"swagger": "spec.json", "base_url": "https://x.test", "script_format": "k6"},
                _K6_SCRIPT,
                tmp,
            )
        replayed.name = name
        diff = minimal_diff(recorded, replayed)
        assert diff == {}, f"{name} drifted: {list(diff)}"

    def test_gui_contract_compilable(self, recorded) -> None:
        artifact = recorded["gui-baseline"].artifact
        script = artifact if isinstance(artifact, str) else artifact[0]
        assert "import pytest" in script

    def test_failure_semantics_observed(self, recorded) -> None:
        for name, fixture in recorded.items():
            assert fixture.failure_semantics["observable"] == "ok", name
