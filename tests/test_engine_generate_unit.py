"""FH2.1 tests: build_engine_generate_unit — real engine in a pipeline unit.

Three trajectories (plan-k §8.1 DoD) against fake LLMs mirroring the
engine_events scenarios: normal completion, truncated-with-salvage
recovery, budget exhaustion. EngineEvent golden diff=0 is covered by
test_engine_event_baseline.py (the engine itself is untouched).
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from testagent.config.models import APIEndpoint
from testagent.engine.llm_client import LLMResponse
from testagent.pipeline.inputs import TaskContext
from testagent.pipeline.manifest import StageSpec, TruncationSpec
from testagent.pipeline.runtime import build_engine_generate_unit
from testagent.pipeline.status import UnitStatus


class ScriptedLLM:
    """Per-call scripted responses with completion metadata."""

    def __init__(self, responses: list[str], finish: str = "stop") -> None:
        self._responses = list(responses)
        self._finish = finish
        self.calls = 0

    async def achat_with_meta(
        self, system_prompt, user_prompt, response_format=None, max_tokens=None
    ):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return LLMResponse(
            text=text, finish_reason=self._finish, completion_tokens=max_tokens or 10
        )

    async def achat(self, system_prompt, user_prompt, response_format=None, max_tokens=None):
        meta = await self.achat_with_meta(system_prompt, user_prompt)
        return meta.text


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
_VALID_2 = json.dumps(
    [
        {
            "id": "TC-XXX",
            "title": "delete user",
            "endpoint": "DELETE /users/{id}",
            "test_type": "functional",
        }
    ]
)


class _Task:
    """Minimal TaskPackage double: manifest + template rendering."""

    def __init__(self, spec: TruncationSpec | None = None) -> None:
        self.manifest = MagicMock()
        # This double stands for an ENGINE-backed package, and the model
        # default is opt-in False (only packages that declare it recover).
        self.manifest.pipeline.truncation = spec or TruncationSpec(enabled=True)
        self.manifest.artifact.item_schema = None

    @staticmethod
    def system_prompt(template: str, ctx: dict[str, Any]) -> str:
        return "SYS"

    @staticmethod
    def render(template: str, ctx: dict[str, Any]) -> str:
        return "USER"


_STAGE = StageSpec(name="s", template="t", system_prompt="sys")


def _ctx(endpoints: list[APIEndpoint]) -> TaskContext:
    return TaskContext(parsed={"endpoints": endpoints}, settings_views={})


def _unit_ctx(endpoints: list[APIEndpoint]) -> dict[str, Any]:
    return {"_unit_batch": endpoints}


_EPS = [
    APIEndpoint(method="GET", path="/users"),
    APIEndpoint(method="POST", path="/users"),
]

#: Per-call output cap the composition root would inject (runtime never re-reads settings).
_CAP = 16000


@pytest.mark.asyncio
class TestEngineGenerateUnit:
    async def test_normal_trajectory(self) -> None:
        llm = ScriptedLLM([_VALID])
        unit = build_engine_generate_unit(llm, output_token_cap=_CAP)
        result = await unit(_Task(), _STAGE, "batch 1/1", _unit_ctx(_EPS), _ctx(_EPS), "s1")
        assert result.status is UnitStatus.SUCCESS
        assert len(result.items) == 2
        assert llm.calls == 1

    async def test_truncated_salvage_recovers(self) -> None:
        """finish_reason=length with salvageable prefix: the engine salvages
        partial items and CONTINUES (continue/reask path), then completes."""
        truncated = _VALID[:-12]  # cut mid-JSON
        llm = ScriptedLLM([truncated, _VALID_2])
        unit = build_engine_generate_unit(llm, output_token_cap=_CAP)
        result = await unit(_Task(), _STAGE, "batch 1/1", _unit_ctx(_EPS), _ctx(_EPS), "s2")
        assert result.status is UnitStatus.SUCCESS
        assert llm.calls >= 2, "engine must continue after salvage"

    async def test_budget_exhausted_maps_empty(self) -> None:
        """Repeatedly empty responses: engine ladder exhausts -> EMPTY (the
        B6a Outcome mapping table, not a new policy)."""
        llm = ScriptedLLM(["", "", ""], finish="length")
        unit = build_engine_generate_unit(llm, output_token_cap=_CAP)
        result = await unit(_Task(), _STAGE, "batch 1/1", _unit_ctx(_EPS), _ctx(_EPS), "s3")
        assert result.status is UnitStatus.EMPTY

    async def test_disabled_spec_delegates_to_plain_path(self) -> None:
        """TruncationSpec.enabled=false -> single-call contract (rollback)."""
        llm = ScriptedLLM([_VALID])
        unit = build_engine_generate_unit(llm, output_token_cap=_CAP)
        result = await unit(
            _Task(TruncationSpec(enabled=False)), _STAGE, "b", _unit_ctx(_EPS), _ctx(_EPS), "s4"
        )
        assert result.status is UnitStatus.SUCCESS
        assert llm.calls == 1, "disabled spec must NOT run the recovery loop"

    async def test_disabled_spec_records_one_fingerprint(self) -> None:
        """The rollback seam must be byte-identical to the plain path — including
        how many fingerprints it records. Recording the engine fingerprint
        before delegating logged every delegated unit twice."""
        from testagent.pipeline.fingerprint import FingerprintLog

        llm = ScriptedLLM([_VALID])
        unit = build_engine_generate_unit(llm, output_token_cap=_CAP)
        log = FingerprintLog()
        await unit(
            _Task(TruncationSpec(enabled=False)),
            _STAGE,
            "b",
            _unit_ctx(_EPS),
            _ctx(_EPS),
            "s6",
            fingerprint_log=log,
        )
        assert [e.params for e in log.entries] == [{"via": "pipeline"}]

    async def test_schema_violation_maps_validation_error(self) -> None:
        task = _Task()
        task.manifest.artifact.item_schema = {"type": "object", "required": ["id", "title"]}
        llm = ScriptedLLM([json.dumps([{"id": "1"}])])  # missing title
        unit = build_engine_generate_unit(llm, output_token_cap=_CAP)
        result = await unit(task, _STAGE, "b", _unit_ctx(_EPS), _ctx(_EPS), "s5")
        assert result.status is UnitStatus.VALIDATION_ERROR
        assert result.items is not None and result.items[0]["id"] == "1"  # retained for audit
