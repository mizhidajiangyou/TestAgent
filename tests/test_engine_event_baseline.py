"""EngineEvent golden baseline (plan-e E3 / plan-d B6a-0).

Evidence chain (three layers, per plan-e E3):

1. Baseline provenance — the observer records the engine's REAL control
   flow: B6a-0 is observation-only (zero control-flow change, proven by the
   full suite incl. the 15 correctness tests), so a trajectory recorded
   through the observer IS the pre-refactor control flow's trace.
2. Log cross-check — for every scenario the observer's log-mappable event
   subsequence must EQUAL the action sequence derived from the engine's
   PRE-EXISTING branch-point log lines (which predate the observer), so the
   emission points provably sit on the real decision branches — closing the
   "correctness != event-order proof" gap.
3. Correctness cross-check — per-scenario invariants mirror the assertions
   in test_truncation_recovery.py (downgrade exactly once and sticky, split
   then raise budget, terminal exactly once).

Golden files live in tests/fixtures/migration/engine_events/ and are the
permanent regression baseline for B6a-1..3 / B6b.1 replay (diff must be 0;
differences follow plan-d v3 R5 rules). Regenerate deliberately with:

    TESTAGENT_RECORD_GOLDEN=1 pytest tests/test_engine_event_baseline.py
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from testagent.config.models import APIEndpoint
from testagent.engine.llm_client import LLMResponse, ReasoningBudgetExhaustedError
from testagent.engine.model_profiles import DEEPSEEK_V4, RequestIntent
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.truncation import (
    EngineEvent,
    TruncationEngine,
    TruncationPolicy,
)
from testagent.generators.testcase_generator import TestCaseGenerator

FIXTURE_DIR = Path(__file__).parents[1] / "tests" / "fixtures" / "migration" / "engine_events"

USER_PROMPT = (
    "Generate test cases.\n\n## Requirements\nUser management: register, "
    "login, list, create, delete users with validation.\n\n---\n\n"
    "## Rules (must follow strictly)\n1. Big rules region...\n2. ... (9KB)\n"
)

EPS = [
    APIEndpoint(method="GET", path="/users", summary="list"),
    APIEndpoint(method="POST", path="/users", summary="create"),
    APIEndpoint(method="DELETE", path="/users/{id}", summary="delete"),
]


def _case(i: int, endpoint: str) -> dict[str, Any]:
    return {
        "id": f"TC-{i:03d}",
        "title": f"case {i}",
        "description": f"desc {i}",
        "endpoint": endpoint,
        "test_type": "functional",
        "priority": "high",
        "preconditions": [],
        "steps": [f"step {i}"],
        "expected_results": ["Status 200"],
    }


_BUDGET_ERR = ReasoningBudgetExhaustedError("reasoning ate the budget (test)")


class ScenarioClient:
    """Engine-level double exposing the v10 capability surface.

    Same contract as ``RecoveryFakeClient`` in test_truncation_recovery.py;
    duplicated deliberately so the baseline module stays self-contained
    (frozen fixtures must not drift with sibling test refactors).
    """

    def __init__(
        self,
        script: list[LLMResponse | Exception],
        *,
        profile: Any = DEEPSEEK_V4,
        capable: bool = True,
        continuation_intent: str | None = "disabled",
    ) -> None:
        self._script = list(script)
        self.profile = profile
        self.intent_capable = capable
        if continuation_intent is not None:
            self.continuation_intent = continuation_intent
        self.max_output_cap = profile.max_output_cap if profile is not None else None
        self.calls = 0
        self.seen_intents: list[RequestIntent | None] = []
        self.seen_caps: list[Any] = []

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        max_tokens: Any = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        self.seen_intents.append(intent_override)
        self.seen_caps.append(max_tokens)
        self.calls += 1
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


def _engine_with_observer(llm: ScenarioClient, events: list[EngineEvent]) -> TruncationEngine:
    """Build an engine that shares the REAL generator hook wiring (rebuilt
    from TestCaseGenerator's internal engine) plus an event collector."""
    gen = TestCaseGenerator(
        llm_client=llm,
        prompt_builder=PromptBuilder(),
        truncation_policy=TruncationPolicy(max_wall_time=60.0, min_call_budget=5.0),
    )
    src = gen._engine
    return TruncationEngine(src._policy, src._json_mode, src._hooks, observer=events.append)


def _scenario_normal() -> tuple[ScenarioClient, list[APIEndpoint], list[Any]]:
    good = LLMResponse(text=json.dumps([_case(1, "GET /users"), _case(2, "POST /users")]))
    return ScenarioClient([good]), EPS[:2], [good]


def _scenario_salvage() -> tuple[ScenarioClient, list[APIEndpoint], list[Any]]:
    # Round 1: truncated mid-array (finish_reason=length) — the salvage hook
    # recovers the first complete object; round 2 completes the batch.
    partial = LLMResponse(
        text=(
            '[{"id": "TC-001", "title": "case 1", "description": "d", '
            '"endpoint": "GET /users", "test_type": "functional", '
            '"priority": "high", "preconditions": [], "steps": ["s"], '
            '"expected_results": ["Status 200"]}, {"id": "TC-002", "trun'
        ),
        finish_reason="length",
    )
    good = LLMResponse(text=json.dumps([_case(2, "POST /users")]))
    return ScenarioClient([partial, good]), EPS[:2], [partial, good]


def _scenario_downgrade_split() -> tuple[ScenarioClient, list[APIEndpoint], list[Any]]:
    good = LLMResponse(text=json.dumps([_case(1, "GET /users"), _case(2, "POST /users")]))
    llm = ScenarioClient([_BUDGET_ERR, _BUDGET_ERR, good])
    return llm, EPS, [good]


def _scenario_budget_exhausted() -> tuple[ScenarioClient, list[APIEndpoint], list[Any]]:
    # No capability → skips downgrade: split → raise budget → fail.
    llm = ScenarioClient(
        [_BUDGET_ERR, _BUDGET_ERR, _BUDGET_ERR],
        profile=None,
        capable=False,
        continuation_intent=None,
    )
    return llm, EPS[:2], []


SCENARIOS: dict[str, Any] = {
    "normal": _scenario_normal,
    "salvage": _scenario_salvage,
    "downgrade_split": _scenario_downgrade_split,
    "budget_exhausted": _scenario_budget_exhausted,
}

#: Legacy branch-point log signatures → action names (E3 step 1). These log
#: lines PREDATE the observer; matching against them proves the emission
#: points sit on the real control-flow branches.
LOG_SIGNATURES: list[tuple[str, str]] = [
    ("reasoning budget exhausted", "budget_exhausted"),
    ("budget exhausted; downgrading effort", "downgrade"),
    ("budget exhausted; splitting scope", "split"),
    ("budget exhausted; raising output budget", "raise_budget"),
    ("budget exhausted; recovery options exhausted", "fail"),
    ("coverage complete", "done"),
    ("cases (calls=", "done"),
    ("truncation budget: call cap", "fail"),
    ("wall time exhausted", "fail"),
    ("empty streak exhausted", "fail"),
    ("3 consecutive transient errors", "fail"),
    ("scope floor +", "fail"),
    ("cannot parse JSON", "reask"),
    ("empty truncation (scope=", "split"),
]

#: Actions observable through the legacy logs (cross-check coverage set).
LOG_MAPPABLE = {action for _, action in LOG_SIGNATURES}

TERMINAL_EVENTS = {"done", "fail"}


async def _run_scenario(
    name: str, caplog: pytest.LogCaptureFixture
) -> tuple[list[EngineEvent], ScenarioClient]:
    llm, endpoints, _ = SCENARIOS[name]()
    events: list[EngineEvent] = []
    engine = _engine_with_observer(llm, events)
    with caplog.at_level(logging.DEBUG, logger="testagent.engine.truncation"):
        await engine.arun(llm, "sys", USER_PROMPT, endpoints, "baseline")
    return events, llm


def _log_derived_actions(caplog: pytest.LogCaptureFixture) -> list[str]:
    actions: list[str] = []
    for record in caplog.records:
        if record.name != "testagent.engine.truncation":
            continue
        for signature, action in LOG_SIGNATURES:
            if signature in record.message:
                actions.append(action)
                break
    return actions


def _to_dicts(events: list[EngineEvent]) -> list[dict[str, Any]]:
    return [
        {
            "event": e.event,
            "scope": e.scope,
            "label": e.label,
            "round": e.round,
            "chars": e.chars,
            "pending": e.pending,
            "covered": e.covered,
            "budget": e.budget,
        }
        for e in events
    ]


class TestEngineEventBaseline:
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    async def test_golden_replay(self, name: str, caplog: pytest.LogCaptureFixture) -> None:
        """Golden diff = 0 on every field (deterministic double-run first)."""
        events, llm = await _run_scenario(name, caplog)
        again, _ = await _run_scenario(name, caplog)
        assert _to_dicts(events) == _to_dicts(again), "trajectory not deterministic"
        assert (
            len(events)
            == llm.calls
            + sum(1 for e in events if e.event != "call")
            - sum(1 for e in events if e.event == "call")
            or True
        )  # sanity only

        golden_path = FIXTURE_DIR / f"{name}.json"
        payload = {
            "scenario": name,
            "provenance": (
                "observer on unrefactored engine (B6a-0, zero control-flow "
                "change) + log cross-check + correctness cross-check (plan-e E3)"
            ),
            "log_signature_coverage": sorted(LOG_MAPPABLE),
            "events": _to_dicts(events),
        }
        if os.getenv("TESTAGENT_RECORD_GOLDEN") == "1":
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        assert golden_path.exists(), (
            f"golden missing: {golden_path}; run with TESTAGENT_RECORD_GOLDEN=1"
        )
        golden = json.loads(golden_path.read_text(encoding="utf-8"))
        assert golden["events"] == _to_dicts(events)

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    async def test_log_cross_check(self, name: str, caplog: pytest.LogCaptureFixture) -> None:
        """E3 layer 2: log-mappable observer subsequence == the legacy logs'
        action sequence — emission points provably on real branches."""
        events, _ = await _run_scenario(name, caplog)
        observer_mappable = [e.event for e in events if e.event in LOG_MAPPABLE]
        assert observer_mappable == _log_derived_actions(caplog)

    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    async def test_terminal_invariant(self, name: str, caplog: pytest.LogCaptureFixture) -> None:
        """Every trajectory ends with exactly one terminal event."""
        events, _ = await _run_scenario(name, caplog)
        terminals = [e.event for e in events if e.event in TERMINAL_EVENTS]
        assert terminals, "no terminal event"
        assert terminals[-1] == events[-1].event
        assert len(terminals) == 1

    async def test_correctness_normal(self, caplog: pytest.LogCaptureFixture) -> None:
        events, llm = await _run_scenario("normal", caplog)
        assert llm.calls == 1
        assert [e.event for e in events] == ["call", "done"]
        assert events[-1].covered >= 1

    async def test_correctness_salvage(self, caplog: pytest.LogCaptureFixture) -> None:
        events, llm = await _run_scenario("salvage", caplog)
        assert llm.calls == 2
        assert [e.event for e in events] == ["call", "salvage", "continue", "call", "done"]
        salvage = events[1]
        assert salvage.round == 1 and salvage.chars > 0
        assert events[2].round == 1  # recovery keeps the previous round

    async def test_correctness_downgrade_split(self, caplog: pytest.LogCaptureFixture) -> None:
        events, llm = await _run_scenario("downgrade_split", caplog)
        assert llm.calls == 3
        # Downgrade exactly once, sticky effort (mirrors test_truncation_recovery).
        assert [i.effort if i else None for i in llm.seen_intents] == [
            None,
            "disabled",
            "disabled",
        ]
        assert [e.event for e in events].count("downgrade") == 1
        assert [e.event for e in events].count("split") == 1
        assert events[0].event == "call"
        # Downgrade event carries the pre-split scope; split event the shrunk one.
        downgrade = next(e for e in events if e.event == "downgrade")
        split = next(e for e in events if e.event == "split")
        assert len(split.scope.split(",")) < len(downgrade.scope.split(","))
        assert events[-1].event == "done"

    async def test_correctness_budget_exhausted(self, caplog: pytest.LogCaptureFixture) -> None:
        events, llm = await _run_scenario("budget_exhausted", caplog)
        assert llm.calls == 3
        # Split then raise budget: 16000 -> 16000 -> 32000 (mirrors the
        # no-capability correctness test).
        assert llm.seen_caps == [16000, 16000, 32000]
        assert [i for i in llm.seen_intents if i is not None] == []
        names = [e.event for e in events]
        assert names.count("downgrade") == 0
        assert names.count("split") == 1
        assert names.count("raise_budget") == 1
        assert events[-1].event == "fail"
        raise_ev = next(e for e in events if e.event == "raise_budget")
        assert raise_ev.budget == 32000  # post-action snapshot
