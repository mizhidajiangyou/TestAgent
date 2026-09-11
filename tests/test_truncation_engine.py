"""Tests for the v6 truncation-aware generation loop.

Covers (plan v6 §七):
- LLMResponse compatibility (no AttributeError; ``chat()`` stays ``str``).
- ``is_truncated`` reading finish_reason / completion_tokens (all branches).
- ``filter_to_scope`` quota clipping + foreign-endpoint drops.
- ``shrink_scope`` smart halving.
- Engine end-to-end: complete response, truncated continuation, empty streak,
  LLMOutputTooLongError scope shrink, transient give-up.
- The ``_call_llm`` compatibility layer (plain MagicMock falls back to achat).
- The ``enable_v4_resume`` legacy fallback flag.
"""

import json
from typing import Any
from unittest.mock import MagicMock

from testagent.config.models import APIEndpoint
from testagent.engine.llm_client import LLMOutputTooLongError, LLMResponse
from testagent.engine.prompt_builder import PromptBuilder
from testagent.engine.truncation import (
    TruncationPolicy,
    build_continue_prompt,
    chars_per_token_for,
    filter_to_scope,
    is_truncated,
    shrink_scope,
)
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.pipeline.truncation_hooks import dict_scope_key

_EPS = [
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


def _resp(cases: list[dict[str, Any]], finish_reason: str | None = None) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(cases, ensure_ascii=False),
        finish_reason=finish_reason,
        completion_tokens=None,
    )


class MetaMockClient:
    """LLM double exposing only the rich async contract (no ``str`` methods)."""

    def __init__(self, script: list[LLMResponse | Exception]) -> None:
        self._script = list(script)
        self.calls = 0
        self.prompts: list[str] = []

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        max_tokens: Any = None,
    ) -> LLMResponse:
        self.prompts.append(user_prompt)
        item = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    async def achat(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("rich client must be driven via achat_with_meta")


def _generator(llm: Any, **policy_kwargs: Any) -> TestCaseGenerator:
    # min_call_budget must be well below max_wall_time, otherwise the float
    # epsilon between deadline computation and the guard check trips the
    # "time too tight" bailout on the very first iteration.
    policy_kwargs.setdefault("max_wall_time", 60.0)
    policy_kwargs.setdefault("min_call_budget", 5.0)
    policy = TruncationPolicy(**policy_kwargs)
    return TestCaseGenerator(
        llm_client=llm,
        prompt_builder=PromptBuilder(),
        truncation_policy=policy,
    )


class TestIsTruncated:
    def test_length_reason_is_truncated(self) -> None:
        r = LLMResponse(text="[1,2", finish_reason="length", completion_tokens=100)
        assert is_truncated(r, TruncationPolicy()) is True

    def test_completion_tokens_heuristic(self) -> None:
        p = TruncationPolicy(output_token_cap=4000)
        near = LLMResponse(text="x", finish_reason=None, completion_tokens=3950)
        far = LLMResponse(text="x", finish_reason=None, completion_tokens=100)
        # text "x" does not end with } or ] → near-cap counts as truncated
        assert is_truncated(near, p) is True
        assert is_truncated(far, p) is False

    def test_none_reason_complete_json_not_truncated(self) -> None:
        p = TruncationPolicy(output_token_cap=4000)
        r = LLMResponse(text=json.dumps([{"a": 1}]), finish_reason=None, completion_tokens=3980)
        # near cap BUT ends with "]" → not truncated
        assert is_truncated(r, p) is False

    def test_stop_reason_with_parseable_json(self) -> None:
        r = LLMResponse(text=json.dumps([1, 2]), finish_reason="stop")
        assert is_truncated(r, TruncationPolicy()) is False

    def test_stop_reason_with_broken_tail(self) -> None:
        r = LLMResponse(text='[{"a": 1', finish_reason="stop")
        assert is_truncated(r, TruncationPolicy()) is True

    def test_no_attribute_error_on_bare_response(self) -> None:
        # v6 P0: the old str-only path raised AttributeError; rich path must not.
        r = LLMResponse(text="", finish_reason="length")
        assert is_truncated(r, TruncationPolicy()) is True


class TestCharsPerToken:
    def test_cjk_text(self) -> None:
        p = TruncationPolicy()
        assert chars_per_token_for("用户管理模块测试" * 10, p) == p.chars_per_token_cjk

    def test_latin_text(self) -> None:
        p = TruncationPolicy()
        assert chars_per_token_for("plain ascii text " * 10, p) == p.chars_per_token_lat

    def test_empty(self) -> None:
        p = TruncationPolicy()
        assert chars_per_token_for("", p) == p.chars_per_token_mix


class TestFilterToScope:
    def test_drops_foreign_endpoints_and_caps_quota(self) -> None:
        items = [
            _case(1, "GET /users"),
            _case(2, "GET /users"),
            _case(3, "GET /users"),  # over quota (2)
            _case(4, "PUT /other"),  # foreign endpoint
            _case(5, "POST /users"),
        ]
        batch = {"GET /users", "POST /users"}
        expected = {"GET /users": 2, "POST /users": 2}
        # B6a-2: the DECLARED scope-key function is injected by the caller.
        kept = filter_to_scope(items, batch, expected, dict_scope_key)
        eps = [str(it["endpoint"]) for it in kept]
        assert eps == ["GET /users", "GET /users", "POST /users"]


class TestShrinkScope:
    def test_halves_and_keeps_covered_first(self) -> None:
        scope = ["A", "B", "C", "D"]
        covered = {"A": 3, "C": 2, "B": 0, "D": 0}
        expected = {"A": 2, "B": 2, "C": 2, "D": 2}
        new_scope, new_set, new_pending, floor = shrink_scope(
            scope, expected, covered, TruncationPolicy(min_scope=1)
        )
        assert len(new_scope) == 2
        assert new_scope[0] == "A"  # best covered first
        assert new_set == set(new_scope)
        assert floor is False
        assert new_pending["A"] == max(0, 2 - 3)

    def test_floor_reached(self) -> None:
        scope = ["A", "B"]
        new_scope, _, _, floor = shrink_scope(
            scope, {"A": 2, "B": 2}, {}, TruncationPolicy(min_scope=1)
        )
        assert len(new_scope) == 1
        assert floor is True


class TestEngineEndToEnd:
    async def test_complete_response_returns_immediately(self) -> None:
        llm = MetaMockClient([_resp([_case(1, "GET /users"), _case(2, "POST /users")])])
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        assert len(cases) == 2
        assert llm.calls == 1

    async def test_truncated_continues_until_quota(self) -> None:
        # Round 1 truncated with partial (only GET covered); round 2 completes.
        llm = MetaMockClient(
            [
                _resp([_case(1, "GET /users")], finish_reason="length"),
                _resp([_case(2, "POST /users"), _case(3, "DELETE /users/{id}")]),
            ]
        )
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        assert len(cases) == 3
        assert llm.calls == 2
        # Round 2's prompt must be a CONTINUATION (mentions already-produced).
        assert "do NOT repeat" in llm.prompts[1]

    async def test_quota_caps_total_cases(self) -> None:
        # Model returns 3 cases for one endpoint; quota keeps only 2, then
        # pending hits zero and the loop stops asking.
        llm = MetaMockClient(
            [
                _resp(
                    [
                        _case(1, "GET /users"),
                        _case(2, "GET /users"),
                        _case(3, "GET /users"),
                        _case(4, "POST /users"),
                        _case(5, "DELETE /users/{id}"),
                    ]
                )
            ]
        )
        gen = _generator(llm)
        items = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        got = [str(it["endpoint"]) for it in items]
        assert got.count("GET /users") == 2  # quota respected
        assert len(items) == 4

    async def test_empty_streak_gives_up(self) -> None:
        llm = MetaMockClient([LLMResponse(text="", finish_reason="stop")])
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        assert cases == []
        # empty + reask(empty) + reask(empty) = max_empty_streak (3) calls
        assert llm.calls == 3

    async def test_llm_output_too_long_shrinks_scope(self) -> None:
        # Call 1 explodes with an empty truncation; the engine shrinks the
        # scope and the continuation (with_meta) succeeds on the smaller set.
        good = _resp([_case(1, "GET /users")])
        llm = MetaMockClient([LLMOutputTooLongError("empty"), good, good])
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        assert len(cases) == 1
        assert llm.calls >= 2

    async def test_transient_errors_return_partial(self) -> None:
        partial = _resp([_case(1, "GET /users")], finish_reason="length")
        llm = MetaMockClient(
            [partial, RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")]
        )
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        assert len(cases) == 1  # partial salvaged, then 3 transients give up

    async def test_call_budget_exhausted(self) -> None:
        # Every call returns an unparseable non-empty string: the loop keeps
        # re-asking until the call budget is hit, then returns what it has.
        bad = LLMResponse(text="not json at all", finish_reason="stop")
        llm = MetaMockClient([bad])
        gen = _generator(llm, max_total_calls=3, min_call_budget=0.0)
        cases = await gen._engine.arun(llm, "sys", "user", _EPS, "batch 1/1")
        assert cases == []
        assert llm.calls == 3


class TestCompatibilityLayer:
    async def test_plain_mock_falls_back_to_achat(self) -> None:
        # A MagicMock parent (only achat configured, like the existing suites)
        # must degrade to the str contract instead of awaiting a MagicMock.
        mock = MagicMock()
        mock.achat = MagicMock(return_value=json.dumps([_case(1, "GET /users")]))
        # Make the await work: achat returns a plain str via coroutine wrapper.

        async def _fake_achat(*a: Any, **k: Any) -> str:
            return json.dumps([_case(1, "GET /users")])

        mock.achat = _fake_achat
        gen = _generator(mock)
        cases = await gen._engine.arun(mock, "sys", "user", _EPS, "batch 1/1")
        assert len(cases) == 1

    def test_v2_flag_falls_back_to_legacy(self) -> None:
        gen = _generator(MagicMock(), enable_v4_resume=False)
        assert gen._truncation_policy.enable_v4_resume is False

    async def test_sync_run_executes_loop(self) -> None:
        llm = MetaMockClient([_resp([_case(1, "GET /users")])])
        gen = _generator(llm)
        cases = gen._engine.run(llm, "sys", "user", _EPS, "batch 1/1")
        assert len(cases) == 1


class TestContinuePrompt:
    def test_mentions_pending_and_fingerprint(self) -> None:
        prompt = build_continue_prompt(
            "BASE", "- case 1 @ GET /users", "batch", {"GET /users": 2, "POST /users": 0}
        )
        assert "BASE" in prompt
        assert "GET /users x2" in prompt
        assert "do NOT repeat" in prompt
