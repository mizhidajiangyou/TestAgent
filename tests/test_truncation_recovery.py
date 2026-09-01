"""Recovery-path tests for the v10 B-phase truncation redesign.

Covers the bug being fixed — "0-char response followed by multiple useless
identical retries with no output" — end to end:

- P0-3: ``length`` + empty on a shared-budget profile short-circuits with a
  typed :class:`ReasoningBudgetExhaustedError` on the FIRST attempt (no
  3x identical internal retries), while transient empties keep the retry.
- P0-2: the intent_override five-layer chain (engine → _call_llm →
  MultiModelLLMClient → _SyncPreferredAdapter → OpenAIClient → compose).
- P1-1/P1-2: engine recovery — one-shot downgrade → split → raise budget →
  fail; clients without intent capability skip the downgrade (x2 raise).
- v7-review leftover: ``_salvage_and_return`` merges salvaged cases.
- 方案 A: slim continuation context excludes the full prompt's Rules region.
"""

import asyncio
import json
import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from testagent.config.models import APIEndpoint
from testagent.engine.llm_client import (
    LLMResponse,
    MultiModelLLMClient,
    OpenAIClient,
    ReasoningBudgetExhaustedError,
)
from testagent.engine.model_profiles import DEEPSEEK_V4, RequestIntent
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.generators.truncation import TruncationPolicy
from tests.test_truncation_engine import MetaMockClient

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


def _chunk(content: str, finish_reason: str = "stop") -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content=content), finish_reason=finish_reason)]
    chunk.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return chunk


def _generator(llm: Any, **policy_kwargs: Any) -> TestCaseGenerator:
    policy_kwargs.setdefault("max_wall_time", 60.0)
    policy_kwargs.setdefault("min_call_budget", 5.0)
    policy = TruncationPolicy(**policy_kwargs)
    return TestCaseGenerator(
        llm_client=llm, prompt_builder=PromptBuilder(), truncation_policy=policy
    )


class RecoveryFakeClient:
    """Engine-level double exposing the v10 capability surface.

    Records the ``intent_override`` it received per call so tests can assert
    the downgrade reached (or was correctly withheld from) the client.
    ``profile=None`` / ``capable=False`` / ``continuation_intent=None``
    simulate a legacy client that predates the v10 attributes.
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
        self.seen_prompts: list[str] = []

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Any = None,
        max_tokens: Any = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        self.seen_prompts.append(user_prompt)
        self.seen_intents.append(intent_override)
        self.seen_caps.append(max_tokens)
        self.calls += 1
        item = self._script[min(self.calls - 1, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


_BUDGET_ERR = ReasoningBudgetExhaustedError("reasoning ate the budget (test)")

_USER_PROMPT = (
    "Generate test cases.\n\n## Requirements\nUser management: register, "
    "login, list, create, delete users with validation.\n\n---\n\n"
    "## Rules (must follow strictly)\n1. Big rules region...\n2. ... (9KB)\n"
)

_GOOD = LLMResponse(text=json.dumps([_case(1, "GET /users"), _case(2, "POST /users")]))


class TestClientShortCircuit:
    """P0-3: budget-exhausted empties short-circuit; transient empties retry."""

    def test_length_empty_deepseek_raises_typed_on_first_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        inner = MagicMock()
        inner.chat.completions.create.return_value = [_chunk("", finish_reason="length")]
        client = OpenAIClient(
            client=inner, model="deepseek-v4-chat", timeout=1.0, max_output_tokens=100
        )

        with pytest.raises(ReasoningBudgetExhaustedError) as excinfo:
            client.chat("sys", "usr")

        # THE bug fix: exactly one request, not 3 identical retries.
        assert inner.chat.completions.create.call_count == 1
        assert excinfo.value.finish_reason == "length"

    def test_stop_empty_deepseek_is_transient_and_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``stop`` + empty is NOT declared exhausted for deepseek (P0-3
        conservative value) — the stream→blocking retry recovers it."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        inner = MagicMock()

        def _create(**kwargs: object) -> object:
            if kwargs.get("stream"):
                return [_chunk("", finish_reason="stop")]
            resp = MagicMock()
            resp.choices = [MagicMock(message=MagicMock(content="RECOVERED"), finish_reason="stop")]
            resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            return resp

        inner.chat.completions.create.side_effect = _create
        client = OpenAIClient(
            client=inner, model="deepseek-v4-chat", timeout=1.0, max_output_tokens=100
        )

        assert client.chat("sys", "usr") == "RECOVERED"
        assert inner.chat.completions.create.call_count == 2  # stream + blocking

    def test_fallback_preserves_typed_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MultiModel fallback still runs, but the typed error survives the
        aggregation instead of being wrapped into a plain RuntimeError."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0)
        inners = []
        for _ in range(2):
            inner = MagicMock()
            inner.chat.completions.create.return_value = [_chunk("", finish_reason="length")]
            inners.append(inner)
        clients = [
            OpenAIClient(
                client=inner,
                model=f"deepseek-v4-chat-{i}",
                timeout=1.0,
                max_output_tokens=100,
            )
            for i, inner in enumerate(inners)
        ]
        mm = MultiModelLLMClient(clients=clients)

        with pytest.raises(ReasoningBudgetExhaustedError):
            mm.chat("sys", "usr")
        # Both models were tried (fallback preserved), each exactly once.
        assert inners[0].chat.completions.create.call_count == 1
        assert inners[1].chat.completions.create.call_count == 1

    def test_intent_override_reaches_create_kwargs(self) -> None:
        inner = MagicMock()
        inner.chat.completions.create.return_value = [_chunk("x")]
        client = OpenAIClient(
            client=inner, model="deepseek-v4-chat", timeout=1.0, max_output_tokens=100
        )

        client.chat_with_meta(
            "sys", "usr", intent_override=RequestIntent(budget=100, effort="disabled")
        )
        kwargs = inner.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert kwargs["temperature"] == 0  # thinking off → temperature restored
        assert kwargs["max_tokens"] == 100

    def test_default_intent_sends_no_temperature_for_deepseek(self) -> None:
        """Model-default thinking stays ON → thinking_dependent profiles must
        NOT send temperature (it is ignored or rejected anyway)."""
        inner = MagicMock()
        inner.chat.completions.create.return_value = [_chunk("x")]
        client = OpenAIClient(
            client=inner, model="deepseek-v4-chat", timeout=1.0, max_output_tokens=100
        )

        client.chat("sys", "usr")
        kwargs = inner.chat.completions.create.call_args.kwargs
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 100


class TestEngineRecovery:
    """Engine-side recovery ladder (plan v10 §5)."""

    async def test_budget_exhausted_downgrades_once_then_splits(self) -> None:
        # Call 1 explodes (budget exhausted) → downgrade; call 2 explodes
        # again → split (no second downgrade); call 3 succeeds on the
        # smaller scope (3 endpoints halve to 1) with the downgraded effort
        # kept sticky.
        llm = RecoveryFakeClient([_BUDGET_ERR, _BUDGET_ERR, _GOOD])
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", _USER_PROMPT, _EPS, "batch 1/1")

        # After the split only the best-covered endpoint survives the scope
        # filter, so one case is accepted from the final response.
        assert len(cases) == 1
        assert llm.calls == 3
        efforts = [intent.effort if intent else None for intent in llm.seen_intents]
        # Downgrade happens exactly once and stays sticky (no escalation).
        assert efforts == [None, "disabled", "disabled"]
        # Downgrade + split rounds use the slim continuation prompt.
        assert "## Rules" not in llm.seen_prompts[1]
        assert "Endpoint signatures" in llm.seen_prompts[1]

    async def test_no_capability_client_skips_downgrade(self) -> None:
        # A client without intent_capable / continuation_intent (legacy fake
        # raising the typed error directly) must NOT receive an
        # intent_override kwarg (capability probe, P0-2) and skips the
        # downgrade: split → raise budget → fail.
        llm = RecoveryFakeClient(
            [_BUDGET_ERR, _BUDGET_ERR, _BUDGET_ERR],
            profile=None,
            capable=False,
            continuation_intent=None,
        )
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", _USER_PROMPT, _EPS[:2], "batch 1/1")

        assert cases == []
        assert llm.calls == 3
        # No intent ever forwarded (capability probe withheld it).
        assert llm.seen_intents == [None, None, None]
        # Split (call 2) then raise budget (call 3): 16000 -> 32000.
        assert llm.seen_caps == [16000, 16000, 32000]

    async def test_raise_budget_clamped_to_profile_cap(self) -> None:
        # Single endpoint (floor from the start): downgrade → raise budget,
        # clamped to the deepseek profile cap when doubling would exceed it.
        llm = RecoveryFakeClient([_BUDGET_ERR, _BUDGET_ERR])
        gen = _generator(llm, output_token_cap=300000)
        cases = await gen._engine.arun(llm, "sys", _USER_PROMPT, _EPS[:1], "batch 1/1")

        assert cases == []
        # Call 1 → downgrade; call 2 → floor + raise budget; call 3 runs at
        # the raised budget (clamped: 300000 x 2 = 600000 → 393216) → FAIL.
        assert llm.calls == 3
        assert llm.seen_caps == [300000, 300000, 393216]

    async def test_salvage_and_return_merges(self) -> None:
        """v7-review leftover fix: salvaged items are merged, not discarded."""
        gen = _generator(RecoveryFakeClient([]))
        engine = gen._engine

        existing = gen._to_test_cases([_case(1, "GET /users")], _EPS)
        # Truncated JSON: first object complete (duplicate of existing),
        # second object complete (new), third cut mid-way.
        raw = (
            "[\n"
            + json.dumps(_case(1, "GET /users"))
            + ",\n"
            + json.dumps(_case(2, "POST /users"))
            + ',\n{"id": "TC-003", "title": "cut of'
        )
        result = engine._salvage_and_return(
            existing,
            raw,
            _EPS,
            {"GET /users", "POST /users"},
            {"GET /users": 2, "POST /users": 2},
            # The loop would pass the produced cases' keys: the salvaged
            # duplicate of case 1 must be deduped against them.
            {gen._case_dedup_key(tc) for tc in existing},
        )
        # Existing case + salvaged NEW case (the duplicate is deduped).
        assert len(result) == 2
        assert {tc.title for tc in result} == {"case 1", "case 2"}

    async def test_slim_continue_excludes_full_context(self) -> None:
        # Round 1 truncated with partial content; round 2 must receive the
        # slim context (signature + summary), never the full prompt again.
        llm = MetaMockClient(
            [
                LLMResponse(text=json.dumps([_case(1, "GET /users")]), finish_reason="length"),
                _GOOD,
            ]
        )
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", _USER_PROMPT, _EPS, "batch 1/1")
        # Round 1 salvaged GET /users; round 2's complete response adds
        # POST /users and the loop returns on the first complete parse.
        assert len(cases) == 2

        second = llm.prompts[1]
        assert "## Rules" not in second  # 9KB rules region dropped
        assert "Endpoint signatures" in second
        assert "Requirement summary" in second
        assert "User management" in second  # extracted requirement text
        assert "do NOT repeat" in second
        assert "Still needed" in second


class TestFiveLayerChain:
    """P0-2: intent_override through all five layers, end to end."""

    def test_intent_override_five_layer_chain(self) -> None:
        """Sync ``run`` (adapter layer) over a real MultiModelLLMClient with
        a real OpenAIClient whose SDK create is mocked.

        Call 1 (stream, default intent): empty + length on deepseek → typed
        budget-exhausted error after exactly ONE request. Call 2 (stream,
        downgraded intent): carries ``extra_body.thinking.type=disabled`` and
        ``temperature=0`` down the whole chain and succeeds.
        """
        create_calls: list[dict[str, Any]] = []
        good_json = json.dumps([_case(1, "GET /users"), _case(2, "POST /users")])

        def _create(**kwargs: Any) -> Any:
            create_calls.append(kwargs)
            if len(create_calls) == 1:
                # First request: reasoning ate the whole shared budget.
                return [_chunk("", finish_reason="length")]
            return [_chunk(good_json, finish_reason="stop")]

        inner = MagicMock()
        inner.chat.completions.create.side_effect = _create
        client = OpenAIClient(
            client=inner, model="deepseek-v4-chat", timeout=1.0, max_output_tokens=100
        )
        mm = MultiModelLLMClient(clients=[client])
        gen = _generator(mm)

        cases = gen._engine.run(mm, "sys", _USER_PROMPT, _EPS, "batch 1/1")

        assert len(cases) == 2
        # Exactly two HTTP requests for the whole recovery — the old code
        # burned 3 identical retries per engine round with no output.
        assert len(create_calls) == 2
        first, second = create_calls
        # Default intent: thinking on → no temperature, no thinking switch.
        assert "temperature" not in first
        assert "extra_body" not in first
        # Downgraded intent reached the SDK through all five layers.
        assert second["extra_body"] == {"thinking": {"type": "disabled"}}
        assert second["temperature"] == 0
        # Downgrade round reuses the slim continuation prompt.
        second_prompt = second["messages"][1]["content"]
        assert "## Rules" not in second_prompt
        assert "Endpoint signatures" in second_prompt

    async def test_legacy_client_raising_typed_error_without_capability(self) -> None:
        """A legacy-shaped fake (no v10 attributes) raising the typed error
        keeps working: no TypeError, recovery degrades to budget raise →
        fail, bounded call count."""
        llm = RecoveryFakeClient(
            [_BUDGET_ERR, _BUDGET_ERR],
            profile=None,
            capable=False,
            continuation_intent=None,
        )
        gen = _generator(llm)
        cases = await gen._engine.arun(llm, "sys", _USER_PROMPT, _EPS[:1], "batch 1/1")
        assert cases == []
        # Single endpoint (floor): call 1 → raise budget, call 2 → FAIL.
        assert llm.calls == 2
        assert llm.seen_intents == [None, None]


class TestObservability:
    """Concurrent-call log attribution and thinking-phase visibility.

    Reproduces the two blind spots from the user's real run: a 6-minute log
    silence while qwen thought (only DEBUG logs existed for reasoning
    tokens), and interleaved streaming logs that looked sequential because
    nothing identified which batch/requirement a line belonged to.
    """

    @staticmethod
    def _deepseek_client(good_json: str) -> OpenAIClient:
        inner = MagicMock()
        inner.chat.completions.create.return_value = [_chunk(good_json, finish_reason="stop")]
        return OpenAIClient(
            client=inner, model="deepseek-v4-chat", timeout=1.0, max_output_tokens=100
        )

    async def test_call_label_shown_in_client_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """The engine's batch label reaches client-side request/response logs
        via the CALL_LABEL contextvar (propagated through to_thread)."""
        good_json = json.dumps([_case(1, "GET /users"), _case(2, "POST /users")])
        client = self._deepseek_client(good_json)
        gen = _generator(client)

        with caplog.at_level("INFO"):
            cases = await gen._engine.arun(client, "sys", _USER_PROMPT, _EPS, "Req X/1")

        assert len(cases) == 2
        messages = [rec.message for rec in caplog.records]
        assert any("[Req X/1] → model" in m for m in messages)
        assert any("[Req X/1] ← model" in m for m in messages)

    async def test_concurrent_aruns_log_own_labels(self, caplog: pytest.LogCaptureFixture) -> None:
        """Two arun calls gathered concurrently each tag their own logs —
        the direct proof that fan-out calls are concurrent and attributable
        (no label crosses into the sibling's log lines)."""
        good_json = json.dumps([_case(1, "GET /users")])
        client = self._deepseek_client(good_json)
        gen = _generator(client)

        with caplog.at_level("INFO"):
            await asyncio.gather(
                gen._engine.arun(client, "sys", _USER_PROMPT, _EPS, "Req A"),
                gen._engine.arun(client, "sys", _USER_PROMPT, _EPS, "Req B"),
            )

        messages = [rec.message for rec in caplog.records]
        assert any("[Req A] → model" in m for m in messages)
        assert any("[Req B] → model" in m for m in messages)
        # No cross-contamination between the concurrent tasks' contexts.
        assert not any("[Req A]" in m and "[Req B]" in m for m in messages)

    def test_thinking_progress_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Reasoning-token streaming is observable: a oneshot 'is thinking'
        notice and the final response log (with the reasoning char count) at
        INFO; throttled progress lines at DEBUG (downgraded from INFO in the
        prompt-engineering pass to cut log noise — the one-shot notice and
        the completion summary carry the signal)."""
        monkeypatch.setattr("testagent.engine.llm_client.THINKING_LOG_INTERVAL", 0.0)

        def _rchunk(text: str) -> MagicMock:
            chunk = MagicMock()
            chunk.choices = [
                MagicMock(delta=MagicMock(content="", reasoning_content=text), finish_reason=None)
            ]
            chunk.usage = None
            return chunk

        inner = MagicMock()
        inner.chat.completions.create.return_value = [
            _rchunk("R" * 5000),
            _rchunk("R" * 6000),
            _chunk("hello", finish_reason="stop"),
        ]
        client = OpenAIClient(client=inner, model="qwen3.8-max", timeout=1.0, max_output_tokens=100)

        with caplog.at_level(logging.DEBUG, logger="testagent.engine.llm_client"):
            assert client.chat("sys", "usr") == "hello"

        info_messages = [rec.message for rec in caplog.records if rec.levelno >= logging.INFO]
        all_messages = [rec.message for rec in caplog.records]
        assert any("is thinking" in m for m in info_messages)
        assert any("reasoning=11000 chars" in m for m in info_messages)
        assert any("thinking: 5000 chars so far" in m for m in all_messages)
        assert any("thinking: 11000 chars so far" in m for m in all_messages)
