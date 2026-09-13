"""Engine-hardening tests (plan-c Step 1): B1.1 blocking timeout semantics,
B1.2 abandoned-worker resource model, B2.1 streaming de-stickiness, B2.2
hard-timeout landing with the v10 boundary (timeout ≠ budget exhaustion).

2026-09-13 additions: streaming wall-clock cap, unknown-model guards
(thinking-cap escape hatch + budget-exhausted short-circuit)."""

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from testagent.engine.llm_client import (
    DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS,
    LLMCallTimeoutError,
    LLMOutputTooLongError,
    OpenAIClient,
    _parse_extra_body,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _client(timeout: float = 1.0, **kwargs: float) -> OpenAIClient:
    sdk = MagicMock()
    return OpenAIClient(client=sdk, model="m", timeout=timeout, **kwargs)


def _blocking_ok(client: OpenAIClient) -> None:
    resp = MagicMock()
    resp.choices = [
        MagicMock(message=MagicMock(content="ok"), reasoning_content=None, finish_reason="stop")
    ]
    resp.usage = None
    client._client.chat.completions.create.return_value = resp


# ----------------------------------------------------------------------
# B1.1 — blocking_hard_timeout semantics
# ----------------------------------------------------------------------


class TestBlockingTimeoutSemantics:
    def test_unset_uses_max_of_timeout_x2_and_default(self) -> None:
        client = _client(timeout=1.0)
        assert client._blocking_hard_timeout == DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS
        long = _client(timeout=400.0)
        assert long._blocking_hard_timeout == 800.0  # timeout*2 wins over 600

    def test_explicit_value_is_honoured_verbatim(self) -> None:
        # THE fix (plan-c B1.1): 60 really means 60 — not a hidden minimum.
        client = _client(blocking_hard_timeout=60.0)
        assert client._blocking_hard_timeout == 60.0

    def test_explicit_below_request_timeout_fails_construction(self) -> None:
        with pytest.raises(ValueError, match="must be >="):
            _client(timeout=300.0, blocking_hard_timeout=60.0)


# ----------------------------------------------------------------------
# B2.2 — hard timeout fires + v10 boundary
# ----------------------------------------------------------------------


class TestHardTimeout:
    def test_wedged_provider_raises_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.05)
        sdk = MagicMock()
        blocker = threading.Event()
        sdk.chat.completions.create.side_effect = lambda **kw: blocker.wait(10)
        client = OpenAIClient(client=sdk, model="m", timeout=0.1, blocking_hard_timeout=0.2)
        with pytest.raises(LLMCallTimeoutError):
            client._blocking_create({"model": "m"}, sid="t")
        blocker.set()  # release the abandoned worker for clean teardown

    def test_timeout_is_transient_not_budget_exhausted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """v10 boundary: a hard timeout must surface as a TRANSIENT error
        (retry + model fallback), never as ReasoningBudgetExhaustedError —
        a downgrade trigger would be wrong for a wedged provider."""
        from testagent.engine.llm_client import ReasoningBudgetExhaustedError

        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.05)
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0.0)
        monkeypatch.setattr("testagent.engine.llm_client.MAX_TIMEOUT_RETRIES_PER_CALL", 0)
        sdk = MagicMock()
        blocker = threading.Event()
        sdk.chat.completions.create.side_effect = lambda **kw: blocker.wait(10)
        client = OpenAIClient(
            client=sdk, model="deepseek-v4-chat", timeout=0.1, blocking_hard_timeout=0.15
        )
        with pytest.raises(LLMCallTimeoutError) as excinfo:
            client.chat("sys", "usr")
        assert not isinstance(excinfo.value, ReasoningBudgetExhaustedError)
        blocker.set()

    def test_timeout_retry_budget_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """MAX_TIMEOUT_RETRIES_PER_CALL=1: at most 2 timed-out attempts per
        _chat_core loop, then the error propagates."""
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.02)
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0.0)
        sdk = MagicMock()
        blocker = threading.Event()
        sdk.chat.completions.create.side_effect = lambda **kw: blocker.wait(10)
        client = OpenAIClient(client=sdk, model="m", timeout=0.05, blocking_hard_timeout=0.06)
        # Force the blocking channel: this test targets the blocking timeout
        # budget specifically (the streaming channel has its own cap, covered
        # by TestStreamingHardTimeout).
        client.set_stream_enabled(False)
        with pytest.raises(LLMCallTimeoutError):
            client.chat("sys", "usr")
        # 3 attempts allowed by MAX_RETRIES but the timeout budget caps at 2.
        assert sdk.chat.completions.create.call_count <= 2
        blocker.set()


# ----------------------------------------------------------------------
# 2026-09-13 — streaming wall-clock cap (Phase 2 "frozen for minutes")
# ----------------------------------------------------------------------


def _stream_chunk(
    content: str | None = None,
    reasoning: str | None = None,
    finish_reason: str | None = None,
    usage: MagicMock | None = None,
) -> MagicMock:
    """Build one streaming chunk carrying visible content and/or reasoning."""
    chunk = MagicMock()
    chunk.choices = [
        MagicMock(
            delta=MagicMock(content=content or "", reasoning_content=reasoning),
            finish_reason=finish_reason,
        )
    ]
    chunk.usage = usage
    return chunk


def _slow_stream(seconds: float, chunk_sleep: float = 0.02) -> Any:
    """Generator that keeps emitting content for ~``seconds`` (never finishes in time)."""

    def _gen() -> Any:
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(chunk_sleep)
            yield _stream_chunk(content="x")

    return _gen()


class TestStreamingHardTimeout:
    """A stream that keeps trickling chunks never trips the transport read
    timeout, so before this cap a single call was bounded only by the
    provider's own budget."""

    def test_unset_uses_max_of_timeout_x2_and_default(self) -> None:
        assert _client(timeout=1.0).stream_hard_timeout == DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS
        assert _client(timeout=400.0).stream_hard_timeout == 800.0  # timeout*2 wins

    def test_explicit_value_is_honoured_verbatim(self) -> None:
        assert _client(stream_hard_timeout=60.0).stream_hard_timeout == 60.0

    def test_explicit_below_request_timeout_fails_construction(self) -> None:
        with pytest.raises(ValueError, match="must be >="):
            _client(timeout=300.0, stream_hard_timeout=60.0)

    def test_stream_exceeding_cap_raises_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.02)
        sdk = MagicMock()
        sdk.chat.completions.create.side_effect = lambda **kw: _slow_stream(5.0)
        client = OpenAIClient(
            client=sdk, model="m", timeout=0.05, max_output_tokens=100, stream_hard_timeout=0.1
        )
        started = time.time()
        with pytest.raises(LLMCallTimeoutError):
            client._stream_completion({"model": "m"}, sid="t")
        # Aborted near the cap, nowhere near the 5s the stream wanted.
        assert time.time() - started < 2.0

    def test_the_cap_is_the_streaming_channel_not_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cap must fire on the streaming channel (where the old code had
        no guard at all), independent of blocking_hard_timeout."""
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.02)
        sdk = MagicMock()
        sdk.chat.completions.create.side_effect = lambda **kw: _slow_stream(5.0)
        client = OpenAIClient(
            client=sdk,
            model="m",
            timeout=0.05,
            max_output_tokens=100,
            blocking_hard_timeout=30.0,  # generous: irrelevant on this channel
            stream_hard_timeout=0.1,
        )
        with pytest.raises(LLMCallTimeoutError):
            client._stream_completion({"model": "m"}, sid="t")


# ----------------------------------------------------------------------
# B1.2 — abandoned worker ledger + metrics + breaker
# ----------------------------------------------------------------------


class TestAbandonedWorkerModel:
    def test_metrics_start_zero(self) -> None:
        client = _client()
        m = client.worker_metrics()
        assert m["active_llm_workers"] == 0
        assert m["abandoned_llm_workers"] == 0
        assert m["abandoned_llm_workers_oldest_age"] == 0.0

    def test_abandoned_worker_counted_until_it_finishes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.05)
        sdk = MagicMock()
        release = threading.Event()
        sdk.chat.completions.create.side_effect = lambda **kw: release.wait(10)
        client = OpenAIClient(client=sdk, model="m", timeout=0.1, blocking_hard_timeout=0.15)
        with pytest.raises(LLMCallTimeoutError):
            client._blocking_create({"model": "m"}, sid="t")
        assert client.abandoned_worker_count == 1
        m = client.worker_metrics()
        assert m["abandoned_llm_workers"] == 1
        assert m["abandoned_llm_workers_oldest_age"] >= 0.0
        # The worker eventually finishes -> removes itself from the ledger.
        release.set()

    def test_breaker_skips_retries_when_abandoned_exceeds_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0.0)
        client = _client()
        _blocking_ok(client)
        # Simulate residue from previously timed-out workers + a tripped
        # breaker: threshold 0 means any abandoned worker surfaces failures.
        client._abandoned_workers.add(999)
        monkeypatch.setattr(client, "_abandoned_breaker_threshold", lambda: 0)
        client._client.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            client.chat("sys", "usr")
        # Breaker tripped on the FIRST failure: no second attempt.
        assert client._client.chat.completions.create.call_count == 1

    def test_concurrent_timeouts_bounded_residue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """N concurrent units timing out together leave exactly N abandoned
        workers (review P0-1: the residue is bounded and observable)."""
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.02)
        sdk = MagicMock()
        release = threading.Event()
        sdk.chat.completions.create.side_effect = lambda **kw: release.wait(10)
        client = OpenAIClient(client=sdk, model="m", timeout=0.05, blocking_hard_timeout=0.08)
        results: list[BaseException | None] = [None] * 3

        def _one(i: int) -> None:
            try:
                client._blocking_create({"model": "m"}, sid=f"u{i}")
            except BaseException as exc:  # recorded for the assertion below
                results[i] = exc

        threads = [threading.Thread(target=_one, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(isinstance(r, LLMCallTimeoutError) for r in results)
        assert client.abandoned_worker_count == 3
        release.set()


# ----------------------------------------------------------------------
# B2.1 — streaming de-stickiness
# ----------------------------------------------------------------------


class TestStreamDestickiness:
    def test_transient_drop_keeps_streaming_enabled(self) -> None:
        client = _client()
        _blocking_ok(client)
        with patch.object(
            client,
            "_stream_completion",
            side_effect=APIConnectionError(request=MagicMock()),
        ):
            client._complete({}, sid="t")
        assert client._stream_enabled is True  # next call retries streaming
        assert client._stream_fail_streak == 1

    def test_httpx_transport_error_counts_as_transient(self) -> None:
        client = _client()
        _blocking_ok(client)
        with patch.object(
            client,
            "_stream_completion",
            side_effect=httpx.ReadError("incomplete chunked read"),
        ):
            client._complete({}, sid="t")
        assert client._stream_enabled is True
        assert client._stream_fail_streak == 1

    def test_programming_bug_propagates_unmasked(self) -> None:
        """A KeyError in stream code is a BUG, not a provider failure — it
        must surface instead of silently falling back to blocking."""
        client = _client()
        with (
            patch.object(client, "_stream_completion", side_effect=KeyError("boom")),
            pytest.raises(KeyError),
        ):
            client._complete({}, sid="t")
        assert client._stream_fail_streak == 0  # not counted

    def test_badrequest_mentioning_stream_disables_sticky(self) -> None:
        client = _client()
        _blocking_ok(client)
        body = MagicMock()
        body.status_code = 400
        err = BadRequestError(
            "stream is not supported", response=MagicMock(status_code=400), body=None
        )
        with patch.object(client, "_stream_completion", side_effect=err):
            client._complete({}, sid="t")
        assert client._stream_enabled is False

    def test_unrelated_badrequest_is_single_call_fallback(self) -> None:
        """A 400 about something else (param typo) must NOT permanently
        disable streaming (review #17)."""
        client = _client()
        _blocking_ok(client)
        err = BadRequestError(
            "invalid model parameter", response=MagicMock(status_code=400), body=None
        )
        with patch.object(client, "_stream_completion", side_effect=err):
            client._complete({}, sid="t")
        assert client._stream_enabled is True

    def test_three_consecutive_transient_failures_disable(self) -> None:
        client = _client()
        _blocking_ok(client)
        with patch.object(
            client,
            "_stream_completion",
            side_effect=APIConnectionError(request=MagicMock()),
        ):
            for _ in range(3):
                client._complete({}, sid="t")
        assert client._stream_enabled is False

    def test_success_resets_streak(self) -> None:
        client = _client()
        client._stream_fail_streak = 2
        client.set_stream_enabled(True)
        assert client._stream_fail_streak == 0

    def test_healthy_stream_resets_streak(self) -> None:
        client = _client()
        client._stream_fail_streak = 2
        with patch.object(client, "_stream_completion", return_value=("ok", "stop", None, 0)):
            client._complete({}, sid="t")
        assert client._stream_fail_streak == 0


# ----------------------------------------------------------------------
# 2026-09-13 — unknown-model guards (no thinking cap, no short-circuit)
# ----------------------------------------------------------------------


class TestUnknownModelThinkingCap:
    """A model absent from the profile registry gets NO thinking control: the
    registry translates the OPENAI_REASONING_EFFORT intent into family dialects,
    so for an unregistered model the effort tier is silently dropped and the
    model thinks unbounded (Phase 1 of the 2026-09-13 report: 32k/45k/103k chars
    of reasoning vs ~13k with the cap active)."""

    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No backoff sleeps, no watchdog-join stall (suite stays snappy)."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0.0)
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.01)

    def test_unregistered_model_receives_no_thinking_parameter(self) -> None:
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = [_stream_chunk("ok", finish_reason="stop")]
        client = OpenAIClient(
            client=sdk,
            model="kimi-k2.7-code",
            timeout=1.0,
            max_output_tokens=100,
            reasoning_effort="low",
        )
        client.chat("sys", "usr")
        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert "extra_body" not in kwargs
        assert "reasoning_effort" not in kwargs

    def test_registered_model_does_receive_the_cap(self) -> None:
        """Contrast case: the same effort tier IS translated for known families."""
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = [_stream_chunk("ok", finish_reason="stop")]
        client = OpenAIClient(
            client=sdk,
            model="qwen3.8-max",
            timeout=1.0,
            max_output_tokens=100,
            reasoning_effort="low",
        )
        client.chat("sys", "usr")
        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking_budget": 4096}

    def test_extra_body_override_reaches_the_request(self) -> None:
        """The escape hatch: the operator supplies the provider's own fragment."""
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = [_stream_chunk("ok", finish_reason="stop")]
        client = OpenAIClient(
            client=sdk,
            model="kimi-k2.7-code",
            timeout=1.0,
            max_output_tokens=100,
            extra_body={"enable_thinking": False},
        )
        client.chat("sys", "usr")
        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"enable_thinking": False}

    def test_extra_body_override_wins_over_profile_fragment(self) -> None:
        sdk = MagicMock()
        sdk.chat.completions.create.return_value = [_stream_chunk("ok", finish_reason="stop")]
        client = OpenAIClient(
            client=sdk,
            model="qwen3.8-max",
            timeout=1.0,
            max_output_tokens=100,
            reasoning_effort="low",
            extra_body={"thinking_budget": 1024},
        )
        client.chat("sys", "usr")
        kwargs = sdk.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking_budget": 1024}


class TestExtraBodyParsing:
    """Configuration errors fail at startup, never silently drop the intent."""

    def test_empty_is_disabled(self) -> None:
        assert _parse_extra_body("") == {}
        assert _parse_extra_body("   ") == {}

    def test_valid_object_is_parsed(self) -> None:
        assert _parse_extra_body('{"enable_thinking": false}') == {"enable_thinking": False}

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            _parse_extra_body("{not json}")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ValueError, match="must be a JSON object"):
            _parse_extra_body("[1, 2, 3]")


class TestBudgetExhaustedGuardForUnregisteredModel:
    """Unregistered models cannot classify BUDGET_EXHAUSTED (no shared-budget
    knowledge), so the v10 short-circuit never applied and an empty body burned
    MAX_RETRIES identical multi-minute reasoning passes."""

    @pytest.fixture(autouse=True)
    def _fast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No backoff sleeps, no watchdog-join stall (suite stays snappy)."""
        monkeypatch.setattr("testagent.engine.llm_client.RETRY_BACKOFF_SECONDS", 0.0)
        monkeypatch.setattr("testagent.engine.llm_client.WAITING_LOG_INTERVAL", 0.01)

    @staticmethod
    def _exhausted_stream() -> list[MagicMock]:
        """Reasoning only, then finish_reason=length with an EMPTY visible body."""
        return [
            _stream_chunk(reasoning="t" * 200),
            _stream_chunk(finish_reason="length"),
        ]

    def test_fails_fast_instead_of_retrying(self) -> None:
        sdk = MagicMock()
        sdk.chat.completions.create.side_effect = lambda **kw: list(self._exhausted_stream())
        client = OpenAIClient(
            client=sdk, model="kimi-k2.7-code", timeout=1.0, max_output_tokens=100
        )
        with pytest.raises(LLMOutputTooLongError, match="consumed the whole output budget"):
            client.chat("sys", "usr")
        assert sdk.chat.completions.create.call_count == 1

    def test_transient_empty_still_retries(self) -> None:
        """REGRESSION GUARD: the fast-fail must NOT touch the classic transient
        empty path (finish_reason=stop/None) whose stream -> blocking recovery
        is a documented fix — see the empty-stream recovery notes."""
        sdk = MagicMock()

        def _dispatch(**kw: Any) -> Any:
            if kw.get("stream"):
                return [_stream_chunk(content="", finish_reason=None)]
            resp = MagicMock()
            resp.choices = [
                MagicMock(
                    message=MagicMock(content="", reasoning_content=None), finish_reason="stop"
                )
            ]
            resp.usage = None
            return resp

        sdk.chat.completions.create.side_effect = _dispatch
        client = OpenAIClient(
            client=sdk, model="kimi-k2.7-code", timeout=1.0, max_output_tokens=100
        )
        with pytest.raises(LLMOutputTooLongError):
            client.chat("sys", "usr")
        assert sdk.chat.completions.create.call_count == 3  # stream + 2 blocking

    def test_empty_length_without_reasoning_still_retries(self) -> None:
        """The fast-fail is gated on observed reasoning: an empty length body
        with NO reasoning (nothing consumed the budget) stays on the transient
        path."""
        sdk = MagicMock()

        def _dispatch(**kw: Any) -> Any:
            if kw.get("stream"):
                return [_stream_chunk(content="", finish_reason="length")]
            resp = MagicMock()
            resp.choices = [
                MagicMock(
                    message=MagicMock(content="", reasoning_content=None), finish_reason="length"
                )
            ]
            resp.usage = None
            return resp

        sdk.chat.completions.create.side_effect = _dispatch
        client = OpenAIClient(
            client=sdk, model="kimi-k2.7-code", timeout=1.0, max_output_tokens=100
        )
        with pytest.raises(LLMOutputTooLongError):
            client.chat("sys", "usr")
        assert sdk.chat.completions.create.call_count == 3
