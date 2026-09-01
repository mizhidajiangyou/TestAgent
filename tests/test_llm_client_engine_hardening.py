"""Engine-hardening tests (plan-c Step 1): B1.1 blocking timeout semantics,
B1.2 abandoned-worker resource model, B2.1 streaming de-stickiness, B2.2
hard-timeout landing with the v10 boundary (timeout ≠ budget exhaustion)."""

import threading
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, BadRequestError

from testagent.engine.llm_client import (
    DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS,
    LLMCallTimeoutError,
    OpenAIClient,
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
        # Force the blocking channel: the hard timeout only guards blocking
        # calls (streaming has its own SDK-level timeout).
        client.set_stream_enabled(False)
        with pytest.raises(LLMCallTimeoutError):
            client.chat("sys", "usr")
        # 3 attempts allowed by MAX_RETRIES but the timeout budget caps at 2.
        assert sdk.chat.completions.create.call_count <= 2
        blocker.set()


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
