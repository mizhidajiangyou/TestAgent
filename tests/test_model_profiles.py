"""Tests for the declarative model profile layer (plan v10 §3-§5)."""

import logging

import pytest

from testagent.engine.model_profiles import (
    DEEPSEEK_V4,
    GENERIC_OPENAI_COMPATIBLE,
    OPENAI_CLASSIC,
    OPENAI_REASONING,
    QWEN_3_8,
    ModelProfile,
    NextAction,
    Outcome,
    RecoveryState,
    RequestIntent,
    classify_response,
    compose_request,
    recovery_plan,
    resolve_profile,
)


class TestResolveProfile:
    def test_explicit_env_overrides_matcher(self) -> None:
        assert resolve_profile("gpt-5", explicit="deepseek-v4") is DEEPSEEK_V4

    def test_invalid_explicit_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="Unknown model profile 'nope'"):
            resolve_profile("deepseek-v4-chat", explicit="nope")

    def test_matcher_per_family(self) -> None:
        assert resolve_profile("deepseek-v4-chat") is DEEPSEEK_V4
        assert resolve_profile("DeepSeek-V4-Reasoner") is DEEPSEEK_V4
        assert resolve_profile("qwen3.8-max") is QWEN_3_8
        assert resolve_profile("qwen3-plus") is QWEN_3_8
        assert resolve_profile("gpt-5-mini") is OPENAI_REASONING
        assert resolve_profile("o3-mini") is OPENAI_REASONING
        assert resolve_profile("gpt-4o") is OPENAI_CLASSIC
        assert resolve_profile("gpt-4.1-mini") is OPENAI_CLASSIC

    def test_unknown_falls_back_to_generic_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING):
            profile = resolve_profile("some-internal-proxy-model")
        assert profile is GENERIC_OPENAI_COMPATIBLE
        assert any("not in the profile registry" in rec.message for rec in caplog.records)

    def test_unverified_profile_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """T14 backfilled qwen3.8 to VERIFIED; the warning must still fire
        for profiles that remain UNVERIFIED."""
        with caplog.at_level(logging.WARNING):
            resolve_profile("qwen3.8-max")
            assert not any("NOT been verified" in rec.message for rec in caplog.records)
        from testagent.engine.model_profiles import _PROFILES

        unverified = next(p for p in _PROFILES if p.verification == "UNVERIFIED")
        with caplog.at_level(logging.WARNING):
            resolve_profile(unverified.matchers[0].replace("*", "-x"))
        assert any("NOT been verified" in rec.message for rec in caplog.records)

    def test_verified_profile_logs_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            resolve_profile("deepseek-v4-chat")
        assert not any("NOT been verified" in rec.message for rec in caplog.records)


class TestComposeRequest:
    def test_response_format_passthrough(self) -> None:
        fmt = {"type": "json_object"}
        kwargs = compose_request(
            DEEPSEEK_V4,
            RequestIntent(budget=8000),
            channel="stream",
            response_format=fmt,
        )
        assert kwargs["response_format"] == fmt

    def test_budget_clamped_to_cap_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            kwargs = compose_request(DEEPSEEK_V4, RequestIntent(budget=500000), channel="stream")
        assert kwargs["max_tokens"] == 393216
        assert any("clamped to model cap" in rec.message for rec in caplog.records)

    def test_deepseek_default_intent_sends_no_temperature(self) -> None:
        # thinking_dependent: model-default thinking stays ON -> no temperature.
        kwargs = compose_request(DEEPSEEK_V4, RequestIntent(budget=8000), channel="stream")
        assert kwargs == {"max_tokens": 8000}

    def test_deepseek_disabled_restores_temperature(self) -> None:
        kwargs = compose_request(
            DEEPSEEK_V4,
            RequestIntent(budget=8000, effort="disabled"),
            channel="stream",
        )
        assert kwargs["temperature"] == 0
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}

    def test_deepseek_low_sends_effort_without_temperature(self) -> None:
        kwargs = compose_request(
            DEEPSEEK_V4,
            RequestIntent(budget=8000, effort="low"),
            channel="stream",
        )
        assert kwargs["reasoning_effort"] == "low"
        assert "temperature" not in kwargs

    def test_qwen_blocking_strips_thinking(self) -> None:
        kwargs = compose_request(
            QWEN_3_8,
            RequestIntent(budget=8000, effort="low"),
            channel="blocking",
        )
        assert "extra_body" not in kwargs
        assert kwargs["max_tokens"] == 8000

    def test_qwen_stream_keeps_thinking_budget(self) -> None:
        kwargs = compose_request(
            QWEN_3_8,
            RequestIntent(budget=8000, effort="low"),
            channel="stream",
        )
        assert kwargs["extra_body"] == {"thinking_budget": 4096}
        # thinking stays on (budget cap only) -> no temperature
        assert "temperature" not in kwargs

    def test_openai_reasoning_contract(self) -> None:
        kwargs = compose_request(
            OPENAI_REASONING,
            RequestIntent(budget=8000, effort="low"),
            channel="stream",
        )
        assert kwargs["max_completion_tokens"] == 8000
        assert kwargs["reasoning_effort"] == "low"
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs

    def test_openai_classic_contract(self) -> None:
        kwargs = compose_request(
            OPENAI_CLASSIC,
            RequestIntent(budget=8000, effort="low"),  # effort silently dropped
            channel="stream",
        )
        assert kwargs == {"max_tokens": 8000, "temperature": 0}

    def test_generic_drops_thinking_and_temperature(self) -> None:
        kwargs = compose_request(
            GENERIC_OPENAI_COMPATIBLE,
            RequestIntent(budget=8000, effort="low"),
            channel="stream",
        )
        assert kwargs == {"max_tokens": 8000}

    def test_unknown_effort_is_silently_dropped(self) -> None:
        kwargs = compose_request(
            DEEPSEEK_V4,
            RequestIntent(budget=8000, effort="ultra"),
            channel="stream",
        )
        assert kwargs == {"max_tokens": 8000}


class TestClassifyResponse:
    def test_length_empty_is_budget_exhausted_for_deepseek(self) -> None:
        assert (
            classify_response(DEEPSEEK_V4, text="", finish_reason="length")
            is Outcome.BUDGET_EXHAUSTED
        )

    def test_stop_empty_is_transient_for_deepseek(self) -> None:
        # P0-3: "stop" is NOT declared for deepseek until diagnosis proves the
        # gateway wraps budget exhaustion as stop — miss over false trigger.
        assert (
            classify_response(DEEPSEEK_V4, text="", finish_reason="stop") is Outcome.TRANSIENT_EMPTY
        )

    def test_none_finish_reason_empty_is_transient(self) -> None:
        assert (
            classify_response(DEEPSEEK_V4, text="", finish_reason=None) is Outcome.TRANSIENT_EMPTY
        )

    def test_nonshared_length_empty_is_transient(self) -> None:
        # classic/generic budgets are not shared with reasoning: an empty body
        # cannot be reasoning starvation — keep the transient retry semantics.
        assert (
            classify_response(OPENAI_CLASSIC, text="", finish_reason="length")
            is Outcome.TRANSIENT_EMPTY
        )
        assert (
            classify_response(GENERIC_OPENAI_COMPATIBLE, text="", finish_reason="length")
            is Outcome.TRANSIENT_EMPTY
        )

    def test_length_with_content_is_truncated_partial(self) -> None:
        assert (
            classify_response(DEEPSEEK_V4, text='[{"a": 1', finish_reason="length")
            is Outcome.TRUNCATED_PARTIAL
        )

    def test_content_is_ok(self) -> None:
        assert classify_response(DEEPSEEK_V4, text='[{"a": 1}]', finish_reason="stop") is (
            Outcome.OK
        )


class TestRecoveryPlan:
    def _plan(self, profile: ModelProfile | None, **kwargs: bool | int) -> NextAction:
        defaults: dict[str, bool | int] = {
            "downgrade_used": False,
            "pending_total": 4,
            "scope_size": 4,
            "budget_raised": False,
            "floor_reached": False,
        }
        defaults.update(kwargs)
        return recovery_plan(profile, RecoveryState(**defaults))  # type: ignore[arg-type]

    def test_full_sequence_downgrade_split_raise_fail(self) -> None:
        # DOWNGRADE -> SPLIT -> RAISE_BUDGET -> FAIL (plan v10 §5.2).
        assert self._plan(DEEPSEEK_V4) is NextAction.DOWNGRADE
        assert self._plan(DEEPSEEK_V4, downgrade_used=True) is NextAction.SPLIT
        assert (
            self._plan(DEEPSEEK_V4, downgrade_used=True, floor_reached=True)
            is NextAction.RAISE_BUDGET
        )
        assert (
            self._plan(DEEPSEEK_V4, downgrade_used=True, floor_reached=True, budget_raised=True)
            is NextAction.FAIL
        )

    def test_no_effort_profile_skips_downgrade(self) -> None:
        assert self._plan(OPENAI_CLASSIC) is NextAction.SPLIT
        assert self._plan(GENERIC_OPENAI_COMPATIBLE) is NextAction.SPLIT
        assert self._plan(None) is NextAction.SPLIT

    def test_floor_with_pending_skips_split_to_raise(self) -> None:
        # Shrinking at the floor is a no-op that could loop forever.
        assert (
            self._plan(DEEPSEEK_V4, downgrade_used=True, floor_reached=True, pending_total=3)
            is NextAction.RAISE_BUDGET
        )

    def test_requirements_only_scope_skips_split(self) -> None:
        # No endpoints: nothing to shrink — go to the budget raise.
        assert (
            self._plan(
                DEEPSEEK_V4,
                downgrade_used=True,
                pending_total=0,
                scope_size=0,
                floor_reached=True,
            )
            is NextAction.RAISE_BUDGET
        )
