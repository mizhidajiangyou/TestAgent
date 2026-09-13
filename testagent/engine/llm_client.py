"""
Unified LLM client supporting OpenAI and Azure OpenAI.

Features:
- Multi-model fallback: try the primary model, on failure try subsequent models.
- Per-model ``OpenAIClient`` instances with their own retry/timeout/usage.
- ``MultiModelLLMClient`` aggregates token usage across all sub-clients.
- ``secondary_client()`` returns a non-primary client for review; falls back
  to the primary with a warning when only one model is configured.
"""

import asyncio
import json
import logging
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast

import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError, OpenAI
from openai.types import ResponseFormatJSONObject
from openai.types.chat import ChatCompletion

from testagent.config.settings import Settings
from testagent.engine.model_profiles import (
    QWEN_3_8,
    ModelProfile,
    Outcome,
    RequestIntent,
    classify_response,
    compose_request,
    resolve_profile,
)

if TYPE_CHECKING:
    from openai import AzureOpenAI

logger = logging.getLogger(__name__)

#: Default max output tokens (can be overridden via OPENAI_MAX_OUTPUT_TOKENS).
DEFAULT_MAX_OUTPUT_TOKENS = 16000

#: Number of retry attempts for LLM calls.
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

#: OpenAI ``response_format`` value that forces a valid JSON object. Used by
#: the optional JSON mode (see ``OPENAI_JSON_MODE`` in settings). A bare JSON
#: array is NOT accepted by this mode, which is why test-case generation wraps
#: its payload in ``{"test_cases": [...]}`` when JSON mode is enabled.
JSON_OBJECT_FORMAT: dict[str, object] = {"type": "json_object"}

#: Interval (seconds) between "still waiting" progress logs during a single
#: blocking LLM call. A slow response (large ``max_tokens``) otherwise looks
#: like a hang; this makes the wait observable. Set to 0/negative to disable.
WAITING_LOG_INTERVAL = 30.0

#: Interval (seconds) between "thinking" progress logs while a reasoning
#: model streams reasoning tokens (no visible content yet). Without this a
#: long thinking phase — minutes at qwen defaults — is pure log silence and
#: looks exactly like a hung request.
THINKING_LOG_INTERVAL = 30.0

#: Per-call label (e.g. "Req REQ-001/3") set by the truncation engine so
#: client-side logs of CONCURRENT fan-out calls are attributable to their
#: batch/requirement. Propagates into ``asyncio.to_thread`` workers because
#: ``to_thread`` copies the calling context. Empty for direct callers
#: (conversation / gui / perf generators) — logs stay unchanged.
CALL_LABEL: ContextVar[str] = ContextVar("testagent_call_label", default="")

#: Hard wall-clock limit for ONE blocking call when the operator did not
#: configure one (plan-c B1.1): resolved as max(timeout * 2, this default) so
#: a legitimately slow completion (blocking ≈ thinking + full generation) is
#: not killed early. An EXPLICIT OPENAI_BLOCKING_HARD_TIMEOUT is honoured
#: verbatim (validated >= OPENAI_TIMEOUT at construction).
DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS = 600.0

#: Max hard-timeout retries within ONE ``_chat_core`` loop (plan-c B2.2).
#: Prevents 3 serial multi-minute waits on a genuinely wedged provider: after
#: this many timed-out attempts the error propagates to the caller (engine
#: recovery / model fallback) instead of burning the remaining attempts.
MAX_TIMEOUT_RETRIES_PER_CALL = 1

#: Consecutive TRANSIENT streaming failures before streaming is disabled for
#: the session (plan-c B2.1). A provider REJECTING streaming outright (HTTP
#: 400 mentioning "stream") disables it immediately; transient transport
#: drops only count toward this threshold.
MAX_STREAM_FAIL_STREAK = 3

#: Default multiple of max_concurrency for the abandoned-worker circuit
#: breaker (plan-c B1.2): when more abandoned (timed-out but still running)
#: worker threads than this accumulate, further retries are skipped in favour
#: of model fallback.
ABANDONED_WORKER_BREAKER_MULTIPLIER = 2

#: Transport-level exceptions counted toward the streaming de-stickiness
#: streak (plan-c B2.1, review #16). Deliberately NARROW: programming bugs
#: (KeyError / TypeError) are NOT here — they propagate as-is instead of
#: being masked as provider failures.
_TRANSIENT_STREAM_ERRORS: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
    httpx.ReadTimeout,
)


class ModelUnavailableError(RuntimeError):
    """Raised when a configured model cannot be reached or is not served.

    Used by the zero-token pre-flight :meth:`OpenAIClient.verify` so a
    misconfigured model / endpoint / API key fails fast (with a clear
    message) before the slow, token-consuming generation phase begins.
    """


class LLMOutputTooLongError(RuntimeError):
    """Raised when the model response is truncated at ``max_tokens`` and the
    partial content is unusable (empty).

    This signals the request is larger than the model's real output limit.
    Retrying the *identical* request cannot help, so it surfaces a specific
    error that the generator turns into a "generate fewer / more compact
    cases" re-ask rather than burning more identical attempts.
    """


class ReasoningBudgetExhaustedError(LLMOutputTooLongError):
    """Raised when reasoning tokens consumed the whole shared output budget
    and the visible answer came back EMPTY (plan v10 §4.3, P0-3).

    This is the ``finish_reason in profile.empty_finish_reasons`` + empty
    body fingerprint on a shared-budget profile (e.g. deepseek ``length``).
    Retrying the *identical* request re-runs the same reasoning and empties
    the budget again, so :meth:`OpenAIClient._chat_core` short-circuits on
    the FIRST occurrence instead of burning its internal retries. The
    engine-side recovery (one downgrade + split) reacts to this exception.

    Subclasses :class:`LLMOutputTooLongError` so existing handlers keep
    working; carries the observed evidence for logging/diagnosis.
    """

    def __init__(
        self,
        message: str,
        *,
        finish_reason: str | None = None,
        completion_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.completion_tokens = completion_tokens
        self.reasoning_tokens = reasoning_tokens


class LLMCallTimeoutError(RuntimeError):
    """Raised when a blocking LLM call exceeds the hard wall-clock limit
    (plan-c B2.2).

    Semantically DISTINCT from :class:`ReasoningBudgetExhaustedError`: a
    timeout means NO response arrived at all (wedged provider / dropped
    connection), not an empty-but-classified budget exhaustion — so the v10
    engine treats it as TRANSIENT (retry + model fallback), never as a
    downgrade trigger.

    The abandoned worker thread is a daemon and cannot be killed; it is
    registered on the client's abandoned-worker ledger (plan-c B1.2) and
    finishes on its own or dies with the process.
    """


@dataclass
class TokenUsage:
    """Cumulative token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def summary(self) -> str:
        """Return a human-readable usage summary."""
        return (
            f"requests={self.request_count}, prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, total_tokens={self.total_tokens}"
        )

    def add(self, other: TokenUsage) -> None:
        """Accumulate another usage record into this one."""
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.request_count += other.request_count


@dataclass
class LLMResponse:
    """Rich result returned by ``chat_with_meta`` / ``achat_with_meta``.

    ``chat()`` / ``achat()`` keep returning ``str`` for backward compatibility;
    the truncation-aware generator path uses this variant to obtain
    ``finish_reason`` / ``completion_tokens`` without breaking existing callers.
    """

    text: str
    finish_reason: str | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    """LLM client interface.

    The ``intent_override`` parameter (plan v10 §4.2, P0-2) is optional on
    every method: capability-detecting callers check ``intent_capable``
    before passing it, so legacy clients and test doubles that do not accept
    it keep working unchanged.
    """

    @property
    def intent_capable(self) -> bool:
        """True when the implementation accepts ``intent_override``."""
        ...

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> str:
        """Send a chat completion request.

        Args:
            system_prompt: System message.
            user_prompt: User message.
            response_format: Optional ``response_format`` payload forwarded to
                the underlying SDK (e.g. ``{"type": "json_object"}``).
            max_tokens: Optional per-call cap overriding the client default.
                The generator's retry loop passes a *smaller* value on later
                attempts after a truncated/empty response, so the model is
                asked for a more compact answer instead of re-failing.
            intent_override: Optional intent (effort tier / budget) overriding
                the client's default for this call — used by the engine's
                one-shot downgrade recovery.

        Returns:
            Model response text.
        """
        ...

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> str:
        """Async variant of :meth:`chat` (see implementations)."""
        ...

    def chat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Rich-result variant of :meth:`chat`.

        Returns an :class:`LLMResponse` carrying ``finish_reason`` and
        ``completion_tokens`` (when reported) so the truncation-aware
        generator can distinguish a genuine ``length`` truncation from a
        heuristic near-cap response without string sniffing.
        """
        ...

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Async variant of :meth:`chat_with_meta`."""
        ...

    def set_session_id(self, session_id: str) -> None:
        """Attach a session id so every log line for a run can be correlated.

        Enables "resume by id": the same id is printed at run start and saved
        with the run's inputs, so a failed/interrupted run can be re-triggered
        with the exact same configuration.
        """
        ...

    def verify(self) -> None:
        """Check the model is reachable without consuming tokens.

        Zero-token metadata call (see :meth:`OpenAIClient.verify`); raises
        :class:`ModelUnavailableError` when the model/endpoint/key is bad so
        the caller can fail fast before the slow generation phase.
        """
        ...

    async def averify(self) -> None:
        """Async variant of :meth:`verify`."""
        ...


class OpenAIClient:
    """OpenAI-compatible LLM client with retry, timeout and usage tracking.

    Holds a :class:`~testagent.engine.model_profiles.ModelProfile` resolved
    from the model name (or passed explicitly by :func:`create_llm_client`)
    and translates every request through
    :func:`~testagent.engine.model_profiles.compose_request`, so parameter
    dialects (budget param name, temperature semantics, effort fragments)
    are data, not branches.
    """

    #: Capability probe (plan v10 §4.2, P0-2): True → callers may pass
    #: ``intent_override`` on the four chat methods.
    intent_capable = True

    def __init__(
        self,
        client: OpenAI | AzureOpenAI,
        model: str,
        timeout: float = 300.0,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        profile: ModelProfile | None = None,
        reasoning_effort: str | None = None,
        continuation_intent: str | None = None,
        blocking_hard_timeout: float | None = None,
        stream_hard_timeout: float | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._timeout = timeout
        self._max_output_tokens = max_output_tokens
        self._profile = profile if profile is not None else resolve_profile(model)
        self._continuation_intent_override = continuation_intent
        # Raw passthrough fragment for models the registry does not know (see
        # LLMSettings.extra_body_json). The registry translates the effort
        # INTENT into family dialects, so an unregistered model gets no
        # thinking control at all; this lets the operator supply the exact
        # fragment their provider documents instead of us guessing it.
        self._extra_body: dict[str, Any] = dict(extra_body) if extra_body else {}
        self._default_intent = RequestIntent(
            budget=max_output_tokens,
            effort=reasoning_effort or None,
            deterministic=True,
        )
        # Hard wall-clock limit for a single blocking call (plan-c B1.1).
        # Unset -> max(timeout*2, DEFAULT) so a legitimately slow completion
        # is not killed; an EXPLICIT value is honoured verbatim but must be
        # >= the request timeout (else construction fails loudly — a hard
        # timeout below the per-request timeout is a configuration error).
        if blocking_hard_timeout is not None:
            if blocking_hard_timeout < timeout:
                raise ValueError(
                    f"blocking_hard_timeout ({blocking_hard_timeout}s) must be >= "
                    f"the request timeout ({timeout}s); a hard timeout below the "
                    "per-request timeout would abort every call."
                )
            self._blocking_hard_timeout = float(blocking_hard_timeout)
        else:
            self._blocking_hard_timeout = max(timeout * 2, DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS)
        # Same contract for the STREAMING channel. A stream that keeps
        # delivering reasoning chunks never trips the transport read timeout,
        # so without this cap the only bound on a single call was the
        # provider's own budget. Symmetry with the blocking path is
        # deliberate: identical default and identical validation.
        if stream_hard_timeout is not None:
            if stream_hard_timeout < timeout:
                raise ValueError(
                    f"stream_hard_timeout ({stream_hard_timeout}s) must be >= "
                    f"the request timeout ({timeout}s); a hard timeout below the "
                    "per-request timeout would abort every call."
                )
            self._stream_hard_timeout = float(stream_hard_timeout)
        else:
            self._stream_hard_timeout = max(timeout * 2, DEFAULT_BLOCKING_HARD_TIMEOUT_SECONDS)
        self.usage = TokenUsage()
        # Guards ``usage`` which may be mutated from multiple worker threads
        # when batches run concurrently via the async shim.
        self._usage_lock = threading.Lock()
        # Correlation id for this run, set by the generator so all log lines
        # (and the saved session record) share it. Enables resume-by-id.
        self._session_id: str | None = None
        # Whether to stream tokens (real-time progress logs). If the provider
        # rejects streaming (or ``stream_options``), we transparently fall
        # back to blocking mode for subsequent attempts.
        self._stream_enabled: bool = True
        # Consecutive TRANSIENT streaming failures (connection resets etc.,
        # plan-c B2.1). Resets on any successful stream; at
        # MAX_STREAM_FAIL_STREAK streaming is disabled for the session so a
        # persistently broken transport stops churning (one dropped chunked
        # read must NOT pin the whole session into blocking mode).
        self._stream_fail_streak = 0
        # Abandoned-worker ledger (plan-c B1.2): workers whose hard timeout
        # fired but whose SDK call may still be running. Entries are removed
        # when the worker eventually finishes, so the counts are live.
        self._abandoned_workers: set[int] = set()
        self._abandoned_ages: dict[int, float] = {}
        self._abandoned_lock = threading.Lock()
        # Count of blocking SDK workers currently being waited on.
        self._active_workers = 0

    @property
    def model_name(self) -> str:
        """Return the model name used by this client."""
        return self._model

    @property
    def max_output_tokens(self) -> int:
        """Return configured max output tokens."""
        return self._max_output_tokens

    @property
    def profile(self) -> ModelProfile:
        """Return this client's model profile."""
        return self._profile

    @property
    def continuation_intent(self) -> str | None:
        """Return the one-shot downgrade effort tier (plan v10 §7).

        Settings override (``OPENAI_CONTINUATION_REASONING_EFFORT``) wins over
        the profile's own default; None means "no downgrade capability".
        """
        if self._continuation_intent_override:
            return self._continuation_intent_override
        return self._profile.continuation_intent

    @property
    def max_output_cap(self) -> int | None:
        """Return the model's real output cap (None = unknown)."""
        return self._profile.max_output_cap

    @property
    def stream_hard_timeout(self) -> float:
        """Return the wall-clock cap (seconds) for a single streaming call."""
        return self._stream_hard_timeout

    def set_session_id(self, session_id: str) -> None:
        """Attach a session id used to tag every log line for this run."""
        self._session_id = session_id

    def set_stream_enabled(self, enabled: bool) -> None:
        """Enable or disable token streaming for this client."""
        self._stream_enabled = enabled
        self._stream_fail_streak = 0  # external override resets the streak

    def _abandoned_breaker_threshold(self) -> int:
        """Circuit-breaker threshold for abandoned workers (plan-c B1.2).

        Defaults to ABANDONED_WORKER_BREAKER_MULTIPLIER times 10 (a
        conservative stand-in for max_concurrency, which the client does not
        know); tests monkeypatch this method to simulate breaker trips.
        """
        return ABANDONED_WORKER_BREAKER_MULTIPLIER * 10

    def _register_transient_stream_failure(self, sid: str, exc: Exception) -> None:
        """Count one transient streaming failure toward the de-stickiness
        streak (plan-c B2.1). The current call falls back to blocking, but
        streaming is retried on the NEXT call until the streak hits
        MAX_STREAM_FAIL_STREAK (then disabled for the session).
        """
        self._stream_fail_streak += 1
        if self._stream_fail_streak >= MAX_STREAM_FAIL_STREAK:
            logger.warning(
                "[%s] streaming failed %d consecutive times for model '%s'; "
                "disabling streaming for this session.",
                sid,
                self._stream_fail_streak,
                self._model,
            )
            self._stream_enabled = False
        else:
            logger.warning(
                "[%s] streaming unavailable for model '%s' (%s); blocking mode "
                "for this call (streaming retries next call).",
                sid,
                self._model,
                exc,
            )

    # -- abandoned-worker observability (plan-c B1.2) ---------------------

    @property
    def abandoned_worker_count(self) -> int:
        """Live count of timed-out-but-still-running blocking workers."""
        with self._abandoned_lock:
            return len(self._abandoned_workers)

    def worker_metrics(self) -> dict[str, int | float]:
        """Resource-model metrics for logging/monitoring (plan-c B1.2).

        ``active_llm_workers`` counts blocking SDK workers currently being
        waited on; ``abandoned_llm_workers`` counts workers whose hard
        timeout fired but whose SDK call may still hold a thread + HTTP
        connection; ``abandoned_llm_workers_oldest_age`` is the age in
        seconds of the oldest abandoned entry (0 when none).
        """
        with self._abandoned_lock:
            abandoned = len(self._abandoned_workers)
            oldest = (
                (time.monotonic() - min(self._abandoned_ages.values()))
                if self._abandoned_ages
                else 0.0
            )
        active = max(0, self._active_workers - abandoned)
        return {
            "active_llm_workers": active,
            "abandoned_llm_workers": abandoned,
            "abandoned_llm_workers_oldest_age": round(oldest, 1),
        }

    def _chat_core(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Single source of truth for the retry / streaming / truncation loop.

        Both :meth:`chat` (``str`` contract) and :meth:`chat_with_meta``
        (rich :class:`LLMResponse`) delegate here; only the return shape
        differs. Truncation semantics (plan v10 §4.3):

        - ``finish_reason == "length"`` WITH partial content → the partial is
          returned (``finish_reason="length"``) so the caller can salvage it.
        - EMPTY content classified ``BUDGET_EXHAUSTED`` (reasoning ate the
          shared budget) → :class:`ReasoningBudgetExhaustedError` on the
          FIRST occurrence — retrying the identical request is guaranteed
          useless (P0-3), the engine-side recovery reacts to it.
        - empty content classified ``TRANSIENT_EMPTY`` → bounded internal
          retries (stream → blocking), then :class:`LLMOutputTooLongError`.
        """
        sid = self._session_id or "-"
        call_label = CALL_LABEL.get("")
        if call_label:
            # Renders as "[<sid>][<label>]" in the "[%s]" log format, making
            # interleaved logs of concurrent fan-out calls attributable.
            sid = f"{sid}][{call_label}"
        base_intent = intent_override if intent_override is not None else self._default_intent
        budget = max_tokens if (max_tokens and max_tokens > 0) else base_intent.budget
        intent = (
            base_intent
            if budget == base_intent.budget
            else RequestIntent(
                budget=budget, effort=base_intent.effort, deterministic=base_intent.deterministic
            )
        )
        last_error = ""
        timeout_attempts = 0
        truncated_hint = (
            " [IMPORTANT: Your previous response was truncated. "
            "Please make each case MORE COMPACT: shorter descriptions, "
            "fewer steps, concise expected_results. Fit within the token limit.]"
        )

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                effective_user_prompt = user_prompt
                if attempt > 1:
                    effective_user_prompt = user_prompt + truncated_hint

                # First attempt streams for live progress; retries drop to the
                # robust blocking channel so a flaky/empty streaming endpoint
                # cannot keep dropping the same request (see the root-cause
                # analysis for the empty finish_reason=length failure mode).
                force_blocking = attempt > 1
                channel = "stream" if (self._stream_enabled and not force_blocking) else "blocking"

                # ``create_kwargs`` is a passthrough to the OpenAI SDK, whose
                # ``response_format`` (and other params) are typed with strict
                # TypedDicts; a heterogeneous ``dict[str, object]`` cannot
                # satisfy those overloads when unpacked, so we use ``Any`` here.
                create_kwargs: dict[str, Any] = {
                    "model": self._model,
                    "timeout": self._timeout,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": effective_user_prompt},
                    ],
                    "stream": True,
                }
                # Profile-driven parameter dialect (plan v10 §4.1): budget
                # param name, temperature semantics and effort fragments are
                # declared data. response_format passes through untouched
                # (P0-1); the budget is clamped to the profile cap (P0-4).
                create_kwargs.update(
                    compose_request(
                        self._profile,
                        intent,
                        channel=channel,
                        response_format=response_format,
                    )
                )
                if self._extra_body:
                    # Operator-supplied fragment for a model the registry does
                    # not know (OPENAI_EXTRA_BODY_JSON). The profile's own
                    # fragment is the base; the operator's keys win.
                    merged_extra = dict(create_kwargs.get("extra_body") or {})
                    merged_extra.update(self._extra_body)
                    create_kwargs["extra_body"] = merged_extra
                if "response_format" in create_kwargs:
                    # The OpenAI SDK types ``response_format`` as a strict
                    # TypedDict (ResponseFormatJSONObject); ``cast`` is a
                    # no-op at runtime, it only satisfies the static checker.
                    create_kwargs["response_format"] = cast(
                        ResponseFormatJSONObject, create_kwargs["response_format"]
                    )

                logger.info(
                    "[%s] → model '%s' (attempt %d/%d, max_tokens=%d, stream=%s)",
                    sid,
                    self._model,
                    attempt,
                    MAX_RETRIES,
                    intent.budget,
                    (self._stream_enabled and not force_blocking),
                )

                content, finish_reason, usage, reasoning_chars = self._complete(
                    create_kwargs, force_blocking=force_blocking, sid=sid
                )
                if usage is not None:
                    self._record_usage(usage)
                content_str = content or ""
                completion_tokens = usage.completion_tokens if usage is not None else None

                logger.info(
                    "[%s] ← model '%s' returned %d chars (finish_reason=%s%s)",
                    sid,
                    self._model,
                    len(content_str),
                    finish_reason,
                    f", reasoning={reasoning_chars} chars" if reasoning_chars else "",
                )

                outcome = classify_response(
                    self._profile, text=content_str, finish_reason=finish_reason
                )

                if outcome is Outcome.BUDGET_EXHAUSTED:
                    # P0-3: reasoning consumed the whole shared budget and the
                    # visible answer is empty. Retrying the identical request
                    # re-runs the same reasoning — short-circuit NOW instead of
                    # burning the internal retries; the typed error triggers
                    # model fallback and the engine's one-shot downgrade.
                    reasoning_tokens = getattr(
                        getattr(usage, "completion_tokens_details", None)
                        if usage is not None
                        else None,
                        "reasoning_tokens",
                        None,
                    )
                    logger.warning(
                        "[%s] Reasoning budget exhausted (finish_reason=%s, "
                        "completion_tokens=%s, reasoning_tokens=%s); "
                        "short-circuiting identical retries (attempt %d/%d).",
                        sid,
                        finish_reason,
                        completion_tokens,
                        reasoning_tokens,
                        attempt,
                        MAX_RETRIES,
                    )
                    raise ReasoningBudgetExhaustedError(
                        f"Model '{self._model}' returned EMPTY content with "
                        f"finish_reason={finish_reason} for session {sid}: "
                        f"reasoning tokens consumed the shared output budget "
                        f"(completion_tokens={completion_tokens}, "
                        f"reasoning_tokens={reasoning_tokens}). Retrying the "
                        f"identical request would fail the same way; the "
                        f"engine recovery (one downgrade + split) takes over.",
                        finish_reason=finish_reason,
                        completion_tokens=completion_tokens,
                        reasoning_tokens=reasoning_tokens,
                    )

                if outcome is Outcome.TRUNCATED_PARTIAL:
                    # Genuine truncation WITH partial content: return it so the
                    # caller's _extract_json / _salvage_truncated_json can
                    # recover what is there.
                    logger.warning(
                        "[%s] Response truncated at budget=%d (attempt %d/%d). "
                        "Returning partial content for salvage.",
                        sid,
                        intent.budget,
                        attempt,
                        MAX_RETRIES,
                    )
                    return LLMResponse(
                        text=content_str.strip(),
                        finish_reason="length",
                        completion_tokens=completion_tokens,
                    )

                if (
                    not content_str.strip()
                    and reasoning_chars > 0
                    and finish_reason == "length"
                    and not self._profile.budget_shared
                ):
                    # Unknown-model blind spot (2026-09-13): the response is
                    # empty because reasoning consumed the whole budget, but
                    # the profile has no budget-sharing knowledge, so
                    # classify_response can never return BUDGET_EXHAUSTED and
                    # the v10 short-circuit does not apply.
                    #
                    # Gate on IN-BAND evidence (empty + length + reasoning was
                    # actually streamed) instead of the profile flag: that
                    # combination is the same signal v10 keys on. The transient
                    # empty-stream recovery (stream -> blocking) is untouched —
                    # it targets stop/None empties, which fall through below.
                    logger.warning(
                        "[%s] Model '%s' returned NO visible output after %d chars "
                        "of reasoning (finish_reason=length, budget=%d). Profile "
                        "'%s' does not declare a shared budget, so the "
                        "budget-exhausted short-circuit is unavailable; failing "
                        "fast instead of re-running the same reasoning. Cap the "
                        "thinking with OPENAI_EXTRA_BODY_JSON (e.g. "
                        "'{\"enable_thinking\": false}') or switch to a model in "
                        "the profile registry (listed in the resolution warning "
                        "at startup), where OPENAI_REASONING_EFFORT translates "
                        "to a real bound.",
                        sid,
                        self._model,
                        reasoning_chars,
                        intent.budget,
                        self._profile.name,
                    )
                    raise LLMOutputTooLongError(
                        f"Model '{self._model}' produced {reasoning_chars} chars of "
                        f"reasoning and NO visible output (finish_reason=length, "
                        f"budget={intent.budget}). The reasoning consumed the whole "
                        f"output budget. Retrying the identical request would "
                        f"re-run the same reasoning, so this fails fast into the "
                        f"engine recovery (split / smaller scope)."
                    )

                if not content_str.strip():
                    # TRANSIENT_EMPTY: the profile says this empty body cannot
                    # be reasoning starvation (non-shared budget or a
                    # non-exhausting finish reason). Retries drop to the
                    # blocking channel (see _complete force_blocking) so a
                    # flaky/empty streaming endpoint cannot keep dropping the
                    # same request — the classic empty-stream recovery.
                    last_error = "Empty response from model" + (
                        " (finish_reason=length)" if finish_reason == "length" else ""
                    )
                    logger.warning("[%s] Attempt %d/%d: %s", sid, attempt, MAX_RETRIES, last_error)
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                        continue
                    # Exhausted every transport: surface a clear, actionable
                    # error instead of the old misleading "too long / do not
                    # lower max_tokens" message.
                    raise LLMOutputTooLongError(
                        f"Model '{self._model}' returned an EMPTY response "
                        f"(finish_reason={finish_reason}) for session {sid} after "
                        f"{MAX_RETRIES} attempts (both streaming and blocking "
                        f"channels). This is an empty/aborted response from the "
                        f"provider — NOT a token-budget overflow (a real overflow "
                        f"would carry partial content, which is salvaged "
                        f"separately). Common causes: the streaming endpoint drops "
                        f"empty streams for this model, the model declined/refused "
                        f"to answer, or the provider does not actually serve this "
                        f"model id. Recovery steps: set OPENAI_STREAM=false to use "
                        f"the blocking endpoint, configure a secondary model "
                        f"(OPENAI_MODEL=a,b), or verify OPENAI_BASE_URL serves "
                        f"'{self._model}'. Lowering OPENAI_MAX_OUTPUT_TOKENS will "
                        f"NOT fix an empty response (it would only make a genuine "
                        f"overflow more likely)."
                    )
                return LLMResponse(
                    text=content_str.strip(),
                    finish_reason=finish_reason,
                    completion_tokens=completion_tokens,
                )
            except LLMOutputTooLongError:
                # Specific, fast-fail: let the generator re-ask with a smaller
                # scope instead of retrying the identical (too-large) request.
                raise
            except LLMCallTimeoutError as exc:
                # Bounded timeout budget (plan-c B2.2): a wedged provider may
                # time out every attempt; cap how many this loop spends before
                # surfacing to the engine (transient path — retry + model
                # fallback, NEVER a v10 downgrade trigger). Both channels raise
                # this type now: the blocking one via blocking_hard_timeout,
                # the streaming one via stream_hard_timeout.
                timeout_attempts += 1
                last_error = str(exc)
                logger.warning(
                    "[%s] Attempt %d/%d hit the hard timeout (channel=%s).",
                    sid,
                    attempt,
                    MAX_RETRIES,
                    channel,
                )
                if timeout_attempts > MAX_TIMEOUT_RETRIES_PER_CALL or attempt >= MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            except Exception as exc:
                # Abandoned-worker circuit breaker (plan-c B1.2): when the
                # residue of timed-out-but-still-running workers already
                # exceeds the breaker threshold, further retries would only
                # pile up threads/connections — surface the error now so the
                # multi-model fallback layer takes over.
                if self.abandoned_worker_count > self._abandoned_breaker_threshold():
                    logger.warning(
                        "[%s] abandoned worker breaker tripped (%d > %d); "
                        "surfacing error for model fallback.",
                        sid,
                        self.abandoned_worker_count,
                        self._abandoned_breaker_threshold(),
                    )
                    raise
                last_error = str(exc)
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s", sid, attempt, MAX_RETRIES, last_error
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(
            f"Failed to get LLM response after {MAX_RETRIES} attempts. Last error: {last_error}"
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> str:
        """Send a chat completion request (``str`` contract, unchanged).

        Streams by default for live progress, retries with backoff, salvages
        truncated partials, and raises :class:`LLMOutputTooLongError` on empty
        responses. See :meth:`_chat_core` for the full semantics; this is a
        thin wrapper returning only ``.text``.
        """
        return self._chat_core(
            system_prompt, user_prompt, response_format, max_tokens, intent_override
        ).text

    def chat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Truncation-aware variant: expose finish_reason / completion_tokens."""
        return self._chat_core(
            system_prompt, user_prompt, response_format, max_tokens, intent_override
        )

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> str:
        """Async variant of :meth:`chat`.

        Runs the (sync, blocking) :meth:`chat` in the default executor so the
        caller's event loop stays responsive. Used by the async generation
        path (``TestCaseGenerator.agenerate``), which fans out many batches
        concurrently via ``asyncio.to_thread``.
        """
        return await asyncio.to_thread(
            self.chat, system_prompt, user_prompt, response_format, max_tokens, intent_override
        )

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Async variant of :meth:`chat_with_meta` (same ``to_thread`` shim).

        Kept source-identical to :meth:`achat` on purpose: the sync SDK core
        has no native async channel, so ``to_thread`` *is* what makes the
        concurrent fan-out possible. If the core ever moves to
        ``openai.AsyncOpenAI``, replace the body with a native ``await`` on
        the async core (the pre-arranged upgrade seam).
        """
        return await asyncio.to_thread(
            self.chat_with_meta,
            system_prompt,
            user_prompt,
            response_format,
            max_tokens,
            intent_override,
        )

    def verify(self) -> None:
        """Verify the model is reachable WITHOUT consuming tokens.

        Issues a ``GET /v1/models/{model}`` metadata request (zero token
        cost). Raises :class:`ModelUnavailableError` if the model cannot be
        retrieved — e.g. bad API key, wrong ``base_url``, or the model id is
        unknown to the endpoint — so callers can fail fast instead of hanging
        for minutes on a generation that can never succeed.

        Azure OpenAI clients are skipped: Azure does not reliably expose the
        ``/models`` API, so a zero-token preflight is not available there and
        we avoid a false negative.
        """
        if getattr(self._client, "azure_endpoint", None) is not None:
            logger.info(
                "Skipping zero-token model verification for Azure endpoint "
                "(model=%s); Azure does not expose a reliable /models API.",
                self._model,
            )
            return
        try:
            model = self._client.models.retrieve(self._model)
            model_id = getattr(model, "id", self._model)
            logger.info(
                "[%s] Model '%s' is available (id=%s).",
                self._session_id or "-",
                self._model,
                model_id,
            )
        except Exception as exc:
            logger.error("Model '%s' verification failed: %s", self._model, exc)
            raise ModelUnavailableError(
                f"Model '{self._model}' is not available: {exc}. "
                "Check OPENAI_API_KEY, OPENAI_BASE_URL and the model name."
            ) from exc

    async def averify(self) -> None:
        """Async variant of :meth:`verify` (runs the sync call in a thread)."""
        await asyncio.to_thread(self.verify)

    def _blocking_create(self, create_kwargs: dict[str, Any], sid: str = "-") -> ChatCompletion:
        """Run the (blocking) chat completion while emitting progress logs.

        The OpenAI SDK call blocks until the full response is streamed back.
        We run it in a daemon worker thread and poll from the calling thread
        so we can emit a periodic ``still waiting (Xs, model=...)`` log when a
        single call is slow. This makes long generations observable instead of
        a silent black screen. It works identically whether ``chat`` is invoked
        synchronously or from inside ``asyncio.to_thread`` (the async shim),
        because the polling happens on whatever thread called ``chat``.

        Hard timeout (plan-c B2.2): when the wall-clock limit fires, the
        caller regains control immediately via :class:`LLMCallTimeoutError`.
        The worker itself is a daemon and cannot be killed — it is registered
        on the abandoned-worker ledger (plan-c B1.2) and removes itself when
        it eventually finishes, so :meth:`worker_metrics` stays live.
        """
        holder: dict[str, ChatCompletion] = {}
        error: dict[str, BaseException] = {}
        # The worker reports its own thread ident (threading.Thread.ident is
        # Optional until start(); inside the worker get_ident() is exact).
        ident_holder: dict[str, int] = {}

        def _run() -> None:
            ident_holder["id"] = threading.get_ident()
            try:
                holder["resp"] = self._client.chat.completions.create(**create_kwargs)
            except Exception as exc:  # re-raised in the caller thread
                error["exc"] = exc
            finally:
                # Worker finished (success, error or post-timeout): remove it
                # from the abandoned ledger so the metrics decay to zero.
                wid = ident_holder.get("id")
                if wid is not None:
                    with self._abandoned_lock:
                        self._abandoned_workers.discard(wid)
                        self._abandoned_ages.pop(wid, None)

        worker = threading.Thread(target=_run, daemon=True)
        with self._abandoned_lock:
            self._active_workers += 1
        worker.start()
        try:
            interval = WAITING_LOG_INTERVAL
            elapsed = 0.0
            timed_out = False
            while worker.is_alive():
                worker.join(timeout=interval)
                if worker.is_alive():
                    elapsed += interval
                    logger.info(
                        "[%s] still waiting (%ds, model=%s) ...", sid, int(elapsed), self._model
                    )
                    if elapsed >= self._blocking_hard_timeout:
                        timed_out = True
                        break

            if timed_out:
                # The worker keeps running (daemon, unkilleable) and its HTTP
                # connection stays open until the provider answers or the
                # socket dies; the caller regains control NOW. Register the
                # worker so the resource metrics reflect the residue.
                wid = ident_holder.get("id")
                if wid is not None:
                    with self._abandoned_lock:
                        self._abandoned_workers.add(wid)
                        self._abandoned_ages.setdefault(wid, time.monotonic())
                raise LLMCallTimeoutError(
                    f"Blocking LLM call for model '{self._model}' exceeded the "
                    f"{self._blocking_hard_timeout:.0f}s hard timeout with no "
                    f"response; abandoning this attempt."
                )

            if "exc" in error:
                raise error["exc"]
            if "resp" not in holder:
                raise RuntimeError("LLM call worker thread terminated without a response")
            return holder["resp"]
        finally:
            with self._abandoned_lock:
                self._active_workers -= 1

    def _record_usage(self, usage: Any) -> None:
        """Accumulate token usage from a usage object (thread-safe)."""
        if usage is None:
            return
        with self._usage_lock:
            self.usage.request_count += 1
            self.usage.prompt_tokens += usage.prompt_tokens or 0
            self.usage.completion_tokens += usage.completion_tokens or 0
            self.usage.total_tokens += usage.total_tokens or 0

    def _complete(
        self,
        create_kwargs: dict[str, Any],
        force_blocking: bool = False,
        sid: str = "-",
    ) -> tuple[str, str | None, Any, int]:
        """Run a chat completion and normalize the result.

        Streams tokens when ``self._stream_enabled`` (real-time progress), and
        transparently falls back to a blocking call if streaming fails (e.g. the
        provider rejects ``stream_options``) or when ``force_blocking`` is set.
        ``force_blocking`` is used by :meth:`chat` on retry attempts to dodge a
        flaky/empty streaming channel (see the empty ``finish_reason=length``
        failure mode). Returns ``(content, finish_reason, usage,
        reasoning_chars)`` where ``usage`` may be ``None`` if the provider did
        not report it and ``reasoning_chars`` counts streamed/returned
        reasoning tokens (0 when the model does not think).
        """
        if self._stream_enabled and not force_blocking:
            try:
                result = self._stream_completion(create_kwargs, sid=sid)
                self._stream_fail_streak = 0  # healthy stream resets streak
                return result
            except BadRequestError as exc:
                # A 400 MAY mean the provider rejects streaming outright, but
                # it may equally be an unrelated parameter error (plan-c B2.1,
                # review #17): only treat it as "streaming unsupported" when
                # the error message mentions stream; otherwise fall through to
                # the generic transient path (this call only).
                if "stream" in str(exc).lower():
                    logger.warning(
                        "[%s] provider rejected streaming for model '%s' (%s); "
                        "disabling streaming for this session.",
                        sid,
                        self._model,
                        exc,
                    )
                    self._stream_enabled = False
                else:
                    self._register_transient_stream_failure(sid, exc)
            except _TRANSIENT_STREAM_ERRORS as exc:
                # Transport-level failure (connection reset / read timeout /
                # incomplete chunked read): blocking fallback for THIS call
                # only; streaming retries next call (plan-c B2.1).
                self._register_transient_stream_failure(sid, exc)
            # Programming errors (KeyError / TypeError / ...) propagate as-is:
            # they are NOT provider failures and must not be masked by a
            # silent blocking fallback (review #16).

        # Blocking fallback path.
        blocking_kwargs = {k: v for k, v in create_kwargs.items() if k != "stream"}
        blocking_kwargs["stream"] = False
        resp = self._blocking_create(blocking_kwargs, sid=sid)
        choice = resp.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason
        usage = resp.usage
        # Reasoning models put the thinking phase on message.reasoning_content
        # in blocking mode; count it for the same observability as streaming.
        reasoning_content = getattr(choice.message, "reasoning_content", None) or ""
        reasoning_chars = len(reasoning_content) if isinstance(reasoning_content, str) else 0
        return content, finish_reason, usage, reasoning_chars

    def _stream_completion(
        self, create_kwargs: dict[str, Any], sid: str = "-"
    ) -> tuple[str, str | None, Any, int]:
        """Stream a chat completion and return ``(content, finish_reason, usage,
        reasoning_chars)``.

        Emits throttled live progress (so the operator sees tokens arrive in
        real time instead of a black screen) and runs a ``still waiting``
        watchdog if the stream stalls. Reasoning tokens (``reasoning_content``)
        stream BEFORE the visible content on thinking models; their progress
        is logged too (every ``THINKING_LOG_INTERVAL``), otherwise a multi-
        minute thinking phase is indistinguishable from a hang. Usage is read
        from the final chunk when the provider supports ``stream_options``;
        otherwise it is ``None``.
        """
        kwargs: dict[str, Any] = dict(create_kwargs)
        kwargs["stream"] = True
        # Ask for cumulative usage in the last chunk. Some OpenAI-compatible
        # gateways ignore this; we tolerate a missing usage.
        kwargs["stream_options"] = {"include_usage": True}

        state: dict[str, Any] = {"stop": False, "last_chunk": time.time()}
        content_parts: list[str] = []
        usage: Any = None
        finish_reason: str | None = None
        streamed_chars = 0
        reasoning_chars = 0
        thinking_logged = False
        last_log = time.time()
        last_think_log = time.time()
        # Wall-clock deadline for THIS call. Checked on every chunk so a stream
        # that keeps trickling reasoning tokens (which keeps the transport
        # read timeout from ever firing) is still bounded.
        started = time.monotonic()

        def _watchdog() -> None:
            while not state["stop"]:
                time.sleep(WAITING_LOG_INTERVAL)
                if time.time() - state["last_chunk"] >= WAITING_LOG_INTERVAL:
                    logger.info(
                        "[%s] still waiting (model=%s, streamed=%d chars, thinking=%d chars) ...",
                        sid,
                        self._model,
                        streamed_chars,
                        reasoning_chars,
                    )

        watcher = threading.Thread(target=_watchdog, daemon=True)
        watcher.start()
        stream: Any = None
        try:
            stream = self._client.chat.completions.create(**kwargs)
            for chunk in stream:
                state["last_chunk"] = time.time()
                elapsed = time.monotonic() - started
                if elapsed >= self._stream_hard_timeout:
                    # Hand the connection back before unwinding: the caller
                    # regains control NOW (LLMCallTimeoutError goes into the
                    # bounded timeout-retry path in _chat_core) instead of
                    # waiting on a provider that may never finish.
                    raise LLMCallTimeoutError(
                        f"Streaming LLM call for model '{self._model}' exceeded the "
                        f"{self._stream_hard_timeout:.0f}s hard timeout "
                        f"(streamed={streamed_chars} chars, thinking={reasoning_chars} "
                        f"chars); abandoning this attempt."
                    )
                if not chunk.choices:
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta is not None:
                    if delta.content:
                        content_parts.append(delta.content)
                        streamed_chars += len(delta.content)
                    # Thinking models (qwen / deepseek / kimi / glm) stream
                    # reasoning tokens on a separate field BEFORE any visible
                    # content. They do NOT count as output, but their progress
                    # IS logged at INFO — otherwise the whole thinking phase is
                    # log silence (DEBUG is hidden at the default level and the
                    # stall watchdog stays suppressed while chunks arrive),
                    # which is indistinguishable from a hang. This is the
                    # 2026-09-13 "Phase 2 frozen for minutes" report.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        if not thinking_logged:
                            thinking_logged = True
                            logger.info(
                                "[%s] %s is thinking (reasoning tokens stream "
                                "first; visible output follows)...",
                                sid,
                                self._model,
                            )
                        reasoning_chars += len(reasoning)
                        now = time.time()
                        if now - last_think_log >= THINKING_LOG_INTERVAL:
                            last_think_log = now
                            logger.info(
                                "[%s] %s still thinking (%ds elapsed, %d chars of "
                                "reasoning, no visible output yet) ...",
                                sid,
                                self._model,
                                int(elapsed),
                                reasoning_chars,
                            )
                if choice.finish_reason:
                    finish_reason = choice.finish_reason
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
                now = time.time()
                if now - last_log >= 3.0 and streamed_chars:
                    last_log = now
                    logger.debug(
                        "[%s] %s streaming: %d chars so far...",
                        sid,
                        self._model,
                        streamed_chars,
                    )
            return "".join(content_parts), finish_reason, usage, reasoning_chars
        finally:
            state["stop"] = True
            watcher.join(timeout=1.0)
            # Release the HTTP connection when we bail out early (hard
            # timeout); on the normal path the iterator is already exhausted.
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:  # pragma: no cover - best-effort cleanup
                    logger.debug("[%s] stream close failed (ignored)", sid)


class MultiModelLLMClient:
    """LLM client that tries multiple models in order with automatic fallback.

    The first model in ``clients`` is the primary. If ``chat()`` raises after
    all internal retries, the next client is tried. Token usage is aggregated
    across all sub-clients so callers see a single total.

    ``secondary_client()`` returns the first non-primary client (preferred for
    review cross-validation). When only one model is configured, the primary
    is returned and a warning is logged so the operator knows the
    "cross-validation" is effectively single-model.
    """

    def __init__(self, clients: list[OpenAIClient]) -> None:
        if not clients:
            raise ValueError("MultiModelLLMClient requires at least one OpenAIClient")
        self._clients: list[OpenAIClient] = list(clients)
        # Aggregate usage snapshot taken lazily to avoid double-counting.
        self._aggregate_usage = TokenUsage()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def available_models(self) -> list[str]:
        """Return the ordered list of model names configured."""
        return [c.model_name for c in self._clients]

    @property
    def primary_model(self) -> str:
        """Return the primary (first) model name."""
        return self._clients[0].model_name

    @property
    def has_secondary(self) -> bool:
        """Return True when at least one non-primary model is configured."""
        return len(self._clients) > 1

    @property
    def clients(self) -> list[OpenAIClient]:
        """Return the underlying per-model clients (read-only view)."""
        return list(self._clients)

    @property
    def max_output_tokens(self) -> int:
        """Return the primary client's configured max output tokens."""
        return self._clients[0].max_output_tokens

    @property
    def profile(self) -> ModelProfile:
        """Return the primary model's profile (intent dialect, plan v10 §4.2)."""
        return self._clients[0].profile

    @property
    def continuation_intent(self) -> str | None:
        """Return the primary model's one-shot downgrade effort tier."""
        return self._clients[0].continuation_intent

    @property
    def max_output_cap(self) -> int | None:
        """Return the primary model's real output cap (None = unknown)."""
        return self._clients[0].max_output_cap

    @property
    def intent_capable(self) -> bool:
        """True when the underlying clients accept ``intent_override``."""
        return bool(self._clients) and bool(getattr(self._clients[0], "intent_capable", False))

    def set_session_id(self, session_id: str) -> None:
        """Propagate the session id to every sub-client for log correlation."""
        for client in self._clients:
            client.set_session_id(session_id)

    def set_stream_enabled(self, enabled: bool) -> None:
        """Enable or disable streaming on every sub-client."""
        for client in self._clients:
            client.set_stream_enabled(enabled)

    # ------------------------------------------------------------------
    # Aggregated usage
    # ------------------------------------------------------------------

    @property
    def usage(self) -> TokenUsage:
        """Aggregate token usage across all sub-clients.

        Each sub-client already accumulates its own usage; we sum them on
        every access so callers always see a fresh total. The internal
        ``_aggregate_usage`` is kept only for API compatibility with code
        that mutates ``usage`` directly.
        """
        total = TokenUsage()
        for c in self._clients:
            total.add(c.usage)
        # Preserve identity so external mutations still work in tests
        self._aggregate_usage.prompt_tokens = total.prompt_tokens
        self._aggregate_usage.completion_tokens = total.completion_tokens
        self._aggregate_usage.total_tokens = total.total_tokens
        self._aggregate_usage.request_count = total.request_count
        return self._aggregate_usage

    # ------------------------------------------------------------------
    # Chat with fallback
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> str:
        """Call the primary model; on failure, fall back to subsequent models.

        Each sub-client already performs internal retries with backoff. We
        only escalate to the next model once a sub-client exhausts its
        retries and raises ``RuntimeError``.

        Args:
            response_format: Optional ``response_format`` payload forwarded to
                every sub-client (see :meth:`OpenAIClient.chat`).
            max_tokens: Optional per-call cap forwarded to every sub-client.
            intent_override: Optional intent forwarded to every sub-client
                (plan v10 §4.2, P0-2 — this is the layer the v9 review missed).
        """
        last_error: Exception | None = None
        for idx, client in enumerate(self._clients):
            label = "primary" if idx == 0 else f"fallback #{idx}"
            try:
                logger.debug("Trying %s model: %s", label, client.model_name)
                return client.chat(
                    system_prompt, user_prompt, response_format, max_tokens, intent_override
                )
            except Exception as exc:
                last_error = exc
                if idx < len(self._clients) - 1:
                    logger.warning(
                        "%s model '%s' failed (%s); falling back to next model.",
                        label,
                        client.model_name,
                        exc,
                    )
                else:
                    logger.error(
                        "%s model '%s' failed (%s); no more fallback candidates.",
                        label,
                        client.model_name,
                        exc,
                    )

        return self._raise_all_failed(last_error)

    async def achat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> str:
        """Async variant of :meth:`chat` with identical fallback semantics.

        Each sub-client's ``achat`` is awaited in order; on failure the next
        model is tried, mirroring :meth:`chat`. Designed to be driven by
        ``asyncio.to_thread`` from ``TestCaseGenerator.agenerate`` so many
        batches run concurrently without blocking the event loop.
        """
        last_error: Exception | None = None
        for idx, client in enumerate(self._clients):
            label = "primary" if idx == 0 else f"fallback #{idx}"
            try:
                logger.debug("Trying %s model (async): %s", label, client.model_name)
                return await client.achat(
                    system_prompt, user_prompt, response_format, max_tokens, intent_override
                )
            except Exception as exc:
                last_error = exc
                if idx < len(self._clients) - 1:
                    logger.warning(
                        "%s model '%s' failed (async); falling back to next model.",
                        label,
                        client.model_name,
                    )
                else:
                    logger.error(
                        "%s model '%s' failed (async); no more fallback candidates.",
                        label,
                        client.model_name,
                    )

        return self._raise_all_failed(last_error, async_label=True)

    def chat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Rich-result variant of :meth:`chat` with identical fallback order."""
        last_error: Exception | None = None
        for idx, client in enumerate(self._clients):
            label = "primary" if idx == 0 else f"fallback #{idx}"
            try:
                logger.debug("Trying %s model (meta): %s", label, client.model_name)
                return client.chat_with_meta(
                    system_prompt, user_prompt, response_format, max_tokens, intent_override
                )
            except Exception as exc:
                last_error = exc
                if idx < len(self._clients) - 1:
                    logger.warning(
                        "%s model '%s' failed (%s); falling back to next model.",
                        label,
                        client.model_name,
                        exc,
                    )
                else:
                    logger.error(
                        "%s model '%s' failed (%s); no more fallback candidates.",
                        label,
                        client.model_name,
                        exc,
                    )
        return self._raise_all_failed(last_error)

    async def achat_with_meta(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, object] | None = None,
        max_tokens: int | None = None,
        intent_override: RequestIntent | None = None,
    ) -> LLMResponse:
        """Async variant of :meth:`chat_with_meta` with identical fallback order."""
        last_error: Exception | None = None
        for idx, client in enumerate(self._clients):
            label = "primary" if idx == 0 else f"fallback #{idx}"
            try:
                logger.debug("Trying %s model (async meta): %s", label, client.model_name)
                return await client.achat_with_meta(
                    system_prompt, user_prompt, response_format, max_tokens, intent_override
                )
            except Exception as exc:
                last_error = exc
                if idx < len(self._clients) - 1:
                    logger.warning(
                        "%s model '%s' failed (async); falling back to next model.",
                        label,
                        client.model_name,
                    )
                else:
                    logger.error(
                        "%s model '%s' failed (async); no more fallback candidates.",
                        label,
                        client.model_name,
                    )
        return self._raise_all_failed(last_error, async_label=True)

    def _raise_all_failed(
        self, last_error: Exception | None, *, async_label: bool = False
    ) -> NoReturn:
        """Surface the failure after every fallback model was tried.

        Typed truncation errors (:class:`ReasoningBudgetExhaustedError` /
        :class:`LLMOutputTooLongError`) are re-raised AS-IS so the engine's
        recovery ladder can classify them — wrapping them in a plain
        ``RuntimeError`` silently disabled the recovery path in production
        (only test fakes raising directly ever hit it). Generic errors keep
        the legacy aggregated ``RuntimeError`` message.
        """
        suffix = " (async)" if async_label else ""
        if isinstance(last_error, LLMOutputTooLongError):
            raise last_error
        raise RuntimeError(
            f"All {len(self._clients)} model(s) failed{suffix}. Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Zero-token pre-flight verification
    # ------------------------------------------------------------------

    def verify(self) -> None:
        """Verify every configured model is reachable (zero token).

        Verifies each sub-client in order. A failure of the **primary** (or
        any model when there is only one) raises :class:`ModelUnavailableError`
        so the caller fails fast. A failure of a **fallback** model only logs
        a warning — that model is only consulted when the primary fails, so a
        misconfigured fallback should not abort otherwise-valid generation.
        """
        for idx, client in enumerate(self._clients):
            label = "primary" if idx == 0 else f"fallback #{idx}"
            try:
                client.verify()
            except ModelUnavailableError:
                if idx == 0 or len(self._clients) == 1:
                    raise
                logger.warning(
                    "%s model '%s' failed verification; it will be skipped until "
                    "the primary model fails. Fix its config to enable fallback.",
                    label,
                    client.model_name,
                )

    async def averify(self) -> None:
        """Async variant of :meth:`verify`."""
        await asyncio.to_thread(self.verify)

    # ------------------------------------------------------------------
    # Secondary client for review
    # ------------------------------------------------------------------

    def secondary_client(self) -> OpenAIClient:
        """Return a non-primary client for cross-validation review.

        Falls back to the primary client (with a warning) when only one
        model is configured, so review can still proceed. Callers should
        inspect ``has_secondary`` or check the returned client's
        ``model_name`` against ``primary_model`` to detect the fallback.
        """
        if len(self._clients) > 1:
            return self._clients[1]
        logger.warning(
            "Only one model ('%s') is configured; review will reuse the primary "
            "model. Cross-validation will be single-model only. "
            "Set OPENAI_MODEL to a comma-separated list (e.g. "
            "'gpt-4o-mini,gpt-4o') to enable true multi-model review.",
            self._clients[0].model_name,
        )
        return self._clients[0]


def _resolve_profile_for(
    model_name: str,
    settings: Settings,
) -> ModelProfile:
    """Resolve the profile for one model, applying settings-driven overrides.

    - ``OPENAI_MODEL_PROFILE`` explicit > matcher > generic fallback
      (resolution itself logs and warns UNVERIFIED profiles);
    - Qwen profiles get their "low" thinking_budget replaced by
      ``OPENAI_CONTINUATION_THINKING_BUDGET``.
    """
    profile = resolve_profile(model_name, explicit=settings.llm.model_profile or None)
    if profile.name == QWEN_3_8.name:
        thinking_budget = settings.llm.continuation_thinking_budget
        translation = {
            tier: dict(fragment) for tier, fragment in QWEN_3_8.effort_translation.items()
        }
        translation["low"] = {"extra_body": {"thinking_budget": thinking_budget}}
        return replace(profile, effort_translation=translation)
    return profile


def _parse_extra_body(raw: str) -> dict[str, Any]:
    """Parse ``OPENAI_EXTRA_BODY_JSON`` into a request fragment.

    The profile registry translates the effort INTENT into each model family's
    dialect, so a model it does not recognise gets NO thinking control at all.
    Rather than guessing a vendor's parameter names, the operator supplies the
    exact fragment their provider documents.

    Args:
        raw: Raw JSON string (empty = feature disabled).

    Returns:
        The parsed object (empty dict when unset).

    Raises:
        ValueError: When the value is not a JSON object — configuration errors
            must fail at startup, not silently drop the operator's intent.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"OPENAI_EXTRA_BODY_JSON is not valid JSON ({exc}). Expected a JSON "
            "object, e.g. '{\"enable_thinking\": false}'."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            "OPENAI_EXTRA_BODY_JSON must be a JSON object, e.g. "
            "'{\"enable_thinking\": false}'; got "
            f"{type(parsed).__name__}."
        )
    logger.warning(
        "OPENAI_EXTRA_BODY_JSON override is active; every request will carry "
        "extra_body=%s (keys win over the profile fragment).",
        parsed,
    )
    return parsed


def create_llm_client(settings: Settings) -> MultiModelLLMClient:
    """Factory function to create a multi-model LLM client.

    Builds one ``OpenAIClient`` per model name in ``settings.llm.models``.
    All clients share the same api_key/base_url (OpenAI-compatible) unless
    Azure is enabled (in which case a single Azure deployment is used and
    fallback across multiple OpenAI models is disabled — Azure users
    should configure the deployment name list explicitly if needed).

    Each model resolves its own profile independently (plan v10 §3.2), so a
    mixed fallback list adapts each family correctly. The configured output
    budget is pre-checked against each profile's real cap (P0-4) — per-request
    clamping in ``compose_request`` remains the hard guard.

    Args:
        settings: Application settings.

    Returns:
        Configured multi-model LLM client.
    """
    timeout = float(settings.llm.timeout)
    max_tokens = settings.llm.max_output_tokens
    reasoning_effort = settings.llm.reasoning_effort or None
    continuation_intent = settings.llm.continuation_reasoning_effort or None
    # Plan-c B1.1: an explicit OPENAI_BLOCKING_HARD_TIMEOUT is honoured
    # verbatim (validated >= OPENAI_TIMEOUT inside OpenAIClient); unset means
    # max(timeout*2, 600) so slow completions are not killed early.
    blocking_hard_timeout: float | None = (
        float(settings.llm.blocking_hard_timeout) if settings.llm.blocking_hard_timeout else None
    )
    # Same knob for the streaming channel (2026-09-13): a stream that keeps
    # delivering reasoning chunks never trips the transport read timeout, so
    # without an explicit cap a single call was unbounded.
    stream_hard_timeout: float | None = (
        float(settings.llm.stream_hard_timeout) if settings.llm.stream_hard_timeout else None
    )
    extra_body = _parse_extra_body(settings.llm.extra_body_json)

    def _build(model_name: str, sdk_client: OpenAI | AzureOpenAI) -> OpenAIClient:
        profile = _resolve_profile_for(model_name, settings)
        if profile.max_output_cap is not None and max_tokens > profile.max_output_cap:
            logger.warning(
                "OPENAI_MAX_OUTPUT_TOKENS=%d exceeds the real output cap of "
                "model '%s' (profile '%s': %d); requests will be clamped.",
                max_tokens,
                model_name,
                profile.name,
                profile.max_output_cap,
            )
        return OpenAIClient(
            client=sdk_client,
            model=model_name,
            timeout=timeout,
            max_output_tokens=max_tokens,
            profile=profile,
            reasoning_effort=reasoning_effort,
            continuation_intent=continuation_intent,
            blocking_hard_timeout=blocking_hard_timeout,
            stream_hard_timeout=stream_hard_timeout,
            extra_body=extra_body,
        )

    if settings.azure_llm.enabled:
        from openai import AzureOpenAI

        azure_client: OpenAI | AzureOpenAI = AzureOpenAI(
            api_key=settings.azure_llm.api_key,
            azure_endpoint=settings.azure_llm.endpoint,
            api_version=settings.azure_llm.api_version,
        )
        # Azure uses deployment names, not model names. If the user lists
        # multiple models, treat each as a separate deployment name.
        deployments = settings.llm.models
        if len(deployments) == 1 and deployments[0] == "gpt-4o-mini":
            # Default: use the configured Azure deployment
            deployments = [settings.azure_llm.deployment]
        clients = [_build(dep, azure_client) for dep in deployments]
        logger.info("Using Azure OpenAI deployments: %s", deployments)
    else:
        client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        clients = [_build(model_name, client) for model_name in settings.llm.models]
        logger.info("Using OpenAI models (in fallback order): %s", settings.llm.models)

    logger.info("Max output tokens: %d", max_tokens)
    mm_client = MultiModelLLMClient(clients=clients)
    # Stream tokens for real-time progress logs when the provider supports it;
    # clients transparently fall back to blocking if streaming is unavailable.
    mm_client.set_stream_enabled(bool(getattr(settings.llm, "stream", True)))
    return mm_client
