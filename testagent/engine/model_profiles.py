"""Declarative model profiles: per-model request dialects as data (plan v10).

Reasoning-style models share one output budget between reasoning tokens and
the visible answer. When the reasoning phase consumes the whole budget the
visible content comes back EMPTY with ``finish_reason="length"`` — retrying
the *identical* request is guaranteed to fail the same way. This module turns
every model-family difference (budget parameter name, temperature semantics,
effort dialect, empty-response classification, real output cap) into
declarative :class:`ModelProfile` data so the client/engine contain no
per-family branching.

Layers (plan v10 §2):
- :func:`resolve_profile` — explicit ``OPENAI_MODEL_PROFILE`` > matcher > generic
- :func:`compose_request` — intent (budget / effort / determinism) → dialect kwargs
- :func:`classify_response` — empty-response semantics → :class:`Outcome`
- :func:`recovery_plan` — DOWNGRADE (once) → SPLIT → RAISE_BUDGET → FAIL

Verification status (plan v10 §3.2): only ``deepseek-v4`` is VERIFIED against
a real API. The remaining profiles carry documentation-inferred values and
MUST log a WARNING when selected; field values are back-filled after the
per-profile diagnosis runs (plan v10 §9) before relying on them.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "AVAILABLE_PROFILE_NAMES",
    "ModelProfile",
    "NextAction",
    "Outcome",
    "RecoveryState",
    "RequestIntent",
    "classify_response",
    "compose_request",
    "recovery_plan",
    "resolve_profile",
]


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ModelProfile:
    """Declarative dialect of one model family (plan v10 §3.1)."""

    name: str
    #: Output budget parameter name ("max_tokens" | "max_completion_tokens").
    budget_param: str
    #: Whether the budget is shared between reasoning tokens and the answer.
    budget_shared: bool
    #: "supported" | "dropped" | "thinking_dependent" (temperature only works
    #: with thinking off for the latter — DeepSeek/Qwen behaviour).
    temperature_mode: str
    #: Intent tier → raw request fragment (the per-family effort dialect).
    effort_translation: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Default continuation intent for the one-shot downgrade (None = the
    #: profile has no effort control and skips downgrading entirely).
    continuation_intent: str | None = None
    can_disable_thinking: bool = False
    #: Thinking mode only works with streaming (Qwen hard constraint); the
    #: blocking channel strips thinking fragments automatically.
    requires_stream_for_thinking: bool = False
    #: Where reasoning consumption is observable ("usage_details" |
    #: "reasoning_content" | "none").
    reasoning_signal: str = "none"
    #: finish_reason values that indicate "budget exhausted" on an empty body.
    empty_finish_reasons: tuple[str, ...] = ("length",)
    #: Real output cap of the model (None = unknown, no clamping).
    max_output_cap: int | None = None
    #: Azure capability guard (None = no guard; UNVERIFIED placeholder).
    min_api_version: str | None = None
    #: "VERIFIED" | "UNVERIFIED" | "N/A" (plan v10 §10 verification scope).
    verification: str = "UNVERIFIED"
    #: Model-name matchers (fnmatch, lowercased); empty for the generic fallback.
    matchers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestIntent:
    """Caller intent for one request; the profile translates it (§4.1)."""

    budget: int
    #: Intent tier (e.g. "low" / "disabled"); None = model default.
    effort: str | None = None
    deterministic: bool = True


class Outcome(Enum):
    """Classification of an LLM response (plan v10 §4.3)."""

    OK = "ok"
    TRUNCATED_PARTIAL = "truncated_partial"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TRANSIENT_EMPTY = "transient_empty"


class NextAction(Enum):
    """Recovery action for a budget-exhausted response (plan v10 §5.2)."""

    DOWNGRADE = "downgrade"
    SPLIT = "split"
    RAISE_BUDGET = "raise_budget"
    FAIL = "fail"


@dataclass(frozen=True)
class RecoveryState:
    """Engine-local recovery state driving :func:`recovery_plan` (§5.2)."""

    downgrade_used: bool
    #: Remaining quota scale — decides whether SPLIT still has room.
    pending_total: int
    scope_size: int
    budget_raised: bool
    #: Whether scope shrinking already hit the floor (no-op to shrink further).
    floor_reached: bool = False


# ----------------------------------------------------------------------
# Built-in profile registry (plan v10 §3.2)
# ----------------------------------------------------------------------

DEEPSEEK_V4 = ModelProfile(
    name="deepseek-v4",
    budget_param="max_tokens",
    budget_shared=True,
    temperature_mode="thinking_dependent",
    effort_translation={
        "low": {"reasoning_effort": "low"},
        "disabled": {"extra_body": {"thinking": {"type": "disabled"}}},
    },
    # One-shot downgrade = disable thinking entirely: zero reasoning overhead
    # AND temperature=0 determinism comes back (plan v10 §5.3).
    continuation_intent="disabled",
    can_disable_thinking=True,
    requires_stream_for_thinking=False,
    reasoning_signal="reasoning_content",
    # P0-3: conservative — "stop" stays on the transient path until diagnosis
    # confirms the gateway ever wraps budget exhaustion as "stop".
    empty_finish_reasons=("length",),
    # Official 384K output cap (checked 2026-08).
    max_output_cap=393216,
    verification="VERIFIED",
    matchers=("deepseek-v4*",),
)

QWEN_3_8 = ModelProfile(
    name="qwen3.8",
    budget_param="max_tokens",
    budget_shared=True,
    temperature_mode="thinking_dependent",
    effort_translation={
        "low": {"extra_body": {"thinking_budget": 4096}},
        "disabled": {"extra_body": {"enable_thinking": False}},
    },
    continuation_intent="low",
    can_disable_thinking=True,
    requires_stream_for_thinking=True,
    reasoning_signal="reasoning_content",
    empty_finish_reasons=("length",),
    max_output_cap=131072,
    verification="UNVERIFIED",
    matchers=("qwen3.8*", "qwen3.7*", "qwen3.6*", "qwen3-max*", "qwen3-plus*"),
)

OPENAI_REASONING = ModelProfile(
    name="openai-reasoning",
    budget_param="max_completion_tokens",
    budget_shared=True,
    temperature_mode="dropped",
    effort_translation={
        "minimal": {"reasoning_effort": "minimal"},
        "low": {"reasoning_effort": "low"},
        "medium": {"reasoning_effort": "medium"},
        "high": {"reasoning_effort": "high"},
    },
    continuation_intent="low",
    can_disable_thinking=False,
    requires_stream_for_thinking=False,
    reasoning_signal="usage_details",
    empty_finish_reasons=("length",),
    max_output_cap=128000,
    min_api_version=None,  # UNVERIFIED placeholder: back-fill after Azure diagnosis
    verification="UNVERIFIED",
    matchers=("gpt-5*", "o1*", "o3*", "o4*"),
)

OPENAI_CLASSIC = ModelProfile(
    name="openai-classic",
    budget_param="max_tokens",
    budget_shared=False,
    temperature_mode="supported",
    effort_translation={},
    continuation_intent=None,
    can_disable_thinking=False,
    requires_stream_for_thinking=False,
    reasoning_signal="none",
    empty_finish_reasons=("length",),
    max_output_cap=16384,
    verification="UNVERIFIED",
    matchers=("gpt-4o*", "gpt-4.1*"),
)

GENERIC_OPENAI_COMPATIBLE = ModelProfile(
    name="generic-openai-compatible",
    budget_param="max_tokens",
    budget_shared=False,
    temperature_mode="dropped",
    effort_translation={},
    continuation_intent=None,
    can_disable_thinking=False,
    requires_stream_for_thinking=False,
    reasoning_signal="none",
    empty_finish_reasons=("length",),
    max_output_cap=None,
    verification="N/A",
    matchers=(),
)

_PROFILES: tuple[ModelProfile, ...] = (
    DEEPSEEK_V4,
    QWEN_3_8,
    OPENAI_REASONING,
    OPENAI_CLASSIC,
    GENERIC_OPENAI_COMPATIBLE,
)

AVAILABLE_PROFILE_NAMES = [p.name for p in _PROFILES]


def _warn_unverified(profile: ModelProfile) -> None:
    if profile.verification == "UNVERIFIED":
        logger.warning(
            "Profile '%s' carries documentation-inferred parameter values that "
            "have NOT been verified against a real API. Functionality works, "
            "but run the per-profile diagnosis and back-fill the fields before "
            "relying on them (plan v10 §9).",
            profile.name,
        )


def resolve_profile(model: str, explicit: str | None = None) -> ModelProfile:
    """Resolve the profile for a model name (plan v10 §3.3).

    Order: explicit ``OPENAI_MODEL_PROFILE`` > matcher > generic fallback.
    An unknown explicit name fails fast listing the available profiles.
    """
    wanted = (explicit or "").strip()
    if wanted:
        for profile in _PROFILES:
            if profile.name == wanted:
                logger.info(
                    "Model '%s' -> profile '%s' (explicit via OPENAI_MODEL_PROFILE)",
                    model,
                    profile.name,
                )
                _warn_unverified(profile)
                return profile
        raise ValueError(
            f"Unknown model profile '{wanted}'. Available profiles: {AVAILABLE_PROFILE_NAMES}"
        )

    lowered = (model or "").lower()
    for profile in _PROFILES:
        if profile.matchers and any(fnmatch(lowered, pat) for pat in profile.matchers):
            logger.info("Model '%s' -> profile '%s' (auto-matched)", model, profile.name)
            _warn_unverified(profile)
            return profile

    logger.warning(
        "Model '%s' is not in the profile registry; thinking/effort parameters "
        "will be dropped. Set OPENAI_MODEL_PROFILE=<profile> to adapt it. "
        "Known profiles: %s",
        model,
        [p.name for p in _PROFILES[:-1]],
    )
    return GENERIC_OPENAI_COMPATIBLE


# ----------------------------------------------------------------------
# compose_request (plan v10 §4.1)
# ----------------------------------------------------------------------


def _strip_thinking_fragment(fragment: dict[str, Any]) -> dict[str, Any]:
    """Remove thinking-control keys (blocking channel on stream-only models)."""
    stripped = {
        k: v for k, v in fragment.items() if k not in ("reasoning_effort", "enable_thinking")
    }
    extra = fragment.get("extra_body")
    stripped.pop("extra_body", None)
    if isinstance(extra, dict):
        kept = {
            k: v
            for k, v in extra.items()
            if k not in ("enable_thinking", "thinking_budget", "thinking")
        }
        if kept:
            stripped["extra_body"] = kept
    return stripped


def _fragment_keeps_thinking_enabled(fragment: dict[str, Any]) -> bool:
    """Whether thinking stays ON after applying the effort fragment."""
    extra = fragment.get("extra_body")
    if isinstance(extra, dict):
        thinking = extra.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            return False
        if extra.get("enable_thinking") is False:
            return False
    # No explicit switch anywhere: thinking-capable profiles default to ON.
    return fragment.get("enable_thinking") is not False


def _temperature_allowed(profile: ModelProfile, thinking_on: bool) -> bool:
    if profile.temperature_mode == "supported":
        return True
    if profile.temperature_mode == "dropped":
        return False
    # "thinking_dependent": temperature only takes effect with thinking off.
    return not thinking_on


def compose_request(
    profile: ModelProfile,
    intent: RequestIntent,
    *,
    channel: str,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate an intent into this profile's request kwargs (plan v10 §4.1).

    P0-1: ``response_format`` is passed through untouched.
    P0-4: the budget is clamped to ``profile.max_output_cap`` with a WARNING.
    """
    kwargs: dict[str, Any] = {}
    budget = intent.budget
    if profile.max_output_cap is not None and budget > profile.max_output_cap:
        logger.warning(
            "budget %d clamped to model cap %d (%s)",
            budget,
            profile.max_output_cap,
            profile.name,
        )
        budget = profile.max_output_cap
    kwargs[profile.budget_param] = budget

    fragment: dict[str, Any] = {}
    if intent.effort and intent.effort in profile.effort_translation:
        fragment = dict(profile.effort_translation[intent.effort])
        if profile.requires_stream_for_thinking and channel == "blocking":
            fragment = _strip_thinking_fragment(fragment)

    thinking_on = _fragment_keeps_thinking_enabled(fragment)
    if intent.deterministic and _temperature_allowed(profile, thinking_on):
        kwargs["temperature"] = 0

    kwargs.update(fragment)
    if response_format is not None:
        kwargs["response_format"] = response_format
    return kwargs


# ----------------------------------------------------------------------
# classify_response (plan v10 §4.3)
# ----------------------------------------------------------------------


def classify_response(
    profile: ModelProfile,
    *,
    text: str,
    finish_reason: str | None,
) -> Outcome:
    """Classify a response, deriving the rules from the profile (v9 §5).

    Non-empty text: ``TRUNCATED_PARTIAL`` when cut at ``length`` else ``OK``.
    Empty text: ``BUDGET_EXHAUSTED`` only when the finish reason is declared
    in ``empty_finish_reasons`` AND the budget is shared (a non-shared model
    cannot have its answer eaten by reasoning — the empty body is transient).
    """
    if text and text.strip():
        if finish_reason == "length":
            return Outcome.TRUNCATED_PARTIAL
        return Outcome.OK
    if profile.budget_shared and finish_reason in profile.empty_finish_reasons:
        return Outcome.BUDGET_EXHAUSTED
    return Outcome.TRANSIENT_EMPTY


# ----------------------------------------------------------------------
# recovery_plan (plan v10 §5.2)
# ----------------------------------------------------------------------


def recovery_plan(profile: ModelProfile | None, state: RecoveryState) -> NextAction:
    """Decide the next recovery action after a budget-exhausted response.

    Invariants (plan v10 §5.1): downgrade at most once; splitting is the main
    recovery; raising the budget is the last resort; everything is bounded by
    the engine's call/time budget. Profiles without effort control (classic /
    generic / None) skip the downgrade and go straight to splitting — the same
    mechanism projected onto the profile's capabilities.
    """
    if not state.downgrade_used and profile is not None and profile.continuation_intent:
        return NextAction.DOWNGRADE
    if not state.floor_reached and (state.pending_total > 1 or state.scope_size > 1):
        return NextAction.SPLIT
    if not state.budget_raised:
        return NextAction.RAISE_BUDGET
    return NextAction.FAIL
