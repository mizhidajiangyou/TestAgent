"""UnitStatus failure taxonomy + v10 Outcome mapping (plan-c B3.1, review P1-6).

CLOSED seven-state discipline: ``UnitStatus`` is an ``enum.Enum`` closed
set. Referencing an undefined member (FAILED, SUCCESS_PARTIAL, ...) fails at
implementation time — Enum access raises ``AttributeError``, a ``Literal``
union fails mypy strict, and a string bypass becomes a never-true comparison
(silent dead branch). "Partial output" and "engine already recovered" are
therefore METADATA on :class:`UnitResult`, never new states; extending the
enum requires revising this module's mapping-table tests first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from testagent.engine.model_profiles import Outcome


class UnitStatus(Enum):
    """Closed taxonomy of unit-level outcomes (plan-c decision 6)."""

    SUCCESS = "success"
    EMPTY = "empty"
    INVALID = "invalid"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    VALIDATION_ERROR = "validation_error"
    CANCELLED = "cancelled"


#: Recovery policy per status (plan-c B3.1). The executor dispatches on this
#: table; engine-recovered-but-still-empty units are NOT re-asked (decision
#: D2 — the v10 ladder already proved the request shape useless).
RECOVERY_POLICY: dict[UnitStatus, str] = {
    UnitStatus.SUCCESS: "none",
    UnitStatus.EMPTY: "re-ask (unless engine_recovered: count as failed, D2)",
    UnitStatus.INVALID: "targeted-repair",
    UnitStatus.TIMEOUT: "retry + model fallback",
    UnitStatus.PROVIDER_ERROR: "retry + model fallback",
    UnitStatus.VALIDATION_ERROR: "targeted-repair",
    UnitStatus.CANCELLED: "never recover",
}


@dataclass
class UnitResult:
    """One generation unit's outcome (status + metadata, plan-c decision 6).

    ``status`` is the ONLY dispatch field. ``partial`` marks salvage output
    from a truncated response; ``engine_recovered`` marks that the v10
    recovery ladder (downgrade/split/raise-budget) already ran for this
    unit — the executor uses it to suppress duplicate re-asks (decision D2).
    """

    status: UnitStatus
    items: list[dict[str, Any]] = field(default_factory=list)
    partial: bool = False
    engine_recovered: bool = False


def unit_status_from_outcome(outcome: Outcome) -> UnitStatus:
    """Map a v10 engine Outcome to a UnitStatus (plan-c B6a mapping table).

    TRUNCATED_PARTIAL maps to SUCCESS because the engine already salvaged
    the partial content into its produced set — the executor only ever sees
    final artifacts, never mid-recovery states.
    """
    from testagent.engine.model_profiles import Outcome as _Outcome

    if outcome is _Outcome.OK:
        return UnitStatus.SUCCESS
    if outcome is _Outcome.TRUNCATED_PARTIAL:
        return UnitStatus.SUCCESS
    if outcome is _Outcome.BUDGET_EXHAUSTED:
        return UnitStatus.EMPTY
    if outcome is _Outcome.TRANSIENT_EMPTY:
        return UnitStatus.PROVIDER_ERROR
    raise ValueError(f"unmapped Outcome: {outcome!r}")


def unit_result_from_engine(
    items: list[dict[str, Any]],
    *,
    outcome: Outcome,
    finish_reason: str | None,
    engine_recovered: bool,
) -> UnitResult:
    """Build a UnitResult from a finished engine run.

    - OK / TRUNCATED_PARTIAL -> SUCCESS (partial=True for the latter).
    - BUDGET_EXHAUSTED with salvage -> SUCCESS + engine_recovered.
    - BUDGET_EXHAUSTED with nothing after the full v10 ladder -> EMPTY +
      engine_recovered (executor does NOT re-ask; decision D2).
    - TRANSIENT_EMPTY -> PROVIDER_ERROR (retry + model fallback).
    """
    status = unit_status_from_outcome(outcome)
    partial = outcome is not None and str(getattr(outcome, "value", "")) == "truncated_partial"
    has_items = bool(items)
    if status is UnitStatus.EMPTY and has_items:
        # The ladder exhausted but salvage produced something usable.
        status = UnitStatus.SUCCESS
    return UnitResult(
        status=status,
        items=items,
        partial=partial,
        engine_recovered=engine_recovered,
    )


def unit_status_from_exception(exc: BaseException) -> UnitStatus:
    """Classify a client-level exception for recovery dispatch.

    Timeout is TRANSIENT (retry + fallback) and NEVER a budget-exhaustion /
    downgrade trigger — a wedged provider did not "think too much", it
    answered nothing at all (plan-c B2.2 v10 boundary).
    """
    from testagent.engine.llm_client import (
        LLMCallTimeoutError,
        LLMOutputTooLongError,
        ReasoningBudgetExhaustedError,
    )

    if isinstance(exc, LLMCallTimeoutError):
        return UnitStatus.TIMEOUT
    if isinstance(exc, ReasoningBudgetExhaustedError):
        # The engine ladder already ran (the client short-circuits on the
        # first occurrence); with no items this is EMPTY + engine_recovered.
        return UnitStatus.EMPTY
    if isinstance(exc, LLMOutputTooLongError):
        return UnitStatus.PROVIDER_ERROR
    return UnitStatus.PROVIDER_ERROR
