"""Obligation coverage model (T5, fix-plan §3.1).

Coverage is anchored on acceptance OBLIGATIONS — requirement ACs and
in-spec facts that must be either covered by a case or explicitly blocked
with a gap reason. Endpoints are 0..N attributes of an obligation, never
the coverage unit themselves (no AC-x-endpoint cartesian).

Pure model: registration, thread-safe accounting (Phase 2 batches fan
out concurrently), completion semantics and report rendering. Lifting the
"expected cases per endpoint" quota into an obligation-driven floor is
T7's budget model; the engine receives it through the additive
``GenericHooks.expected_for`` hook so the domain-free engine never sees
obligations.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "BindingBasis",
    "Obligation",
    "ObligationRegistry",
    "ObligationSource",
    "ObligationState",
    "ObligationStatus",
    "register_spec_obligations",
]

_FORMAT_FACTS = ("email", "date", "date-time", "uri", "uuid", "ipv4")


class ObligationSource(StrEnum):
    REQUIREMENT = "requirement"
    SPEC = "spec"


class ObligationStatus(StrEnum):
    UNCOVERED = "uncovered"
    COVERED = "covered"
    GAP_BLOCKED = "gap_blocked"


class BindingBasis(StrEnum):
    EXPLICIT = "explicit"
    KEYWORD = "keyword"
    LLM_SUGGESTION = "llm_suggestion"
    NONE = "none"


@dataclass(frozen=True)
class Obligation:
    """One acceptance obligation (fix-plan §3.1)."""

    id: str
    requirement_id: str | None
    statement: str
    source: ObligationSource
    endpoint_bindings: tuple[str, ...] = ()
    binding_basis: BindingBasis = BindingBasis.NONE


@dataclass
class ObligationState:
    """Mutable runtime state of one obligation (held by the registry)."""

    obligation: Obligation
    status: ObligationStatus = ObligationStatus.UNCOVERED
    covered_by: list[str] = field(default_factory=list)
    gap_reason: str | None = None


class ObligationRegistry:
    """Thread-safe obligation ledger: register -> cover / gap_blocked.

    Completion semantics (fix-plan §3.1 rule 2): coverage is complete when
    every obligation is ``covered`` or ``gap_blocked`` — a ``gap_blocked``
    entry MUST carry a gap reason and appear in the report. Silent zero
    coverage is impossible: uncovered obligations keep the run incomplete.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, ObligationState] = {}

    def register(self, obligation: Obligation) -> None:
        """Register one obligation; duplicate ids are rejected (uniqueness
        contract — a second declaration would silently fork accounting)."""
        with self._lock:
            if obligation.id in self._states:
                raise ValueError(f"obligation id already registered: {obligation.id!r}")
            self._states[obligation.id] = ObligationState(obligation=obligation)

    def register_many(self, obligations: Sequence[Obligation]) -> None:
        for obligation in obligations:
            self.register(obligation)

    def register_requirement_acs(
        self,
        requirement_id: str,
        statements: Sequence[str],
        endpoint_bindings: tuple[str, ...] = (),
        binding_basis: BindingBasis = BindingBasis.KEYWORD,
    ) -> list[str]:
        """Register one obligation per acceptance criterion statement."""
        ids: list[str] = []
        for i, statement in enumerate(statements, 1):
            oid = f"{requirement_id}-AC{i}"
            self.register(
                Obligation(
                    id=oid,
                    requirement_id=requirement_id,
                    statement=statement,
                    source=ObligationSource.REQUIREMENT,
                    endpoint_bindings=endpoint_bindings,
                    binding_basis=binding_basis,
                )
            )
            ids.append(oid)
        return ids

    def cover(self, obligation_id: str, case_id: str) -> None:
        with self._lock:
            state = self._states.get(obligation_id)
            if state is None:
                raise KeyError(f"unknown obligation: {obligation_id!r}")
            if case_id not in state.covered_by:
                state.covered_by.append(case_id)
            state.status = ObligationStatus.COVERED
            state.gap_reason = None

    def cover_many(self, case_id: str, obligation_ids: Sequence[str]) -> None:
        """Account one case against every obligation it declares."""
        for oid in obligation_ids:
            self.cover(oid, case_id)

    def gap_blocked(self, obligation_id: str, reason: str) -> None:
        if not reason:
            raise ValueError("gap_blocked requires a gap reason (no silent gaps)")
        with self._lock:
            state = self._states.get(obligation_id)
            if state is None:
                raise KeyError(f"unknown obligation: {obligation_id!r}")
            state.status = ObligationStatus.GAP_BLOCKED
            state.gap_reason = reason

    def states(self) -> list[ObligationState]:
        with self._lock:
            return list(self._states.values())

    def uncovered(self) -> list[ObligationState]:
        return [s for s in self.states() if s.status is ObligationStatus.UNCOVERED]

    def is_complete(self) -> bool:
        """All obligations ∈ {covered, gap_blocked}."""
        return not self.uncovered()

    def summary(self) -> dict[str, int]:
        states = self.states()
        return {
            "total": len(states),
            "covered": sum(1 for s in states if s.status is ObligationStatus.COVERED),
            "gap_blocked": sum(1 for s in states if s.status is ObligationStatus.GAP_BLOCKED),
            "uncovered": sum(1 for s in states if s.status is ObligationStatus.UNCOVERED),
        }

    def render_report(self) -> str:
        states = self.states()
        counts = self.summary()
        lines = [
            "# Obligation coverage report",
            "",
            f"Total: {counts['total']} | covered: {counts['covered']} | "
            f"gap_blocked: {counts['gap_blocked']} | uncovered: {counts['uncovered']}",
            "",
            "| obligation | source | status | covered_by | gap_reason |",
            "|---|---|---|---|---|",
        ]
        for s in states:
            ob = s.obligation
            lines.append(
                f"| {ob.id} | {ob.source.value} | {s.status.value} "
                f"| {', '.join(s.covered_by) or '-'} | {s.gap_reason or '-'} |"
            )
        lines.append("")
        return "\n".join(lines)


def _spec_fact(endpoint_id: str, fact_id: str, statement: str) -> Obligation:
    return Obligation(
        id=f"SPEC-{endpoint_id}-{fact_id}",
        requirement_id=None,
        statement=statement,
        source=ObligationSource.SPEC,
        endpoint_bindings=(endpoint_id,),
        binding_basis=BindingBasis.EXPLICIT,
    )


def register_spec_obligations(endpoints: Sequence[Any]) -> list[Obligation]:
    """Derive in-spec obligations from parsed endpoints (fix-plan §3.1 rule 1).

    Facts registered: required body properties, numeric bounds
    (minimum/maximum), format-validated fields and declared error responses
    (4xx/5xx). These are facts the SPEC documents — a faithful generation
    should exercise them or the report shows why not.
    """
    obligations: list[Obligation] = []
    for ep in endpoints:
        endpoint_id = (
            f"{ep.method.upper()}_{ep.path.strip('/').replace('/', '_').replace('{}', 'id')}"
        )
        body = getattr(ep, "request_body", None)
        schema = body.get("schema", {}) if isinstance(body, dict) else {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", []) or []) if isinstance(schema, dict) else set()
        if isinstance(props, dict):
            for name, pschema in props.items():
                pschema = pschema if isinstance(pschema, dict) else {}
                if name in required:
                    obligations.append(
                        _spec_fact(
                            endpoint_id, f"required_{name}", f"{name} is a required body field"
                        )
                    )
                if pschema.get("format") in _FORMAT_FACTS:
                    obligations.append(
                        _spec_fact(
                            endpoint_id,
                            f"format_{name}",
                            f"{name} must satisfy format={pschema['format']}",
                        )
                    )
                for bound in ("minimum", "maximum"):
                    if pschema.get(bound) is not None:
                        obligations.append(
                            _spec_fact(
                                endpoint_id,
                                f"{bound}_{name}",
                                f"{name} declares {bound}={pschema[bound]}",
                            )
                        )
        for code in getattr(ep, "responses", None) or []:
            if str(code)[:1] in {"4", "5"}:
                obligations.append(
                    _spec_fact(endpoint_id, f"error_{code}", f"declared error response {code}")
                )
    return obligations
