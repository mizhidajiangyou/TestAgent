"""Executability gates (T10, fix-plan §3.4).

Pure, zero-LLM post-generation gates over a case's raw dict shape:

- **Gate-A placeholder closure**: every placeholder must be produced before
  it is consumed inside the step sequence (preconditions count as available;
  expected_results / cleanup cannot produce). Failures: ``orphan`` (never
  produced), ``forward_reference`` (consumed before produced).
- **Gate-B binding contract**: ``binds`` producers/consumers must reference
  spec endpoints verbatim and (when the spec is known) existing fields.
  Spec-unknown schemas are recorded as ``spec_blind`` — blindness is not a
  violation. Failures: ``malformed_bind``, ``producer_endpoint_not_found``,
  ``producer_field_rejected``, ``consumer_endpoint_not_found``,
  ``consumer_field_rejected``.
- **Grade**: ``INTEGRATION`` (closure complete, no violations) / ``DRAFT``
  (orphans, forward references, endpoint or malformed violations) /
  ``REJECTED`` (field-level rejections: inventing fields the spec does not
  have). Grades land on the artifact and in the report — nothing is
  silently deleted.
- **Metrics**: empty denominators render ``n/a`` (never 1.0-as-pass).

Field sets come from the parsed spec: request fields = body properties +
parameter names; response fields = ``APIEndpoint.response_schemas`` (T3).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GATE_EXEMPT_PLACEHOLDERS",
    "GateResult",
    "gate_a_closure",
    "gate_b_bindings",
    "grade_case",
    "normalize_placeholder",
    "placeholder_closure_metrics",
]

#: Canonical placeholder syntax (fix-plan §3.4).
_PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]{1,30})>")
#: Legacy/stray styles normalized into the canonical one.
_PLACEHOLDER_ALIASES = (
    (re.compile(r"\{\{\s*uuid\s*\}\}"), "<RUN_ID>"),
    (re.compile(r"<uuid>", re.IGNORECASE), "<RUN_ID>"),
    (re.compile(r"<uniq>", re.IGNORECASE), "<RUN_ID>"),
    (re.compile(r"<token>", re.IGNORECASE), "<TOKEN>"),
)

#: Provided by the environment — exempt from closure and from Gate-B.
GATE_EXEMPT_PLACEHOLDERS = frozenset({"TOKEN", "BEARER_TOKEN", "API_KEY", "BASE_URL", "SESSION_ID"})

#: Steps containing one of these verbs may produce values (creation verbs).
_PRODUCER_VERBS = ("POST", "PUT", "PATCH", "CREATE", "REGISTER", "SIGNUP")
#: Assignment / store patterns that produce a placeholder value.
_STORE_RE = re.compile(
    r"<([A-Z][A-Z0-9_]{1,30})>\s*=?\s*(?:=|:=)\s*\S+|(?:store|save|保存|存为|store as|save as)\s+(?:as\s+)?<([A-Z][A-Z0-9_]{1,30})>",
    re.IGNORECASE,
)
#: Cleanup context: cannot produce.
_CLEANUP_RE = re.compile(r"\b(cleanup|clean up|删除|清理|reset)\b", re.IGNORECASE)

_BIND_ENDPOINT_RE = re.compile(
    r"^(?P<method>[A-Za-z]+)\s+(?P<path>/\S+)\s+(?P<loc>response|request|body|params|path|query)\.(?P<field>[\w.\[\]]+)$"
)

_PRODUCER_LOCATIONS = ("response",)
_CONSUMER_LOCATIONS = ("body", "request", "params", "path", "query")


def normalize_placeholder(text: str) -> str:
    """Normalize stray placeholder styles into the canonical ``<UPPER>`` form."""
    for pattern, replacement in _PLACEHOLDER_ALIASES:
        text = pattern.sub(replacement, text)
    return text


def _placeholders_in(text: str) -> list[str]:
    return _PLACEHOLDER_RE.findall(normalize_placeholder(text))


@dataclass(frozen=True)
class GateResult:
    """Outcome of both gates for one case (writes to the artifact)."""

    grade: str
    orphans: list[str] = field(default_factory=list)
    forward_references: list[str] = field(default_factory=list)
    bind_violations: list[dict[str, str]] = field(default_factory=list)
    spec_blind: list[str] = field(default_factory=list)
    closure_rate: str = "n/a"
    setup_dependency_completeness: str = "n/a"

    def to_meta(self) -> dict[str, Any]:
        return {
            "grade": self.grade,
            "orphans": self.orphans,
            "forward_references": self.forward_references,
            "bind_violations": self.bind_violations,
            "spec_blind": self.spec_blind,
            "closure_rate": self.closure_rate,
            "setup_dependency_completeness": self.setup_dependency_completeness,
        }


def gate_a_closure(case: dict[str, Any]) -> tuple[list[str], list[str], str]:
    """Gate-A: placeholder closure over the step sequence.

    Returns ``(orphans, forward_references, closure_rate)``. ``orphan`` =
    consumed but never produced anywhere; ``forward_reference`` = consumed
    before its production step. The rate is
    ``1 - (orphan + forward) / total_refs``; no references -> ``1.0``.
    """
    preconditions = " ".join(str(p) for p in case.get("preconditions", []) or [])
    steps = [str(s) for s in case.get("steps", []) or []]
    expected = " ".join(str(e) for e in case.get("expected_results", []) or [])
    description = str(case.get("description", "") or "")

    available: set[str] = set(GATE_EXEMPT_PLACEHOLDERS)
    available.update(_placeholders_in(preconditions))
    # binds producers declare production events: the value is produced by the
    # step that invokes the producer endpoint (Gate-B validates the binding
    # itself; closure only needs the production order).
    bind_producers: dict[str, str] = {}
    binds = case.get("binds") or {}
    if isinstance(binds, dict):
        for key, spec in binds.items():
            if isinstance(spec, dict):
                m = re.match(r"^([A-Za-z]+)\s+(/\S+)", str(spec.get("producer", "")))
                if m:
                    bind_producers[str(key).strip("<>")] = f"{m.group(1).upper()} {m.group(2)}"

    # Pass 1: per-step production events.
    steps_data: list[tuple[list[str], set[str]]] = []
    for step in steps:
        step_upper = step.upper()
        is_cleanup = bool(_CLEANUP_RE.search(step))
        has_producer_verb = any(verb in step_upper for verb in _PRODUCER_VERBS)
        has_store = bool(_STORE_RE.search(step))
        produced: set[str] = set()
        for m in _STORE_RE.finditer(normalize_placeholder(step)):
            name = m.group(1) or m.group(2)
            if name:
                produced.add(name)
        if not is_cleanup:
            for name, endpoint in bind_producers.items():
                if endpoint in step and has_producer_verb:
                    produced.add(name)
            # Production events are EXPLICIT ONLY (defect ④, 2026-09-19
            # review): store/capture verbs, `<ID> = value` assignment or a
            # binds producer endpoint. A bare creation verb must NOT mark
            # every placeholder in the step as produced — a step like
            # "POST /orders body {buyer: <USER_ID>}" CONSUMES <USER_ID>,
            # and marking it produced made forward consumption close
            # spuriously (orphan lost, closure faked at 1.0).
        steps_data.append((_placeholders_in(step), produced))

    production_step: dict[str, int] = {}
    for idx, (_, produced) in enumerate(steps_data):
        for name in produced:
            production_step.setdefault(name, idx)

    # Pass 2: classify every reference.
    orphans: list[str] = []
    forwards: list[str] = []
    total_refs = 0
    for idx, (consumed, _) in enumerate(steps_data):
        for name in consumed:
            total_refs += 1
            if name in available or name in GATE_EXEMPT_PLACEHOLDERS:
                continue
            if name not in production_step:
                orphans.append(name)
            elif production_step[name] > idx:
                forwards.append(name)
    for section in (expected, description):
        for name in _placeholders_in(section):
            total_refs += 1
            if name in available or name in GATE_EXEMPT_PLACEHOLDERS:
                continue
            if name not in production_step:
                orphans.append(name)

    deduped_orphans = sorted(set(orphans))
    deduped_forwards = sorted(set(forwards))
    violation_count = len(deduped_orphans) + len(deduped_forwards)
    rate = "1.0" if total_refs == 0 else str(round(1 - violation_count / total_refs, 4))
    return deduped_orphans, deduped_forwards, rate


def _parse_bind_side(raw: Any) -> tuple[str, str, str, str]:
    """Parse ``METHOD /path location.field`` -> (method, path, loc, field).

    Empty strings mark a malformed side.
    """
    if not isinstance(raw, str):
        return "", "", "", ""
    m = _BIND_ENDPOINT_RE.match(raw.strip())
    if not m:
        return "", "", "", ""
    return (
        m.group("method").upper(),
        m.group("path"),
        m.group("loc"),
        m.group("field"),
    )


def _endpoint_index(endpoints: Sequence[Any]) -> dict[str, Any]:
    return {f"{ep.method.upper()} {ep.path}": ep for ep in endpoints}


def _request_fields(ep: Any) -> set[str]:
    fields: set[str] = set()
    fields.update(re.findall(r"\{(\w+)\}", str(getattr(ep, "path", ""))))
    for p in getattr(ep, "parameters", None) or []:
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            fields.add(p["name"])
            fields.add(p["name"].split(".")[-1])
    body = getattr(ep, "request_body", None)
    if isinstance(body, dict):
        props = (body.get("schema") or {}).get("properties", {})
        if isinstance(props, dict):
            for name in props:
                fields.add(str(name))
    return fields


def _response_fields(ep: Any) -> tuple[set[str], bool]:
    """(field names, schema_known) from T3's response_schemas."""
    schemas = getattr(ep, "response_schemas", None) or {}
    if not schemas:
        return set(), False
    fields: set[str] = set()
    for schema in schemas.values():
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if isinstance(props, dict):
            fields.update(str(k) for k in props)
    return fields, True


def _field_matches(field_path: str, known: set[str]) -> bool:
    """Dotted field paths compare on their last segment (fix-plan rule)."""
    last = field_path.split(".")[-1].strip("[]")
    return last in known or field_path in known


def gate_b_bindings(
    case: dict[str, Any], endpoints: Sequence[Any]
) -> tuple[list[dict[str, str]], list[str]]:
    """Gate-B: binding contract. Returns ``(violations, spec_blind)``."""
    binds = case.get("binds") or {}
    violations: list[dict[str, str]] = []
    blind: list[str] = []
    if not isinstance(binds, dict):
        if binds:
            violations.append(
                {"class": "malformed_bind", "key": "", "detail": "binds is not an object"}
            )
        return violations, blind
    index = _endpoint_index(endpoints)

    for key, spec in binds.items():
        name = str(key).strip("<>")
        if not isinstance(spec, dict) or "producer" not in spec:
            violations.append(
                {"class": "malformed_bind", "key": name, "detail": "missing producer"}
            )
            continue
        p_method, p_path, p_loc, p_field = _parse_bind_side(spec.get("producer"))
        if not p_method:
            violations.append(
                {
                    "class": "malformed_bind",
                    "key": name,
                    "detail": f"unparseable producer: {spec.get('producer')!r}",
                }
            )
            continue
        producer = index.get(f"{p_method} {p_path}")
        if producer is None:
            violations.append(
                {
                    "class": "producer_endpoint_not_found",
                    "key": name,
                    "detail": f"{p_method} {p_path} not in spec",
                }
            )
        else:
            fields, known = _response_fields(producer)
            if p_loc not in _PRODUCER_LOCATIONS:
                violations.append(
                    {
                        "class": "malformed_bind",
                        "key": name,
                        "detail": f"producer location {p_loc!r}",
                    }
                )
            elif not known:
                blind.append(f"{name}:producer")
            elif not _field_matches(p_field, fields):
                violations.append(
                    {
                        "class": "producer_field_rejected",
                        "key": name,
                        "detail": f"field {p_field!r} not in {p_method} {p_path} response schema",
                    }
                )
        consumer = spec.get("consumer")
        if consumer:
            c_method, c_path, c_loc, c_field = _parse_bind_side(consumer)
            if not c_method:
                violations.append(
                    {
                        "class": "malformed_bind",
                        "key": name,
                        "detail": f"unparseable consumer: {consumer!r}",
                    }
                )
                continue
            cons_ep = index.get(f"{c_method} {c_path}")
            if cons_ep is None:
                violations.append(
                    {
                        "class": "consumer_endpoint_not_found",
                        "key": name,
                        "detail": f"{c_method} {c_path} not in spec",
                    }
                )
            else:
                if c_loc not in _CONSUMER_LOCATIONS:
                    violations.append(
                        {
                            "class": "malformed_bind",
                            "key": name,
                            "detail": f"consumer location {c_loc!r}",
                        }
                    )
                    continue
                known_fields = _request_fields(cons_ep)
                if not known_fields:
                    blind.append(f"{name}:consumer")
                elif not _field_matches(c_field, known_fields):
                    violations.append(
                        {
                            "class": "consumer_field_rejected",
                            "key": name,
                            "detail": f"field {c_field!r} not in {c_method} {c_path} request schema",
                        }
                    )
    return violations, blind


def grade_case(case: dict[str, Any], endpoints: Sequence[Any]) -> GateResult:
    """Run Gate-A + Gate-B and grade the case (fix-plan §3.4)."""
    orphans, forwards, rate = gate_a_closure(case)
    violations, blind = gate_b_bindings(case, endpoints)
    hard = [
        v
        for v in violations
        if v["class"]
        in {
            "producer_field_rejected",
            "consumer_field_rejected",
        }
    ]
    if hard:
        grade = "REJECTED"
    elif orphans or forwards or violations:
        grade = "DRAFT"
    else:
        grade = "INTEGRATION"
    return GateResult(
        grade=grade,
        orphans=orphans,
        forward_references=forwards,
        bind_violations=violations,
        spec_blind=blind,
        closure_rate=rate,
    )


def placeholder_closure_metrics(
    cases: Sequence[dict[str, Any]], endpoints: Sequence[Any]
) -> dict[str, Any]:
    """Run-level metrics; empty denominators render ``n/a`` (never 1.0)."""
    total_orphans = 0
    violations: dict[str, int] = {}
    dependency_cases = 0
    bind_declared_cases = 0
    for case in cases:
        result = grade_case(case, endpoints)
        total_orphans += len(result.orphans)
        for v in result.bind_violations:
            violations[v["class"]] = violations.get(v["class"], 0) + 1
        has_dependency = any(
            "<" in str(step) and ">" in str(step) for step in case.get("steps", []) or []
        )
        if has_dependency:
            dependency_cases += 1
            if case.get("binds"):
                bind_declared_cases += 1
    return {
        "closure_rate": "n/a" if not cases else _aggregate_closure(cases, endpoints),
        "orphan_placeholders": total_orphans,
        "bind_violations": violations,
        "setup_dependency_completeness": (
            "n/a" if dependency_cases == 0 else f"{bind_declared_cases / dependency_cases:.4f}"
        ),
    }


def _aggregate_closure(cases: Sequence[dict[str, Any]], endpoints: Sequence[Any]) -> str:
    refs = 0
    bad = 0
    for case in cases:
        orphans, forwards, _ = gate_a_closure(case)
        case_refs = len(_placeholders_in(json_of(case)))
        refs += case_refs
        bad += len(orphans) + len(forwards)
    if refs == 0:
        return "1.0"
    return f"{1 - bad / refs:.4f}"


def json_of(case: dict[str, Any]) -> str:
    import json

    return json.dumps(case, ensure_ascii=False)
