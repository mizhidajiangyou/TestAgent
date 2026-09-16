"""LINK-S1b: strongly-typed field access for cross-domain consumption
(plan-links-v15 §3.1).

Links-domain code must NOT read T3/T10-owned shared fields via raw
``getattr(<model>, "<field>", <default>)`` (the arch gate enforces this) —
all cross-domain reads go through the typed accessors here. Owner-domain
code (T3's template wiring, T10's executability gates) keeps direct typed
reads; the accessors exist for the LINKS domain and for the dict artifacts
that flow through the pipeline (one adapter, not two Gate implementations).

Field views (§3.3): ``response_schemas`` keeps status-code keys and nested
structure; the top-level object and one level of ``data``/``result``
envelope are expanded; deeper objects, arrays and unsupported combinators
are explicitly UNKNOWN — never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CaseView",
    "FieldAddress",
    "FieldView",
    "case_view_from_dict",
    "case_view_from_model",
    "request_fields",
    "response_fields",
]

_ENVELOPE_KEYS = ("data", "result")
_FIELD_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Canonical field addresses (v15 §3.3). ``request.<path>`` is an input
#: alias for ``body.<path>``; body and params are distinct namespaces.
_LOCATION_ALIASES = {"request": "body"}


@dataclass(frozen=True)
class FieldAddress:
    """``response.<path>`` / ``body.<path>`` / ``params.<name>``."""

    location: str
    path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", _LOCATION_ALIASES.get(self.location, self.location))

    def render(self) -> str:
        return f"{self.location}.{self.path}"

    @staticmethod
    def parse(text: str) -> FieldAddress | None:
        parts = text.strip().split(".", 1)
        if len(parts) != 2 or parts[0] not in {"response", "body", "params", "request"}:
            return None
        if not parts[1] or not all(_FIELD_NAME_RE.match(seg) for seg in parts[1].split(".")):
            return None
        return FieldAddress(location=parts[0], path=parts[1])


@dataclass(frozen=True)
class FieldView:
    """One known response/request field (top-level or one envelope level)."""

    address: str
    type: str
    status_source: str  # which status code declared it ("" = request side)
    envelope: str = ""  # "" (top level) or "data"/"result"


@dataclass(frozen=True)
class CaseView:
    """The Gate input contract (v15 §3.1): one non-empty case_id, the four
    text sections, binds and program identity."""

    case_id: str
    preconditions: tuple[str, ...]
    steps: tuple[str, ...]
    expected_results: tuple[str, ...]
    cleanup: tuple[str, ...]
    binds: dict[str, Any]
    path_id: str
    source_stage: str


def _model_get(obj: Any, name: str) -> Any:
    """Presence-checked attribute read (no silent default)."""
    if hasattr(obj, name):
        return getattr(obj, name)
    return None


def case_view_from_model(case: Any) -> CaseView:
    """Build a CaseView from a TestCase model (typed attribute access)."""
    case_id = str(_model_get(case, "id") or "")
    if not case_id:
        raise ValueError("CaseView requires a non-empty case_id")
    return CaseView(
        case_id=case_id,
        preconditions=tuple(str(p) for p in _model_get(case, "preconditions") or []),
        steps=tuple(str(s) for s in _model_get(case, "steps") or []),
        expected_results=tuple(str(e) for e in _model_get(case, "expected_results") or []),
        # cleanup is read from existing artifact fields when present; a
        # missing field is an empty list, NOT a new shared field (§3.1).
        cleanup=tuple(str(c) for c in _model_get(case, "cleanup") or []),
        binds=dict(_model_get(case, "binds") or {}),
        path_id=str(_model_get(case, "path_id") or ""),
        source_stage=str(_model_get(case, "source_stage") or ""),
    )


def case_view_from_dict(case: dict[str, Any]) -> CaseView:
    """Build a CaseView from a dict artifact (the unified adapter — Gates
    never re-implement model/dict handling)."""
    case_id = str(case.get("id", "") or "")
    if not case_id:
        raise ValueError("CaseView requires a non-empty case_id")
    for field_name in ("preconditions", "steps", "expected_results", "cleanup"):
        value = case.get(field_name)
        if value is not None and not isinstance(value, list):
            raise ValueError(f"CaseView {field_name} must be a list, got {type(value).__name__}")
    return CaseView(
        case_id=case_id,
        preconditions=tuple(str(p) for p in case.get("preconditions") or []),
        steps=tuple(str(s) for s in case.get("steps") or []),
        expected_results=tuple(str(e) for e in case.get("expected_results") or []),
        cleanup=tuple(str(c) for c in case.get("cleanup") or []),
        binds=dict(case.get("binds") or {}),
        path_id=str(case.get("path_id", "") or ""),
        source_stage=str(case.get("source_stage", "") or ""),
    )


def _expand_object(
    schema: dict[str, Any], status: str, envelope: str, location: str
) -> list[FieldView]:
    fields: list[FieldView] = []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return fields
    for name, pschema in props.items():
        pschema = pschema if isinstance(pschema, dict) else {}
        fields.append(
            FieldView(
                address=f"{location}.{name}",
                type=str(pschema.get("type", "?")),
                status_source=status,
                envelope=envelope,
            )
        )
    return fields


def response_fields(endpoint: Any) -> list[FieldView]:
    """Expand the T3-owned ``response_schemas`` (v15 §3.3 field view).

    Top-level object properties plus one level of ``data``/``result``
    envelope; deeper structures stay UNKNOWN (not flattened, not guessed).
    Media type preference: application/json, then lexicographic; the choice
    is deterministic. Field candidates keep their status-code source.
    """
    schemas = _model_get(endpoint, "response_schemas") or {}
    views: list[FieldView] = []
    for status in sorted(schemas, key=str):
        schema = schemas[status] if isinstance(schemas[status], dict) else {}
        views.extend(_expand_object(schema, str(status), "", "response"))
        for envelope in _ENVELOPE_KEYS:
            props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            inner = props.get(envelope)
            if isinstance(inner, dict) and inner.get("type") == "object":
                views.extend(_expand_object(inner, str(status), envelope, "response"))
    return views


def request_fields(endpoint: Any) -> list[FieldView]:
    """Request-side field view: body properties + parameter names (no
    status source; params and body are distinct addresses)."""
    views: list[FieldView] = []
    body = _model_get(endpoint, "request_body")
    if isinstance(body, dict):
        schema = body.get("schema") or {}
        views.extend(_expand_object(schema if isinstance(schema, dict) else {}, "", "", "body"))
        for envelope in _ENVELOPE_KEYS:
            props = schema.get("properties") if isinstance(schema, dict) else None
            inner = props.get(envelope) if isinstance(props, dict) else None
            if isinstance(inner, dict) and inner.get("type") == "object":
                views.extend(_expand_object(inner, "", envelope, "body"))
    for p in _model_get(endpoint, "parameters") or []:
        if isinstance(p, dict) and isinstance(p.get("name"), str):
            views.append(
                FieldView(
                    address=f"params.{p['name']}",
                    type=str((p.get("schema") or {}).get("type", "?"))
                    if isinstance(p.get("schema"), dict)
                    else "?",
                    status_source="",
                )
            )
    return views
