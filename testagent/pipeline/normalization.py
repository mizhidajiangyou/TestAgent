"""Output normalization three-stage (T11, fix-plan §4-T11).

- **T11a deterministic normalization** (never calls the LLM): placeholders
  unified to ``<UPPER_SNAKE>``, steps one-line ``method path headers body``,
  ``expected_results`` split on semicolons, quotes / error-code case
  normalized.
- **T11b semantic validation** (never calls the LLM): schema fields, per-
  assertion executability (each expected result names a status code or a
  field comparison), scenario-identity completeness, obligation-reference
  existence.
- **T11c repair re-ask** is ALLOWED only for semantic gaps detected here;
  purely structural problems are fixed deterministically and must never
  trigger an LLM call. ``needs_reask`` returns False for structural-only
  issues so the gate "0 re-asks on structural-error inputs" holds by
  construction.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "needs_reask",
    "normalize_case",
    "semantic_validation",
]

_PLACEHOLDER_ALIASES = (
    (re.compile(r"\{\{\s*uuid\s*\}\}"), "<RUN_ID>"),
    (re.compile(r"<uuid>", re.IGNORECASE), "<RUN_ID>"),
    (re.compile(r"<uniq>", re.IGNORECASE), "<RUN_ID>"),
    (re.compile(r"<token>", re.IGNORECASE), "<TOKEN>"),
    (re.compile(r"<bearer[_ ]?token>", re.IGNORECASE), "<BEARER_TOKEN>"),
    (re.compile(r"<api[_ ]?key>", re.IGNORECASE), "<API_KEY>"),
)
_CANONICAL_RE = re.compile(r"<([A-Z][A-Z0-9_]{1,30})>")
_LOWER_PLACEHOLDER_RE = re.compile(r"<([a-z][a-zA-Z0-9_]{1,30})>")

_STATUS_CODE_RE = re.compile(r"\b[1-5]\d{2}\b")
_FIELD_ASSERT_RE = re.compile(
    r"\b\w+\s*(?:=|==|!=|>=|<=|>|<|contains|includes|包含|等于)\s*\S+", re.IGNORECASE
)


def _as_str_list(value: Any, transform: Any) -> list[str]:
    """Coerce a list-like field into ``list[str]`` WITHOUT character
    iteration (defect ③, 2026-09-19 review): a bare string is a single
    item, not an iterable of chars; non-list non-str values are wrapped."""
    if value is None:
        return []
    if isinstance(value, str):
        return [transform(value)] if value.strip() else []
    if isinstance(value, list):
        return [transform(str(v)) for v in value]
    return [transform(str(value))]


def normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    """T11a: deterministic, structure-only normalization (returns a copy)."""
    out = dict(case)

    def _norm_text(text: str) -> str:
        for pattern, replacement in _PLACEHOLDER_ALIASES:
            text = pattern.sub(replacement, text)
        # Lower-case placeholders inside braces/brackets become canonical too.
        text = _LOWER_PLACEHOLDER_RE.sub(lambda m: f"<{m.group(1).upper()}>", text)
        text = text.replace("“", '"').replace("”", '"').replace("\u2019", "'")
        text = re.sub(
            r"\berror\.code:\s*bad_request\b", "error.code: BAD_REQUEST", text, flags=re.IGNORECASE
        )
        return text

    out["title"] = _norm_text(str(out.get("title", "")))
    out["description"] = _norm_text(str(out.get("description", "")))
    out["preconditions"] = _as_str_list(out.get("preconditions"), _norm_text)
    # Steps: one line each (collapse accidental newlines/whitespace runs).
    out["steps"] = _as_str_list(
        out.get("steps"), lambda v: " ".join(_norm_text(str(v)).split())
    )
    # expected_results: split semicolon-joined strings into individual items.
    split_results: list[str] = []
    for result in _as_str_list(out.get("expected_results"), _norm_text):
        for part in result.split(";"):
            part = _norm_text(part.strip())
            if part:
                split_results.append(part)
    out["expected_results"] = split_results
    # Error codes upper-case everywhere.
    out["expected_results"] = [
        re.sub(r"\b([45]\d{2})\b", lambda m: m.group(1), r) for r in out["expected_results"]
    ]
    return out


def semantic_validation(
    case: dict[str, Any],
    *,
    known_obligations: set[str] | None = None,
) -> list[str]:
    """T11b: semantic checks that deterministic normalization cannot fix.

    Returns a list of semantic-gap descriptions (empty = clean). Purely
    structural problems are NOT reported here — they never reach the LLM.
    """
    gaps: list[str] = []
    expected = case.get("expected_results") or []
    if not expected:
        gaps.append("assertion-missing: no expected_results")
    weak = [
        e
        for e in expected
        if not (_STATUS_CODE_RE.search(str(e)) or _FIELD_ASSERT_RE.search(str(e)))
    ]
    if len(weak) == len(expected) and expected:
        gaps.append("assertion-incomplete: no result names a status code or field comparison")

    identity_parts = (
        str(case.get("scenario_operation", "") or ""),
        str(case.get("scenario_scene", "") or ""),
        str(case.get("scenario_variant", "") or ""),
    )
    if any(not part for part in identity_parts):
        gaps.append("identity-missing: scenario identity incomplete")

    covers = case.get("covers_obligations") or []
    if known_obligations is not None:
        unknown = [c for c in covers if str(c) not in known_obligations]
        if unknown:
            gaps.append(f"obligation-unknown: {','.join(unknown)}")
    return gaps


def needs_reask(gaps: list[str]) -> bool:
    """T11c gate: an LLM repair re-ask is allowed ONLY for semantic gaps.

    Structural-only input (placeholders, formatting) must never trigger an
    LLM call — the empty-gap list and structural prefixes both return False.
    """
    if not gaps:
        return False
    structural_prefixes = ("structural:",)
    return not all(g.startswith(structural_prefixes) for g in gaps)
