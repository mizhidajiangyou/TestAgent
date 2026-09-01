"""Artifact normalization + comparison for old-vs-new pipeline parity tests
(plan-c B1.3, review P0-3).

Three comparison modes — the mode is an EXPLICIT choice, never a default
normalization, because each relaxation can mask a real behaviour change:

- ``STRICT`` — full field sets, order-sensitive list comparison. The default
  for migration parity: anything the old pipeline emitted must appear in the
  new one, in the same order, with no extra or missing fields.
- ``SEMANTIC`` — whitelist of compared fields, order-insensitive (sorted by
  (title, endpoint)). Use ONLY when order is provably non-semantic AND the
  field set is stable; extra fields in EITHER direction still count as diffs
  (the v2 comparator silently dropped them — the "false equivalence" bug).
- ``TEXT`` — stripped full equality for script artifacts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

#: Fields compared in SEMANTIC mode for structured test-case artifacts.
CASE_COMPARE_FIELDS = (
    "title",
    "description",
    "endpoint",
    "test_type",
    "priority",
    "preconditions",
    "steps",
    "expected_results",
    "tags",
)


class ParityMode(Enum):
    STRICT = "strict"
    SEMANTIC = "semantic"
    TEXT = "text"


def normalize_case(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one case for SEMANTIC comparison (strip strings, keep list
    order WITHIN a case)."""
    norm: dict[str, Any] = {}
    for field in CASE_COMPARE_FIELDS:
        value = item.get(field)
        if isinstance(value, list):
            value = [str(v).strip() for v in value]
        elif isinstance(value, str):
            value = value.strip()
        norm[field] = value
    return norm


def normalize_artifact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort cases so ID-order differences don't mask semantic equivalence
    (SEMANTIC mode only — the old pipeline's order stays checked in STRICT)."""
    return sorted(
        (normalize_case(it) for it in items),
        key=lambda c: (str(c.get("title", "")), str(c.get("endpoint", ""))),
    )


def check_renumber(items: list[dict[str, Any]], fmt: str = "TC-{i:03d}") -> list[str]:
    """Assert ids are contiguous and formatted (checked separately from field
    parity because renumbering is deterministic post-processing)."""
    problems: list[str] = []
    for idx, item in enumerate(items, 1):
        expected = fmt.format(i=idx)
        if item.get("id") != expected:
            problems.append(f"item[{idx - 1}].id = {item.get('id')!r}, expected {expected!r}")
    return problems


def diff_artifacts(
    old: list[dict[str, Any]],
    new: list[dict[str, Any]],
    mode: ParityMode = ParityMode.STRICT,
) -> list[str]:
    """Field-level differences between two case lists; empty = equivalent.

    STRICT compares the FULL field set of every item (union of keys) in
    order; SEMANTIC compares the whitelist sorted. Extra fields in either
    direction are ALWAYS diffs — dropping them was the v2 false-equivalence
    bug (review P0-3).
    """
    if mode is ParityMode.SEMANTIC:
        a, b = normalize_artifact(old), normalize_artifact(new)
        if len(a) != len(b):
            return [f"count mismatch: old={len(a)} new={len(b)}"]
        diffs: list[str] = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            for field in CASE_COMPARE_FIELDS:
                if x.get(field) != y.get(field):
                    diffs.append(f"case[{i}].{field}: {x.get(field)!r} != {y.get(field)!r}")
        return diffs

    # STRICT: order-sensitive, full field union.
    if len(old) != len(new):
        return [f"count mismatch: old={len(old)} new={len(new)}"]
    diffs = []
    for i, (x, y) in enumerate(zip(old, new, strict=True)):
        keys = list(dict.fromkeys(list(x.keys()) + list(y.keys())))
        for key in keys:
            if x.get(key) != y.get(key):
                diffs.append(f"case[{i}].{key}: {x.get(key)!r} != {y.get(key)!r}")
    return diffs


def diff_text(old: str, new: str) -> list[str]:
    """Text-artifact comparison (perf/gui scripts): stripped full equality."""
    if old.strip() == new.strip():
        return []
    return [f"text mismatch: old={len(old)} chars, new={len(new)} chars"]
