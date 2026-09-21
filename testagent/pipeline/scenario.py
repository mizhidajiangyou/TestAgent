"""Scenario identity + deterministic dedup (T8, fix-plan §3.2).

The model DECLARES a scenario identity per case (part of the output
contract); the program normalizes deterministically and dedups. The
program never guesses intent from natural language, and review only
suggests — it does not delete.

- Dedup key = ``operation + scene + normalized_variant``.
- Same key keeps the STRONGER assertion (more expected_results > field-level
  assertion > status-code assertion; ties keep the first).
- Cases WITHOUT a declared identity are kept and flagged
  ``scenario_identity_missing`` in the report — report noise beats silent
  deletion.
- Every dedup action lands in the returned report (who was removed, by
  what key, replaced by which case).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "SCENE_VOCABULARY",
    "covered_identities",
    "dedup_cases",
    "dedup_key",
    "identity_missing_keys",
    "normalize_identity",
]

#: Controlled scene vocabulary (first version, fix-plan §3.2).
SCENE_VOCABULARY = frozenset(
    {
        "create",
        "read",
        "update",
        "delete",
        "list",
        "auth",
        "pagination",
        "sort",
        "boundary",
        "lockout",
        "error-handling",
        "cleanup",
        "other",
    }
)

#: Variant synonym normalization (applied to whole tokens / key values).
_SYNONYMS: dict[str, str] = {
    "dup": "duplicate",
    "重复": "duplicate",
    "repeated": "duplicate",
}

_PLACEHOLDER_VALUE_RE = re.compile(r'"[^"]*"|\b[\w.+-]*@[\w.-]+\.\w+\b')
_KEY_VALUE_RE = re.compile(r"(\w+)\s*=\s*([^,;]+)")


def _normalize_variant(variant: str) -> str:
    text = (variant or "").strip().lower()
    text = _PLACEHOLDER_VALUE_RE.sub("<value>", text)
    pairs = _KEY_VALUE_RE.findall(text)
    if pairs:
        normalized = sorted((k.strip(), v.strip()) for k, v in pairs)
        text = ",".join(f"{k}={v}" for k, v in normalized)
    tokens = [t.strip() for t in re.split(r"[,;|/]\s*", text) if t.strip()]
    normalized_tokens = [_SYNONYMS.get(tok, tok) for tok in tokens]
    return "|".join(normalized_tokens) if normalized_tokens else ""


def normalize_identity(operation: str, scene: str, variant: str) -> tuple[str, str, str]:
    """Deterministic normalization: lowercase/trim, scene folded into the
    controlled vocabulary (unknown -> ``other``), variant synonyms + key
    sorting + placeholder abstraction."""
    op = (operation or "").strip().upper().replace("  ", " ")
    scene_raw = (scene or "").strip().lower().replace("_", "-")
    if scene_raw not in SCENE_VOCABULARY:
        scene_raw = "other"
    return op, scene_raw, _normalize_variant(variant)


def dedup_key(operation: str, scene: str, variant: str) -> str:
    op, scene_n, variant_n = normalize_identity(operation, scene, variant)
    return f"{op}||{scene_n}||{variant_n}"


def _assertion_strength(case: dict[str, Any]) -> tuple[int, int, int]:
    """Stronger assertion heuristic (fix-plan §3.2): more expected results
    > field-level assertions > status-code assertions."""
    expected = case.get("expected_results") or []
    text = " ".join(str(e) for e in expected).lower()
    field_level = sum(
        1 for e in expected if re.search(r"[.:]\s*\w+\s*=|\bfield\b|字段", str(e).lower())
    )
    status_codes = len(re.findall(r"\b[1-5]\d{2}\b", text))
    return (len(expected), field_level, status_codes)


def _identity_of(case: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(case.get("scenario_operation", "") or ""),
        str(case.get("scenario_scene", "") or ""),
        str(case.get("scenario_variant", "") or ""),
    )


def covered_identities(cases: Iterable[Any]) -> list[dict[str, str]]:
    """The declared identities a later phase must NOT regenerate (T8).

    Injected into the Phase 2 prompt alongside the case summary: the summary
    says what exists, this says which operation+scene+variant combinations are
    already taken. Cases without a complete identity are absent by design —
    the program does not guess intent from natural language.
    """
    return [
        {
            "operation": tc.scenario_operation,
            "scene": tc.scenario_scene,
            "variant": tc.scenario_variant,
        }
        for tc in cases
        if getattr(tc, "scenario_operation", "") and getattr(tc, "scenario_scene", "")
    ]


def identity_missing_keys(cases: Iterable[dict[str, Any]]) -> list[str]:
    """Case ids without a declared identity (kept, reported — never killed)."""
    missing: list[str] = []
    for case in cases:
        op, scene, variant = _identity_of(case)
        if not (op and scene and variant):
            missing.append(str(case.get("id", "")))
    return missing


def dedup_cases(
    cases: list[dict[str, Any]], phase: str = ""
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic identity dedup. Returns ``(kept, removed)`` where each
    removed entry records the key and the winner that replaced it."""
    best_by_key: dict[str, tuple[tuple[int, int, int], dict[str, Any]]] = {}
    order: list[str] = []
    missing_ids: list[str] = []
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []

    for case in cases:
        op, scene, variant = _identity_of(case)
        if not (op and scene and variant):
            missing_ids.append(str(case.get("id", "")))
            kept.append(case)
            continue
        key = dedup_key(op, scene, variant)
        strength = _assertion_strength(case)
        if key not in best_by_key:
            best_by_key[key] = (strength, case)
            order.append(key)
            continue
        prev_strength, prev_case = best_by_key[key]
        if strength > prev_strength:
            best_by_key[key] = (strength, case)
            removed.append(
                {
                    "removed_id": str(prev_case.get("id", "")),
                    "key": key,
                    "kept_id": str(case.get("id", "")),
                    "reason": "stronger assertion",
                }
            )
        else:
            removed.append(
                {
                    "removed_id": str(case.get("id", "")),
                    "key": key,
                    "kept_id": str(prev_case.get("id", "")),
                    "reason": "duplicate identity; earlier kept",
                }
            )

    for key in order:
        kept.append(best_by_key[key][1])

    for entry in removed:
        entry["phase"] = phase
    if missing_ids and removed:
        pass  # report rendering joins both below
    return kept, removed


def render_dedup_report(removed: list[dict[str, Any]], missing_ids: list[str]) -> str:
    """Dedup report: every removal with its key and winner + the
    identity-missing cases (kept by policy)."""
    lines = ["# Scenario dedup report", ""]
    lines.append(f"Removed duplicates: {len(removed)}")
    lines.append(f"Identity-missing cases kept: {len(missing_ids)}")
    lines.append("")
    for entry in removed:
        lines.append(
            f"- removed {entry['removed_id']} (key={entry['key']}) "
            f"in favor of {entry['kept_id']} [{entry['reason']}] ({entry.get('phase', '')})"
        )
    if missing_ids:
        lines.append("")
        lines.append("scenario_identity_missing (kept): " + ", ".join(missing_ids))
    lines.append("")
    return "\n".join(lines)
