"""Generic dict-level engine hooks (plan-d B6a-1 / R4).

The truncation engine (:mod:`testagent.engine.truncation`) is domain-free:
it produces and merges plain ``dict`` items, with the five data
capabilities injected through :class:`GenericHooks`. This module provides
the generic dict-level pieces so pipeline hosts can wire an engine without
writing their own:

- ``scope_key`` reads a configurable item field (manifest
  ``TruncationSpec.scope_key_field``, wired in B6b.1);
- ``dedup_key`` joins configurable item fields with optional lowercasing
  (fuzzy dedup, mirroring the legacy testcase key shape) and folds the
  RESOLVED scope key into the field slot it names, so identity and
  coverage accounting attribute scope-less items identically;
- ``build_reask`` defaults to a faithful generic port of the legacy
  targeted re-ask (product-noun parameterization is B6a-3 scope);
- ``build_continue_context`` stays ``None`` so the engine's built-in
  full-prompt continuation applies unless the host supplies a slim one.

``extract`` / ``salvage`` are deliberately REQUIRED factory inputs:
parsing semantics (fences, envelopes, bracket-balanced salvage) are
host-specific and battle-tested where they exist — this module must not
ship a second, subtly different parser. The legacy TestCaseGenerator
keeps its own adapter (``_item_scope_key`` / ``_engine_dedup_key``)
because its keys normalize ``test_type`` through the domain enum and must
stay byte-equal to the pre-B6a keys.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from testagent.engine.truncation import GenericHooks

__all__ = ["dict_dedup_key", "dict_scope_key", "generic_reask", "make_dict_hooks"]


def dict_scope_key(item: dict[str, Any], field: str = "endpoint") -> str:
    """The item's DECLARED scope key from ``field`` (``""`` when unset)."""
    return str(item.get(field, "") or "")


def dict_dedup_key(
    item: dict[str, Any],
    scope_key: str,
    fields: Sequence[str] = ("title", "endpoint", "test_type"),
    lower: bool = True,
) -> str:
    """Join ``fields`` values into a dedup identity.

    The resolved ``scope_key`` replaces the slot of the field that equals
    the scope field (default ``endpoint``), so the identity uses the same
    attribution the coverage accounting does — an item that declares no
    scope falls back to the run's primary scope instead of an empty slot.
    """
    parts: list[str] = []
    for field in fields:
        value = scope_key if field == "endpoint" else str(item.get(field, "") or "")
        parts.append(value.strip().lower() if lower else value.strip())
    return "|".join(parts)


def generic_reask(
    original_prompt: str,
    failed_output: str,
    label: str,
    error_type: str = "non_parseable",
) -> str:
    """Targeted re-ask prompt (generic port of the legacy builder).

    Tells the LLM its previous output was not usable, shows the failed
    output, and asks for a fix. The wording mirrors the legacy testcase
    generator's ``_build_reask_prompt`` byte-for-byte (the product-noun
    parameterization lands with B6a-3).
    """
    truncated = failed_output[:2000]
    if len(failed_output) > 2000:
        truncated += "\n... [truncated]"

    if error_type == "empty":
        diagnosis = (
            f"Your previous response for '{label}' was COMPLETELY EMPTY "
            "(no content was returned at all)."
        )
        fixes = (
            "Regenerate the COMPLETE valid JSON array from scratch in a single "
            "block. Do not stop early or split the output. Keep each case compact "
            "and produce only the high-value cases so the response stays complete "
            "and within limits."
        )
    elif error_type == "truncated":
        diagnosis = (
            f"Your previous response for '{label}' was TRUNCATED: the JSON "
            "array/object was started but never closed, so it could not be parsed."
        )
        fixes = (
            "Generate FEWER test cases so the output fits within the token limit. "
            "Make each case more compact: shorter descriptions, fewer steps, "
            "concise expected_results. Ensure every object and array is properly closed."
        )
    else:
        diagnosis = (
            f"Your previous response for '{label}' was NOT valid JSON and could not be parsed."
        )
        fixes = (
            "Return ONLY a valid JSON array. Common fixes:\n"
            "- Remove any text before [ or after ]\n"
            "- Remove markdown code fences (```)\n"
            "- Ensure all strings are properly escaped (no unescaped quotes)\n"
            "- Ensure all objects and arrays are properly closed"
        )

    return (
        f"{original_prompt}\n\n"
        f"---\n"
        f"IMPORTANT: {diagnosis}\n\n"
        f"Here is what you returned:\n\n"
        f"{truncated}\n\n"
        f"{fixes}\n"
    )


def make_dict_hooks(
    *,
    extract: Callable[[str], list[Any] | None],
    salvage: Callable[[str], list[Any] | None],
    scope_field: str = "endpoint",
    dedup_fields: Sequence[str] = ("title", "endpoint", "test_type"),
    dedup_lower: bool = True,
    build_reask: Callable[[str, str, str, str], str] | None = None,
    build_continue_context: Callable[[str, list[Any], str, str, dict[str, int]], str] | None = None,
) -> GenericHooks:
    """Assemble dict-level :class:`GenericHooks` (B6b.1's building block —
    wired from manifest ``TruncationSpec`` config there).

    ``extract`` / ``salvage`` are host-supplied on purpose: parsing
    semantics are host-specific and reusing the host's battle-tested
    parser beats shipping a second one here.
    """
    return GenericHooks(
        extract=extract,
        salvage=salvage,
        scope_key=lambda item: dict_scope_key(item, scope_field),
        dedup_key=lambda item, scope_key: dict_dedup_key(
            item, scope_key, dedup_fields, dedup_lower
        ),
        build_reask=build_reask or generic_reask,
        build_continue_context=build_continue_context,
    )
