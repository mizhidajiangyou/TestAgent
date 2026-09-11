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

from testagent.engine.truncation import EngineContext, GenericHooks

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
    *,
    product: str = "test cases",
    item: str = "case",
    items: str = "cases",
    compactness_hint: str = "shorter descriptions, fewer steps, concise expected_results",
) -> str:
    """Targeted re-ask prompt (generic port of the legacy builder).

    Tells the LLM its previous output was not usable, shows the failed
    output, and asks for a fix. With the default nouns the wording mirrors
    the legacy testcase generator's ``_build_reask_prompt`` byte-for-byte
    (pinned by TestGenericReaskEquivalence); non-testcase hosts override
    the product nouns (B6a-3):

    - ``product``: plural product name ("Generate FEWER {product} ...");
    - ``item`` / ``items``: singular/plural item nouns;
    - ``compactness_hint``: per-item shrink advice (testcase defaults name
      the 10-column fields; other artifacts supply their own).
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
            f"block. Do not stop early or split the output. Keep each {item} compact "
            f"and produce only the high-value {items} so the response stays complete "
            "and within limits."
        )
    elif error_type == "truncated":
        diagnosis = (
            f"Your previous response for '{label}' was TRUNCATED: the JSON "
            "array/object was started but never closed, so it could not be parsed."
        )
        fixes = (
            f"Generate FEWER {product} so the output fits within the token limit. "
            f"Make each {item} more compact: {compactness_hint}. "
            "Ensure every object and array is properly closed."
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
    scope_item_key: Callable[[Any], str] | None = None,
    build_reask: Callable[[str, str, str, str], str] | None = None,
    build_continue_context: Callable[[EngineContext], str] | None = None,
) -> GenericHooks:
    """Assemble dict-level :class:`GenericHooks` (B6b.1's building block —
    wired from manifest ``TruncationSpec`` config there).

    ``extract`` / ``salvage`` are host-supplied on purpose: parsing
    semantics are host-specific and reusing the host's battle-tested
    parser beats shipping a second one here.

    ``scope_item_key`` derives scope keys from the run's opaque scope
    items (B6a-2). When omitted, plain ``str`` items are their own key
    and ``dict`` items read ``scope_field``; any other item type requires
    an explicit ``scope_item_key``.
    """
    if scope_item_key is None:

        def _default_scope_item_key(item: Any) -> str:
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return dict_scope_key(item, scope_field)
            raise TypeError(
                f"cannot derive a scope key from {type(item).__name__}; "
                "pass scope_item_key explicitly"
            )

        resolved_scope_item_key: Callable[[Any], str] = _default_scope_item_key
    else:
        resolved_scope_item_key = scope_item_key

    return GenericHooks(
        extract=extract,
        salvage=salvage,
        scope_key=lambda item: dict_scope_key(item, scope_field),
        dedup_key=lambda item, scope_key: dict_dedup_key(
            item, scope_key, dedup_fields, dedup_lower
        ),
        scope_item_key=resolved_scope_item_key,
        build_reask=build_reask or generic_reask,
        build_continue_context=build_continue_context,
    )
