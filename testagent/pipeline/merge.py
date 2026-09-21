"""Stage-result merging (plan-c B4.5d): concat + baseline + dedup + renumber."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from testagent.pipeline.inputs import TaskContext
    from testagent.pipeline.manifest import MergeSpec


@dataclass
class StageResult:
    """One stage's aggregated outcome (units already flattened)."""

    name: str
    items: list[dict[str, Any]] = field(default_factory=list)
    units_total: int = 0
    units_failed: int = 0


def _as_dicts(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [it for it in value if isinstance(it, dict)]
    return []


def _dedup_key(
    item: dict[str, Any], keys: list[str], normalize: str, namespace: str = ""
) -> tuple[Any, ...]:
    """H1 dedup key, optionally namespaced by one column (LINK-S6b, v15 §7.2).

    With ``namespace="path_id"`` two cases that match on every dedup key are
    still duplicates ONLY inside the same path: an L3b case for path A must not
    be swallowed by an ordinary case (empty path) that happens to share its
    title/endpoint/type. Links-off runs have no path_id anywhere, so the
    namespace part is the empty string for every item and the grouping is
    exactly the frozen one.
    """
    parts = [str(item.get(k, "")) for k in keys]
    if normalize == "lower":
        parts = [p.lower() for p in parts]
    if namespace:
        return (str(item.get(namespace, "") or ""), *tuple(parts))
    return tuple(parts)


def apply_merge(
    spec: MergeSpec,
    stage_results: list[StageResult],
    ctx: TaskContext,
) -> list[dict[str, Any]]:
    """Concat stages, prepend baseline, dedup, renumber (plan-c B4.5d).

    The baseline (historical cases) is prepended FIRST — matching the legacy
    generator's merge order — then net-new stage items, deduplicated against
    the baseline keys (the historical version wins), finally renumbered.
    """
    items: list[dict[str, Any]] = []
    baseline = ctx.parsed.get(spec.baseline_input) if spec.baseline_input else None
    if baseline:
        items.extend(_as_dicts(baseline))
    for result in stage_results:
        items.extend(result.items)

    if spec.dedup.keys:
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for item in items:
            key = _dedup_key(
                item, spec.dedup.keys, spec.dedup.normalize, spec.dedup.namespace_field
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        items = deduped

    if spec.renumber.field:
        for idx, item in enumerate(items, 1):
            item[spec.renumber.field] = spec.renumber.format.format(i=idx)
    return items
