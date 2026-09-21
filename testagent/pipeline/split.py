"""Unit splitting + stage gating (plan-c B4.5b)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from testagent.pipeline.inputs import TaskContext
    from testagent.pipeline.manifest import StageSpec, WhenSpec


def make_units(stage: StageSpec, ctx: TaskContext) -> list[tuple[str, dict[str, Any]]]:
    """Return ``(label, unit_context)`` pairs per split strategy.

    - single: one unit carrying the whole parsed context.
    - per_input: one unit per item of ``split.input`` (item under
      ``_unit_item``).
    - batch: units of ``split.batch_size`` items (``_unit_batch``).
    """
    split = stage.split
    if split.by == "single":
        return [(stage.name, dict(ctx.parsed))]

    items = ctx.parsed.get(split.input, [])
    if not isinstance(items, list):
        items = [items]
    if split.by == "per_input":
        return [
            (f"{stage.name} {i}/{len(items)}", {**ctx.parsed, "_unit_item": it})
            for i, it in enumerate(items, 1)
        ]

    # batch — and "clusters" without a links pass, which must produce the very
    # same groups AND label format so a links-off run keeps its recorded
    # prompt fingerprints (LINK-S5b).
    def _sized_batches() -> list[list[Any]]:
        size = max(1, split.batch_size)
        return [items[i : i + size] for i in range(0, len(items), size)]

    groups = _sized_batches()
    if split.by == "clusters":
        links = getattr(ctx, "links", None)
        if links is not None:
            groups = links.clusters_for(list(items))
    return [
        (f"{stage.name} batch {i}/{len(groups)}", {**ctx.parsed, "_unit_batch": g})
        for i, g in enumerate(groups, 1)
    ]


def evaluate_when(when: WhenSpec, ctx: TaskContext) -> bool:
    """AND over ``input_present`` and NOT over ``input_absent``."""

    def present(name: str) -> bool:
        return bool(ctx.parsed.get(name) or ctx.raw.get(name))

    return all(present(n) for n in when.input_present) and all(
        not present(n) for n in when.input_absent
    )
