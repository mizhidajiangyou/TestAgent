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
    # batch
    batches = [
        items[i : i + split.batch_size] for i in range(0, len(items), max(1, split.batch_size))
    ]
    return [
        (f"{stage.name} batch {i}/{len(batches)}", {**ctx.parsed, "_unit_batch": b})
        for i, b in enumerate(batches, 1)
    ]


def evaluate_when(when: WhenSpec, ctx: TaskContext) -> bool:
    """AND over ``input_present`` and NOT over ``input_absent``."""

    def present(name: str) -> bool:
        return bool(ctx.parsed.get(name) or ctx.raw.get(name))

    return all(present(n) for n in when.input_present) and all(
        not present(n) for n in when.input_absent
    )
