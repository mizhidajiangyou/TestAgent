"""Production wiring of ``generate_unit`` for the PipelineExecutor
(plan-c B4.6 runtime side).

Deliberately free of legacy-generator imports (arch gate, plan-c B4.11):
the unit generator talks to the LLM client directly, renders through the
task package's own jinja environment and validates through the pipeline's
validators. The truncation-engine-backed variant for structured artifacts
arrives with B6a/B6b and is injected here the same way.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from testagent.engine.llm_client import CALL_LABEL, LLMResponse
from testagent.pipeline.status import UnitResult, UnitStatus
from testagent.pipeline.validators import strip_fences, validate_structured, validate_text

if TYPE_CHECKING:
    from testagent.engine.llm_client import LLMClient
    from testagent.pipeline.fingerprint import FingerprintLog
    from testagent.pipeline.inputs import TaskContext
    from testagent.pipeline.manifest import StageSpec
    from testagent.pipeline.registry import TaskPackage

_WRAPPER_KEYS = ("test_cases", "cases", "data", "items")


def _extract_json_list(raw: str) -> list[dict[str, Any]] | None:
    """Parse an LLM response into a list of dicts (fences + envelope)."""
    text = strip_fences(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Salvage: outermost array/object span.
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    if isinstance(parsed, list):
        return [it for it in parsed if isinstance(it, dict)]
    if isinstance(parsed, dict):
        for key in _WRAPPER_KEYS:
            value = parsed.get(key)
            if isinstance(value, list):
                return [it for it in value if isinstance(it, dict)]
        return [parsed]
    return None


def build_generate_unit(llm: LLMClient) -> Any:
    """Create the async unit generator closure over the LLM client."""

    async def generate_unit(
        task: TaskPackage,
        stage: StageSpec,
        label: str,
        unit_ctx: dict[str, Any],
        ctx: TaskContext,
        session_id: str,
        fingerprint_log: FingerprintLog | None = None,
    ) -> UnitResult:
        # Split-specific context: the unit's item(s) ride alongside the full
        # parsed context so templates can render ``_unit_item`` / ``_unit_batch``.
        render_ctx = {**ctx.parsed, **ctx.settings_views, **_unit_views(unit_ctx)}
        system_prompt = task.system_prompt(stage.system_prompt)
        user_prompt = task.render(stage.template, render_ctx)
        if fingerprint_log is not None:
            fingerprint_log.record(
                model=getattr(llm, "primary_model", "llm"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                params={"via": "pipeline"},
                label=CALL_LABEL.get(""),
            )

        result: LLMResponse = await llm.achat_with_meta(system_prompt, user_prompt)
        raw = result.text or ""
        if not raw.strip():
            return UnitResult(status=UnitStatus.EMPTY)

        if stage.output.contract == "text":
            script = strip_fences(raw)
            problems = validate_text(script, task.manifest.artifact.validators, ctx)
            if problems:
                return UnitResult(status=UnitStatus.INVALID)
            return UnitResult(status=UnitStatus.SUCCESS, items=[{"script": script}])

        items = _extract_json_list(raw)
        if items is None:
            return UnitResult(status=UnitStatus.INVALID)
        schema = task.manifest.artifact.item_schema
        if schema and validate_structured(items, schema):
            return UnitResult(status=UnitStatus.VALIDATION_ERROR, items=items)
        return UnitResult(status=UnitStatus.SUCCESS, items=items)

    return generate_unit


def _unit_views(unit_ctx: dict[str, Any]) -> dict[str, Any]:
    """Expose split payloads under stable template variable names."""
    views: dict[str, Any] = {}
    item = unit_ctx.get("_unit_item")
    batch = unit_ctx.get("_unit_batch")
    if item is not None:
        views["unit_item"] = item
    if batch is not None:
        views["unit_batch"] = batch
    return views
