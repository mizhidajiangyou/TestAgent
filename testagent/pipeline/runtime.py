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
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from testagent.engine.llm_client import CALL_LABEL, LLMResponse
from testagent.pipeline.review_hooks import (
    DEFAULT_REVIEW_SYSTEM_PROMPT,
    REVIEW_DISABLED,
    REVIEW_FAILED,
    REVIEW_REJECTED,
    REVIEWED,
    ReviewHooks,
    make_list_hooks,
    make_text_hooks,
)
from testagent.pipeline.status import UnitResult, UnitStatus
from testagent.pipeline.validators import strip_fences, validate_structured, validate_text

if TYPE_CHECKING:
    from testagent.engine.llm_client import LLMClient
    from testagent.pipeline.fingerprint import FingerprintLog
    from testagent.pipeline.inputs import TaskContext
    from testagent.pipeline.manifest import StageSpec
    from testagent.pipeline.registry import TaskPackage

logger = logging.getLogger(__name__)

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
        # file: system prompts render with the run context (B5.1: format /
        # output-language variance must reach the system prompt — the
        # fingerprint gate compares it against the legacy generator).
        system_prompt = task.system_prompt(stage.system_prompt, render_ctx)
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


def build_engine_generate_unit(llm: LLMClient) -> Any:
    """FH2.1: engine-backed unit generator (plan-d B6b.1).

    Drives the REAL :class:`TruncationEngine` (truncated salvage → slim
    continue → scope shrink → budget ladder) inside the pipeline unit,
    configured from the manifest's ``TruncationSpec``:

    - ``enabled=false`` falls back to the plain ``build_generate_unit``
      behavior contract (single call, no recovery) — rollback seam;
    - ``scope_key_field`` feeds ``make_dict_hooks``'s declared scope key;
    - ``policy.from_settings`` mirrors ``OPENAI_MAX_OUTPUT_TOKENS`` into
      ``TruncationPolicy.output_token_cap``.

    Outcome → UnitStatus mapping is the existing ``unit_status_from_outcome``
    table (B6a); EngineEvent semantics are untouched (golden diff=0 gate).
    ``build_generate_unit`` stays as the rollback path.
    """

    async def generate_unit(
        task: TaskPackage,
        stage: StageSpec,
        label: str,
        unit_ctx: dict[str, Any],
        ctx: TaskContext,
        session_id: str,
        fingerprint_log: FingerprintLog | None = None,
    ) -> UnitResult:
        render_ctx = {**ctx.parsed, **ctx.settings_views, **_unit_views(unit_ctx)}
        system_prompt = task.system_prompt(stage.system_prompt, render_ctx)
        user_prompt = task.render(stage.template, render_ctx)
        if fingerprint_log is not None:
            fingerprint_log.record(
                model=getattr(llm, "primary_model", "llm"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                params={"via": "pipeline-engine"},
                label=CALL_LABEL.get(""),
            )

        spec = task.manifest.pipeline.truncation
        if not spec.enabled:
            # Rollback seam: delegate to the single-call implementation.
            plain = build_generate_unit(llm)
            result: UnitResult = await plain(
                task, stage, label, unit_ctx, ctx, session_id, fingerprint_log
            )
            return result

        from testagent.config.settings import get_settings
        from testagent.engine.truncation import TruncationEngine, TruncationPolicy
        from testagent.pipeline.truncation_hooks import make_dict_hooks

        policy = TruncationPolicy(output_token_cap=get_settings().llm.max_output_tokens)

        def _extract(raw: str) -> list[dict[str, Any]] | None:
            items = _extract_json_list(raw)
            return items

        def _salvage(raw: str) -> list[dict[str, Any]] | None:
            """Brace-depth salvage for truncated JSON (B4.11: the legacy
            generator's salvage parser cannot be imported, so a compact
            host-side repair lives here — complete the missing closers for
            the outermost array and return the parseable prefix)."""
            text = strip_fences(raw)
            start = text.find("[")
            if start == -1:
                return _extract_json_list(raw)
            depth = 0
            in_str = False
            esc = False
            last_complete = -1
            for i, ch in enumerate(text[start:], start):
                if esc:
                    esc = False
                    continue
                if ch == "\\":
                    esc = True
                    continue
                if ch == '"':
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        last_complete = i
            if last_complete == -1:
                return None
            candidate = text[start : last_complete + 1] + "]"
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                return None
            return (
                [it for it in parsed if isinstance(it, dict)] if isinstance(parsed, list) else None
            )

        def _scope_item_key(item: Any) -> str:
            if spec.scope_key_field:
                if isinstance(item, dict):
                    return str(item.get(spec.scope_key_field, "") or "")
                return str(getattr(item, spec.scope_key_field, "") or "")
            if isinstance(item, dict):
                method = item.get("method", "")
                path = item.get("path", "")
                if method and path:
                    return f"{method} {path}"
                return str(item.get("endpoint", "") or "")
            # APIEndpoint-like scope items: full_path "METHOD /path" is the
            # engine's batch scope key (B6a-2; defect-found in FH2.1 tests —
            # str(dataclass) repr made every item out_of_scope).
            method = getattr(item, "method", "")
            path = getattr(item, "path", "")
            if method and path:
                return f"{method} {path}"
            return str(item)

        hooks = make_dict_hooks(
            extract=_extract,
            salvage=_salvage,
            scope_field=spec.scope_key_field or "endpoint",
            scope_item_key=_scope_item_key,
        )
        engine = TruncationEngine(policy, json_mode=False, hooks=hooks)
        scope_items = _unit_scope_items(unit_ctx)
        items = await engine.arun(llm, system_prompt, user_prompt, scope_items, label)

        if not items:
            return UnitResult(status=UnitStatus.EMPTY)

        schema = task.manifest.artifact.item_schema
        if schema and validate_structured(items, schema):
            return UnitResult(status=UnitStatus.VALIDATION_ERROR, items=items)
        return UnitResult(status=UnitStatus.SUCCESS, items=items)

    return generate_unit


def _unit_scope_items(unit_ctx: dict[str, Any]) -> list[Any]:
    """Scope items for the engine: the split unit's endpoints (batch) or
    the single item; falls back to the full parsed endpoint list."""
    batch = unit_ctx.get("_unit_batch")
    if batch:
        return list(batch)
    item = unit_ctx.get("_unit_item")
    if item is not None:
        return [item]
    return []


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


# ----------------------------------------------------------------------
# Review runtime (plan-d B5.0 / v3 R1+R2+R6 / plan-e E5)
# ----------------------------------------------------------------------

# Terminal states (REVIEWED/REVIEW_REJECTED/REVIEW_FAILED/REVIEW_DISABLED)
# live with the review protocol in review_hooks.py; they are importable
# from here as well for the existing B5.0 import paths (tests, container).

__all__ = [
    "REVIEWED",
    "REVIEW_DISABLED",
    "REVIEW_FAILED",
    "REVIEW_REJECTED",
    "ReviewOutcome",
    "build_engine_generate_unit",
    "build_generate_unit",
    "build_review_runner",
]


@dataclass
class ReviewOutcome:
    """One review pass's outcome (status + final artifact + audit meta)."""

    status: str
    artifact: Any
    meta: dict[str, Any] = field(default_factory=dict)


def _resolve_review_flag(value: bool | str, settings: Any, attr: str, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value == "from_settings":
        return bool(getattr(settings, attr, default))
    return str(value).lower() in ("1", "true", "yes")


def _resolve_review_rounds(value: int | str, settings: Any) -> int:
    if isinstance(value, int):
        return max(0, value)
    if value == "from_settings":
        return max(0, int(getattr(settings, "review_max_rounds", 2)))
    return 2


def _review_context_value(task: Any, spec: Any, render_ctx: dict[str, Any]) -> str | None:
    """Render the review-context block from the task's own template (B5.1).

    Task packages own their review-context SHAPE via
    ``review.context_template`` (perf: load config + endpoint signatures;
    gui: target URL + requirements). The rendered string is byte-equal to
    the legacy generators' Python-assembled ``context_text`` — the
    fingerprint gate pins it. ``None`` when the task declares no third
    context variable or no template for it.

    The render is rstrip()ed: the legacy ``context_text`` never carries
    trailing whitespace, and a file-level trailing newline (editors add
    them) must not leak into the request fingerprint.
    """
    if len(spec.context) < 3 or not spec.context_template:
        return None
    return str(task.render(spec.context_template, render_ctx)).rstrip()


def build_review_runner(llm: Any) -> Any:
    """Create the review runner closure over the LLM client (B5.0).

    The runner reuses :class:`testagent.engine.review.ReviewLoop` — the
    SAME implementation the legacy generators drive (round alternation,
    per-round built-in retry, original-fallback on all-rounds-failed) — so
    review mechanics cannot drift between old and new pipelines. The
    pipeline-specific behaviour (prompt building, parsing, apply, retention
    classification) is injected through :class:`ReviewHooks` (R6).

    Signature: ``runner(task, artifact, snapshot_artifact, ctx) -> ReviewOutcome``.
    ``snapshot_artifact`` is the pre-review snapshot content used as the
    REVIEW_FAILED fallback (R2).
    """

    async def run_review(
        task: Any,
        artifact: Any,
        snapshot_artifact: Any,
        ctx: Any,
        settings: Any = None,
    ) -> ReviewOutcome:
        from typing import cast

        from testagent.engine.review import ReviewLoop

        spec = task.manifest.review
        settings = settings if settings is not None else getattr(ctx, "settings_views", None)
        if not _resolve_review_flag(spec.enabled, settings, "review_enabled", False):
            return ReviewOutcome(REVIEW_DISABLED, artifact, {"status": REVIEW_DISABLED})
        max_rounds = _resolve_review_rounds(spec.max_rounds, settings)
        if max_rounds <= 0 or not spec.template:
            return ReviewOutcome(REVIEW_DISABLED, artifact, {"status": REVIEW_DISABLED})
        # Legacy script generators skip review for empty artifacts
        # (``if self._review_enabled and script``) — an empty script must
        # not burn a review round (B5.1 parity).
        if isinstance(artifact, str) and not artifact:
            return ReviewOutcome(
                REVIEW_DISABLED, artifact, {"status": REVIEW_DISABLED, "reason": "empty_artifact"}
            )

        render_ctx = {**ctx.parsed, **ctx.settings_views}
        # Script-review templates (semantically-equivalent migrations of the
        # legacy script_review_prompt.j2) name their variables via
        # review.context — e.g. ["script", "round", "context_text"]; the
        # third slot is the review-context block, rendered once from the
        # task's own ``review.context_template`` (B5.1). Defaults keep the
        # pipeline-native names (artifact/round) working.
        artifact_var, round_var = "artifact", "round"
        if len(spec.context) >= 2:
            artifact_var, round_var = spec.context[0], spec.context[1]
        inject: dict[str, Any] = {
            artifact_var: None,  # filled per round below
            round_var: 1,
        }
        context_value = _review_context_value(task, spec, render_ctx)
        if context_value is not None:
            inject[spec.context[2]] = context_value

        def _render_review(artifact_value: Any, round_idx: int) -> str:
            variables = dict(inject)
            variables[artifact_var] = artifact_value
            variables[round_var] = round_idx
            return str(task.render(spec.template, {**render_ctx, **variables}))

        if isinstance(artifact, list):
            hooks: ReviewHooks[Any] = make_list_hooks(
                lambda serialized, round_idx: _render_review(serialized, round_idx)
            )
        else:
            # Legacy script-review parse guards (B5.1): each review candidate
            # passes the manifest's artifact validators (jmeter: xml checks;
            # k6: none — the when-guards encode the format conditionality).
            def _candidate_valid(candidate: str) -> bool:
                return not validate_text(candidate, task.manifest.artifact.validators, ctx)

            hooks = make_text_hooks(
                lambda script, round_idx: _render_review(script, round_idx),
                validate=_candidate_valid,
            )

        # file: review system prompts render with the run context (B5.1:
        # format / output-language variance, same as the stage prompt).
        system_prompt = (
            task.system_prompt(spec.system_prompt, render_ctx)
            if spec.system_prompt
            else DEFAULT_REVIEW_SYSTEM_PROMPT
        )

        # The ReviewLoop drives LLM mechanics; secondary client for odd
        # rounds (probe for fakes / single-model configs).
        secondary = getattr(llm, "secondary_client", None)
        review_llm = secondary() if callable(secondary) else llm
        loop = ReviewLoop[Any](
            primary_llm=llm,
            review_llm=review_llm,
            # PromptBuilder slot is unused by design (prompt construction is
            # fully injected per call); cast keeps the legacy signature.
            prompt_builder=cast(Any, None),
            max_rounds=max_rounds,
        )

        # Round classification state (R2): a truthy parse that was NOT
        # adopted can only have been rejected by the retention guard.
        tracker = {"parse_ok": 0, "last_parsed": None, "current": artifact}

        def _tracked_parse(raw: str) -> Any | None:
            parsed = hooks.parse(raw)
            if parsed:
                tracker["parse_ok"] += 1
                tracker["last_parsed"] = parsed
            return parsed

        try:
            result = await loop.arun(
                artifact,
                build_prompt=lambda current, round_idx: (
                    system_prompt,
                    hooks.build_prompt(current, round_idx),
                ),
                parse=_tracked_parse,
                label=f"{task.name}:review",
            )
        except Exception as exc:  # review blew up → snapshot fallback (R2)
            logger.warning("review failed for task %s: %s", task.name, exc)
            return ReviewOutcome(
                REVIEW_FAILED,
                snapshot_artifact,
                {"status": REVIEW_FAILED, "failure_reason": str(exc)[:200]},
            )

        if result.used_review:
            accepted = hooks.apply(artifact, result.artifact)
            return ReviewOutcome(
                REVIEWED,
                accepted,
                {
                    "status": REVIEWED,
                    "rounds_executed": result.rounds_executed,
                    "rounds_succeeded": result.rounds_succeeded,
                },
            )
        if tracker["parse_ok"] > 0:
            # Parsed fine but never adopted → retention rejection. Sanity:
            # the hook guard must agree (no other adoption-failure cause).
            assert not hooks.retention_check(artifact, tracker["last_parsed"])
            return ReviewOutcome(
                REVIEW_REJECTED,
                artifact,  # original kept (R2)
                {
                    "status": REVIEW_REJECTED,
                    "rounds_executed": result.rounds_executed,
                    "rejected_reason": "retention_guard",
                },
            )
        return ReviewOutcome(
            REVIEW_FAILED,
            snapshot_artifact,  # R2: unparseable → snapshot content
            {
                "status": REVIEW_FAILED,
                "rounds_executed": result.rounds_executed,
                "failure_reason": "unparseable_review_output",
            },
        )

    return run_review
