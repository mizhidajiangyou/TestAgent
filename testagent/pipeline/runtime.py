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

from testagent.config.prompt_contract import append_authoritative_table
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
from testagent.pipeline.testcase_adapter import dict_to_testcase, testcase_to_full_dict
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


def _render_unit_prompts(
    task: TaskPackage, stage: StageSpec, unit_ctx: dict[str, Any], ctx: TaskContext
) -> tuple[str, str]:
    """One unit's ``(system, user)`` pair — the ONLY place that assembles it.

    Both generate_unit variants (single-call and engine-backed) call this, so
    the rollback path cannot drift from the production path, and the legacy
    append order (template, then the run's authoritative value table) holds for
    either. Links material is appended LAST and only when a links pass exists:
    a links-off run must stay byte-identical to the pre-links recording.
    """
    render_ctx = {
        **ctx.parsed,
        **ctx.settings_views,
        **_unit_views(unit_ctx),
        **_prompt_views(task, unit_ctx, ctx),
    }
    system_prompt = task.system_prompt(stage.system_prompt, render_ctx)
    user_prompt = task.render(stage.template, render_ctx)
    table = ctx.quality.conflict_table if ctx.quality is not None else ""
    user_prompt = append_authoritative_table(user_prompt, table)
    if ctx.links is not None:
        # L0 index (+ L1 neighbours of this batch), v15 §5.4: batching material.
        block = ctx.links.context_block(stage.name, unit_ctx)
        if block:
            user_prompt = f"{user_prompt}\n\n{block}"
    return system_prompt, user_prompt


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
        system_prompt, user_prompt = _render_unit_prompts(task, stage, unit_ctx, ctx)
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
                # A rejected script used to vanish silently: the run said
                # "1 unit failed" and left no reason anywhere, because the text
                # chain has no raw-audit sink. The validators know exactly why.
                logger.warning(
                    "[%s] script rejected by validators %s (%d chars returned): %s",
                    CALL_LABEL.get(""),
                    [v.kind for v in task.manifest.artifact.validators],
                    len(script),
                    "; ".join(problems),
                )
                return UnitResult(status=UnitStatus.INVALID)
            return UnitResult(status=UnitStatus.SUCCESS, items=[{"script": script}])

        items = _extract_json_list(raw)
        if items is None:
            logger.warning(
                "[%s] response is not parseable as a JSON list (%d chars): %.120s",
                CALL_LABEL.get(""),
                len(raw),
                raw.strip(),
            )
            return UnitResult(status=UnitStatus.INVALID)
        schema = task.manifest.artifact.item_schema
        if schema and validate_structured(items, schema):
            return UnitResult(status=UnitStatus.VALIDATION_ERROR, items=items)
        return UnitResult(status=UnitStatus.SUCCESS, items=items)

    return generate_unit


def build_engine_generate_unit(
    llm: LLMClient, *, output_token_cap: int, json_mode: bool = False
) -> Any:
    """FH2.1: engine-backed unit generator (plan-d B6b.1).

    Drives the REAL :class:`TruncationEngine` (truncated salvage → slim
    continue → scope shrink → budget ladder) inside the pipeline unit,
    configured from the manifest's ``TruncationSpec``:

    - ``enabled=false`` falls back to the plain ``build_generate_unit``
      behavior contract (single call, no recovery) — rollback seam;
    - ``scope_key_field`` feeds ``make_dict_hooks``'s declared scope key;
    - ``policy.from_settings`` mirrors ``OPENAI_MAX_OUTPUT_TOKENS`` into
      ``TruncationPolicy.output_token_cap`` — resolved by the composition
      root and passed in as ``output_token_cap``, never re-read from the
      settings singleton here (an injected Settings would be ignored, and
      a cached singleton makes the run depend on test/process history).

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
        spec = task.manifest.pipeline.truncation
        if not spec.enabled:
            # Rollback seam: delegate to the single-call implementation. This
            # must happen BEFORE rendering/recording — otherwise a delegated
            # unit records two fingerprints (one per path) and the plain path
            # it is supposed to reproduce byte-for-byte no longer matches.
            plain = build_generate_unit(llm)
            result: UnitResult = await plain(
                task, stage, label, unit_ctx, ctx, session_id, fingerprint_log
            )
            return result

        system_prompt, user_prompt = _render_unit_prompts(task, stage, unit_ctx, ctx)
        if fingerprint_log is not None:
            fingerprint_log.record(
                model=getattr(llm, "primary_model", "llm"),
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                params={"via": "pipeline-engine"},
                label=CALL_LABEL.get(""),
            )

        from testagent.engine.truncation import TruncationEngine, TruncationPolicy
        from testagent.pipeline.truncation_hooks import make_dict_hooks

        quality = ctx.quality
        policy = TruncationPolicy(output_token_cap=output_token_cap)

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
            """Scope key of a BATCH OBJECT (the endpoint side). APIEndpoint
            objects key on "METHOD /path" (full_path, B6a-2); dicts key on
            method+path or their "endpoint" field. ``spec.scope_key_field``
            applies to the PRODUCED dict items (make_dict_hooks' item side),
            NOT to these scope objects — an APIEndpoint has no "endpoint"
            attribute and keying it through the field made the whole batch
            set empty (every item out_of_scope, FH2.1 debug finding)."""
            if isinstance(item, dict):
                if spec.scope_key_field:
                    return str(item.get(spec.scope_key_field, "") or "")
                method = item.get("method", "")
                path = item.get("path", "")
                if method and path:
                    return f"{method} {path}"
                return str(item.get("endpoint", "") or "")
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
            # Obligation-driven quota floor (T7) — absent unless the package
            # declares the quality line, in which case the executor attaches a
            # per-run pass onto the TaskContext.
            expected_for=quality.expected_for if quality is not None else None,
        )
        engine = TruncationEngine(
            policy,
            json_mode=json_mode,
            hooks=hooks,
            # T1: every engine call lands a raw row. Without the sink the
            # session's reconciliation reports ``raw_calls: 0`` beside 60
            # delivered cases — an audit that cannot reconcile at all.
            raw_sink=quality.emit_raw_record if quality is not None else None,
        )
        scope_items = _unit_scope_items(unit_ctx, ctx)
        items = await engine.arun(llm, system_prompt, user_prompt, scope_items, label)

        if not items:
            return UnitResult(status=UnitStatus.EMPTY)

        schema = task.manifest.artifact.item_schema
        if schema and validate_structured(items, schema):
            return UnitResult(status=UnitStatus.VALIDATION_ERROR, items=items)
        if quality is not None:
            # Same conversion the legacy chain ran per batch (T8 identity dedup,
            # T11a normalization, endpoint scoping, T11b semantic validation).
            # Schema validation above stays on the raw model output.
            endpoints = list(ctx.parsed.get("endpoints") or [])
            raw_items = list(items)
            items = [
                testcase_to_full_dict(tc) for tc in quality.to_test_cases(raw_items, endpoints)
            ]
        else:
            raw_items = list(items)
        if ctx.links is not None:
            # v15 §6.1: identity fields are program-owned. The attempt is read
            # off the raw output (the conversion drops those columns, so the
            # artifact alone would show a clean — and unrecorded — claim).
            ctx.links.note_forgeries(raw_items, stage.name)
            unit_item = unit_ctx.get("_unit_item")
            path_id = str(unit_item.get("path_id", "")) if isinstance(unit_item, dict) else ""
            ctx.links.stamp(items, stage=stage.name, path_id=path_id)
        return UnitResult(status=UnitStatus.SUCCESS, items=items)

    return generate_unit


def _unit_scope_items(unit_ctx: dict[str, Any], ctx: TaskContext) -> list[Any]:
    """Engine scope items (legacy parity, FH2.1 debug finding): phase2 batch
    units scope to the batch endpoints; EVERYTHING ELSE — including phase1
    per-input requirement units — scopes to the full spec endpoint list
    (the legacy generator passes ``endpoints`` for both phases; requirement
    text items are NOT scope keys). No endpoints at all -> empty list, the
    engine's filter is bypassed (pure-requirements behavior)."""
    batch = unit_ctx.get("_unit_batch")
    if batch:
        return list(batch)
    endpoints = ctx.parsed.get("endpoints")
    if endpoints:
        return list(endpoints)
    item = unit_ctx.get("_unit_item")
    if item is not None and getattr(item, "method", None) and getattr(item, "path", None):
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


def _prompt_views(task: TaskPackage, unit_ctx: dict[str, Any], ctx: TaskContext) -> dict[str, Any]:
    """Template variables as PROMPT TEXT, in the units the templates expect.

    ``tasks/*/prompts`` were migrated byte-for-byte from the legacy templates,
    and those templates spell their material ``{{ endpoints }}`` /
    ``{{ requirements }}`` — which the legacy builders passed as *rendered
    strings* scoped to the unit (this batch's signature, this one requirement).
    The parsed run context holds objects instead, so a naive render hands the
    model an ``[APIEndpoint(method='GET', ...)]`` repr of the WHOLE spec on
    every unit: valid JSON out, silently degraded prompting.

    Which rendering a variable carries is declared per package
    (``manifest.prompt_views``) because the legacy builders disagreed: the
    testcase chain shows the rich signature, perf and gui the plain text.
    """
    from testagent.config.models import APIEndpoint, RequirementItem
    from testagent.config.prompt_contract import case_coverage_text
    from testagent.parsers.requirement_parser import RequirementParser
    from testagent.parsers.swagger_parser import SwaggerParser, endpoints_to_rich_signature

    mapping = task.manifest.prompt_views
    views: dict[str, Any] = {}
    if not mapping:
        return views

    endpoints = [ep for ep in (ctx.parsed.get("endpoints") or []) if isinstance(ep, APIEndpoint)]
    scoped = [ep for ep in (unit_ctx.get("_unit_batch") or []) if isinstance(ep, APIEndpoint)]
    unit_item = unit_ctx.get("_unit_item")
    if isinstance(unit_item, APIEndpoint):
        scoped = [unit_item]
    scope = scoped or endpoints
    style = mapping.get("endpoints")
    if style == "rich":
        views["endpoints"] = endpoints_to_rich_signature(scope) if scope else ""
    elif style == "text":
        views["endpoints"] = SwaggerParser.endpoints_to_text(scope) if scope else ""

    requirements = [
        req for req in (ctx.parsed.get("requirements") or []) if isinstance(req, RequirementItem)
    ]
    if isinstance(unit_item, RequirementItem):
        requirements = [unit_item]
    if mapping.get("requirements") == "text":
        views["requirements"] = (
            RequirementParser.requirements_to_text(requirements) if requirements else ""
        )

    historical = ctx.parsed.get("historical_cases")
    if mapping.get("historical_cases") == "coverage":
        items = historical if isinstance(historical, list) else []
        views["historical_cases"] = case_coverage_text(
            [tc for tc in (dict_to_testcase(item) for item in items) if tc]
        )
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

    No trimming here: Jinja already drops the TEMPLATE's own final newline
    (``keep_trailing_newline`` is off), so what remains is content. Stripping
    further ate a real newline the legacy GUI context carries between the
    requirements block and the next section header.
    """
    if len(spec.context) < 3 or not spec.context_template:
        return None
    return str(task.render(spec.context_template, render_ctx))


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
