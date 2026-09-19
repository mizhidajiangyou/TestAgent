"""PipelineExecutor + pre-review snapshots (plan-c B4.6, review P0-3/P1-8/#9).

Orchestration ONLY: renders templates, splits units, calls the LLM through
the truncation engine, merges, snapshots, reviews, writes. Artifact-shape
knowledge lives in hooks/validators; LLM dialects live in the client.

Snapshots are ARTIFACT snapshots (``pre_review``), not pipeline checkpoints
— they let ``checkpoint recover`` hand back the pre-review artifact when a
review explodes, and never claim mid-pipeline resume (that is a future
manifest_version feature; the name says exactly what it is).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from testagent.engine.concurrency import gather_with_concurrency
from testagent.engine.llm_client import CALL_LABEL, ReasoningBudgetExhaustedError
from testagent.engine.truncation import TruncationPolicy
from testagent.pipeline.merge import StageResult, apply_merge
from testagent.pipeline.quality_pass import QualityPass, QualityRunConfig
from testagent.pipeline.review_hooks import REVIEW_DISABLED
from testagent.pipeline.split import evaluate_when, make_units
from testagent.pipeline.status import (
    UnitResult,
    UnitStatus,
    unit_result_from_engine,
    unit_status_from_exception,
)
from testagent.pipeline.testcase_adapter import dict_to_testcase, testcase_to_full_dict

if TYPE_CHECKING:
    from testagent.config.models import TestCase
    from testagent.config.settings import Settings
    from testagent.engine.llm_client import MultiModelLLMClient
    from testagent.engine.model_profiles import Outcome
    from testagent.pipeline.fingerprint import FingerprintLog
    from testagent.pipeline.inputs import TaskContext
    from testagent.pipeline.registry import TaskPackage

logger = logging.getLogger(__name__)

SNAPSHOT_SUFFIX = ".pre_review_snapshot.json"
DEFAULT_SNAPSHOT_KEEP_LAST = 50


def _reduce_text_artifact(items: list[dict[str, Any]]) -> str:
    """Reduce text-unit items ``[{"script": ...}]`` back to the script
    artifact (B5.1). V1 text tasks are single-unit (``split: single``) —
    one unit, one script; a hypothetical multi-unit text task joins its
    scripts in unit order (deterministic, documented)."""
    scripts = [str(it.get("script", "")) for it in items if isinstance(it, dict)]
    return "\n\n".join(scripts)


def _cases_of(items: list[dict[str, Any]]) -> list[TestCase]:
    """Artifact dicts → case objects (tolerant converter; junk entries drop)."""
    return [tc for tc in (dict_to_testcase(item) for item in items) if tc]


@dataclass
class PipelineResult:
    """One pipeline run's outcome."""

    task: str
    artifact: list[dict[str, Any]] | str
    stage_stats: list[StageResult] = field(default_factory=list)
    session_id: str = ""  # every caller can find the snapshot with this
    review_meta: dict[str, Any] | None = None
    units_failed: int = 0


class PipelineExecutor:
    """Drives one task package end to end (plan-c B4.6)."""

    def __init__(
        self,
        llm_client: MultiModelLLMClient,
        settings: Settings,
        generate_unit: Any = None,
        review_runner: Any = None,
    ) -> None:
        """``generate_unit``: callable(task, stage, label, unit_ctx, ctx) ->
        UnitResult. Injected so the executor stays LLM-mechanics-free and
        tests drive it with fakes; production wires the engine-backed
        implementation (see testagent.pipeline.runtime).

        ``review_runner``: async callable(task, artifact, snapshot_artifact,
        ctx, settings) -> ReviewOutcome (plan-d B5.0, injected the same
        way; None disables the review post-process entirely).
        """
        self._llm = llm_client
        self._settings = settings
        self._generate_unit = generate_unit
        self._review_runner = review_runner

    # -- public API ----------------------------------------------------

    def _start_quality(
        self, task: TaskPackage, ctx: TaskContext, session_id: str
    ) -> QualityPass | None:
        """Attach the per-run case quality line when the package opts in.

        Everything the pass needs comes from the injected Settings (never the
        singleton), and its state — obligation ledger, dedup ledger, raw
        dumper — is per-session by construction (defect ⑨'s lesson: session
        state on a shared object cross-pollutes concurrent runs).
        """
        if not task.manifest.pipeline.quality.enabled:
            ctx.quality = None
            return None
        quality = QualityPass(
            QualityRunConfig.from_settings(
                self._settings,
                default_expected_per_endpoint=(
                    TruncationPolicy().default_expected_cases_per_endpoint
                ),
            ),
            session_id=session_id,
        )
        quality.start_session(
            list(ctx.parsed.get("requirements") or []),
            list(ctx.parsed.get("endpoints") or []),
        )
        ctx.quality = quality
        return quality

    async def arun(
        self,
        task: TaskPackage,
        ctx: TaskContext,
        session_id: str | None = None,
        fingerprint_log: FingerprintLog | None = None,
    ) -> PipelineResult:
        """Run the task's stages, merge, snapshot, (review), return."""
        session_id = session_id or uuid.uuid4().hex[:12]
        self._llm.set_session_id(session_id)
        quality = self._start_quality(task, ctx, session_id)

        if self._llm.intent_capable and self._resolve_flag(
            task.manifest.pipeline.verify_model, default=True
        ):
            await self._llm.averify()  # zero-token pre-flight

        results: list[StageResult] = []
        units_failed = 0
        for stage in task.manifest.pipeline.stages:
            if not evaluate_when(stage.when, ctx):
                continue
            units = make_units(stage, ctx)
            outcomes = await gather_with_concurrency(
                self._resolve_concurrency(task),
                *(
                    self._exec_unit(task, stage, label, unit_ctx, ctx, session_id, fingerprint_log)
                    for label, unit_ctx in units
                ),
            )
            if task.manifest.pipeline.fan_out_recover:
                outcomes = await self._recover_empty(
                    task, stage, units, outcomes, ctx, session_id, fingerprint_log
                )
            items = self._flatten(outcomes, stage)
            failed = sum(1 for o in outcomes if o.status is not UnitStatus.SUCCESS)
            units_failed += failed
            results.append(
                StageResult(
                    name=stage.name, items=items, units_total=len(units), units_failed=failed
                )
            )

        merged_items = apply_merge(task.manifest.pipeline.merge, results, ctx)
        # Text artifacts (B5.1): units carry the script as ``{"script": ...}``
        # items through the shared merge machinery; the pipeline-level
        # artifact is the script STRING (legacy parity — a text task's
        # artifact is one script, not a list of items).
        merged: list[dict[str, Any]] | str = (
            _reduce_text_artifact(merged_items)
            if task.manifest.artifact.type == "text"
            else merged_items
        )

        # Quality line (QL-1): the deterministic case passes the legacy chain
        # ran on the merged, pre-review artifact. They run BEFORE the snapshot
        # so the snapshot stays the "content that survived to review" truth.
        if quality is not None and isinstance(merged, list):
            merged = [
                testcase_to_full_dict(tc)
                for tc in quality.post_merge(
                    _cases_of(merged), list(ctx.parsed.get("endpoints") or [])
                )
            ]

        # Pre-review snapshot (review P0-3): the artifact survives a review
        # explosion; recover does NOT re-run anything (no --resume illusion).
        snapshot_artifact = merged
        self._write_snapshot(task.name, session_id, merged)

        # Review post-process (plan-d B5.0, lifecycle R1/R2): runs AFTER the
        # snapshot; REVIEW_FAILED falls back to the snapshot content, and the
        # terminal status rides on review_meta for parity assertions.
        # REVIEW_DISABLED writes NO meta file — the legacy generators wrote
        # ``<stem>.meta.json`` only when a review actually ran (B5.1 parity).
        review_meta: dict[str, Any] | None = None
        if self._review_runner is not None:
            outcome = await self._review_runner(
                task, merged, snapshot_artifact, ctx, self._settings
            )
            merged = outcome.artifact
            if outcome.status != REVIEW_DISABLED:
                review_meta = dict(outcome.meta)
            # T13: review output is not trusted blind — the degenerate-stub
            # cleanup re-runs on whatever the reviewer returned.
            if quality is not None and isinstance(merged, list):
                merged = [
                    testcase_to_full_dict(tc) for tc in quality.after_review(_cases_of(merged))
                ]

        if quality is not None:
            quality.finish_session(_cases_of(merged) if isinstance(merged, list) else [])

        return PipelineResult(
            task=task.name,
            artifact=merged,
            stage_stats=results,
            session_id=session_id,
            review_meta=review_meta,
            units_failed=units_failed,
        )

    # -- unit execution --------------------------------------------------

    async def _exec_unit(
        self,
        task: TaskPackage,
        stage: Any,
        label: str,
        unit_ctx: dict[str, Any],
        ctx: TaskContext,
        session_id: str,
        fingerprint_log: FingerprintLog | None,
    ) -> UnitResult:
        """Render one unit's prompts and generate its artifact slice."""
        token = CALL_LABEL.set(f"{task.name}:{label}")
        try:
            if self._generate_unit is None:
                raise RuntimeError(
                    "PipelineExecutor constructed without generate_unit; wire "
                    "testagent.pipeline.runtime.build_generate_unit()."
                )
            unit_result: UnitResult = await self._generate_unit(
                task, stage, label, unit_ctx, ctx, session_id, fingerprint_log
            )
            return unit_result
        except asyncio.CancelledError:
            return UnitResult(status=UnitStatus.CANCELLED)
        except Exception as exc:
            status = unit_status_from_exception(exc)
            logger.warning("[%s:%s] unit failed (%s): %s", task.name, label, status.value, exc)
            return UnitResult(
                status=status,
                engine_recovered=isinstance(exc, ReasoningBudgetExhaustedError),
            )
        finally:
            CALL_LABEL.reset(token)

    async def _recover_empty(
        self,
        task: TaskPackage,
        stage: Any,
        units: list[tuple[str, dict[str, Any]]],
        outcomes: list[UnitResult],
        ctx: TaskContext,
        session_id: str,
        fingerprint_log: FingerprintLog | None,
    ) -> list[UnitResult]:
        """Fan-out recovery by UnitStatus (plan-c B3.1 + decision D2).

        EMPTY without engine recovery -> one serial re-ask (concurrency=1).
        EMPTY WITH engine recovery -> NOT re-asked (D2): the v10 ladder
        already proved the request shape useless; counted as failed.
        TIMEOUT / PROVIDER_ERROR -> retried serially once (model fallback
        already happened client-side). INVALID / VALIDATION_ERROR ->
        targeted repair is the review layer's job, not a re-ask.
        CANCELLED -> never recovered.
        """
        retryable = [
            i
            for i, o in enumerate(outcomes)
            if (o.status is UnitStatus.EMPTY and not o.engine_recovered)
            or o.status in (UnitStatus.TIMEOUT, UnitStatus.PROVIDER_ERROR)
        ]
        if not retryable or not any(o.status is UnitStatus.SUCCESS for o in outcomes):
            # All-empty means the model is genuinely down — a serial retry
            # would fail identically (legacy _fan_out_recover semantics).
            return outcomes
        logger.info(
            "[%s:%s] recovering %d/%d units serially",
            task.name,
            stage.name,
            len(retryable),
            len(units),
        )
        recovered = await gather_with_concurrency(
            1,
            *(
                self._exec_unit(
                    task, stage, units[i][0], units[i][1], ctx, session_id, fingerprint_log
                )
                for i in retryable
            ),
        )
        merged = list(outcomes)
        for idx, result in zip(retryable, recovered, strict=True):
            merged[idx] = result
        return merged

    def _flatten(self, outcomes: list[UnitResult], stage: Any) -> list[dict[str, Any]]:
        """Concat unit items, unwrapping envelope keys when declared.

        Only SUCCESS (and EMPTY-with-items salvage) outcomes contribute
        items: VALIDATION_ERROR etc. may carry their rejected items for
        auditing (runtime.py contract), and those MUST NOT enter the final
        artifact (defect ⑧, 2026-09-19 review)."""
        items: list[dict[str, Any]] = []
        unwrap = stage.output.unwrap_keys
        for outcome in outcomes:
            if outcome.status not in (UnitStatus.SUCCESS, UnitStatus.EMPTY):
                continue
            for item in outcome.items:
                if unwrap and isinstance(item, dict):
                    for key in unwrap:
                        value = item.get(key)
                        if isinstance(value, list):
                            items.extend(v for v in value if isinstance(v, dict))
                            break
                    else:
                        items.append(item)
                else:
                    items.append(item)
        return items

    # -- snapshots ---------------------------------------------------------

    def _snapshot_path(self, session_id: str) -> Path:
        return Path(self._settings.output_dir) / f"{session_id}{SNAPSHOT_SUFFIX}"

    def _write_snapshot(self, task_name: str, session_id: str, artifact: Any) -> Path:
        final = self._snapshot_path(session_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": session_id,
            "task": task_name,
            "created_at": datetime.now(UTC).isoformat(),
            "stage": "pre_review",
            "count": len(artifact) if isinstance(artifact, list) else 1,
            "artifact": artifact,
        }
        # Atomic write: a crash never leaves a half-written snapshot, and
        # concurrent same-id writers cannot interleave bytes.
        tmp = final.with_name(final.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, final)
        self._prune_snapshots()
        return final

    def read_snapshot(self, session_id: str) -> Any:
        payload = json.loads(self._snapshot_path(session_id).read_text(encoding="utf-8"))
        return payload["artifact"]

    def _prune_snapshots(self) -> None:
        """Keep the newest N snapshots (review #9; env-overridable)."""
        keep = int(os.getenv("CHECKPOINT_KEEP_LAST", DEFAULT_SNAPSHOT_KEEP_LAST))
        out = Path(self._settings.output_dir)
        snapshots = sorted(
            out.glob(f"*{SNAPSHOT_SUFFIX}"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        for stale in snapshots[keep:]:
            stale.unlink(missing_ok=True)

    # -- flag resolution ---------------------------------------------------

    def _resolve_flag(self, value: bool | str, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value == "from_settings":
            return default
        return value.lower() in ("1", "true", "yes")

    def _resolve_concurrency(self, task: TaskPackage) -> int:
        value = task.manifest.pipeline.max_concurrency
        if isinstance(value, int):
            return max(1, value)
        raw = getattr(self._settings.llm, "max_concurrency", 5)
        return max(1, int(raw) if not isinstance(raw, str) else 5)


def list_snapshots(output_dir: str) -> list[dict[str, Any]]:
    """Discover recoverable snapshots without knowing session ids."""
    found: list[dict[str, Any]] = []
    for path in Path(output_dir).glob(f"*{SNAPSHOT_SUFFIX}"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError, OSError:
            continue
        found.append(
            {
                "session_id": payload.get("session_id", path.stem),
                "task": payload.get("task", "?"),
                "created_at": payload.get("created_at", ""),
                "count": payload.get("count", 0),
                "path": str(path),
            }
        )
    return sorted(found, key=lambda c: str(c["created_at"]), reverse=True)


#: Session ids come from the command line (``--resume``, ``checkpoint recover``)
#: and are interpolated into file paths, so an unchecked value is a path
#: traversal vector. One holder for the rule, used by every caller that turns an
#: id into a path. Path separators are what makes a value dangerous, so they
#: are what is rejected (ids we mint are ``uuid4().hex[:12]``; hand-written
#: fixtures legitimately use shorter names).
SESSION_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,64}")


def validate_session_id(session_id: str) -> str:
    """Return the id unchanged, or raise ``ValueError`` when it could escape
    the directory it is joined into."""
    if not SESSION_ID_RE.fullmatch(session_id or ""):
        raise ValueError(
            f"invalid session id: {session_id!r} "
            "(expected 1-64 chars of [A-Za-z0-9._-], no path separators)"
        )
    return session_id


def recover_snapshot(session_id: str, output_dir: str) -> dict[str, Any] | None:
    """Load one snapshot payload by session id (None when absent)."""
    path = Path(output_dir) / f"{validate_session_id(session_id)}{SNAPSHOT_SUFFIX}"
    if not path.exists():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def outcome_of(result: UnitResult, outcome: Outcome) -> UnitResult:
    """Adapter helper kept for symmetry with the mapping table docs."""
    return unit_result_from_engine(
        result.items,
        outcome=outcome,
        finish_reason=None,
        engine_recovered=result.engine_recovered,
    )
