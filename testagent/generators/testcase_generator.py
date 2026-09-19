"""
Test case generator using LLM.

Two-phase strategy:
  Phase 1: Generate from requirements, one concurrent call per requirement.
  Phase 2 (optional): If API endpoints exist, generate API-specific cases
                       (boundary, security, integration) per endpoint batch.
Merge and re-number all cases.

Inspired by guardrails' targeted re-ask: when JSON parse fails, the retry
prompt tells the LLM exactly what went wrong and includes the failed output.

Optional multi-round cross-validation review:
  When ``review_enabled`` is true, the generator runs ``review_max_rounds``
  refinement passes. Odd rounds use the secondary LLM (non-primary model),
  even rounds use the primary model. This way two different models
  cross-validate each other. If only one model is configured, the secondary
  falls back to the primary and a warning is logged by the client.
"""

import csv
import json
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from testagent.config.models import (
    APIEndpoint,
    RequirementItem,
    TestCase,
    TestCaseGenInput,
    TestPriority,
    TestType,
)
from testagent.engine.concurrency import gather_with_concurrency
from testagent.engine.llm_client import JSON_OBJECT_FORMAT, LLMClient
from testagent.engine.prompt_builder import (
    PromptBuilder,
    endpoints_to_rich_signature,
    endpoints_to_signature,
    extract_requirement_summary,
)
from testagent.engine.raw_dump import RawResponseDumper
from testagent.engine.review import ReviewLoop
from testagent.engine.truncation import (
    EngineContext,
    GenericHooks,
    TruncationEngine,
    TruncationPolicy,
    build_continue_prompt,
)
from testagent.generators.base import BaseGenerator
from testagent.parsers.requirement_parser import RequirementParser
from testagent.pipeline.binding import bind_requirements
from testagent.pipeline.consistency import (
    ConflictPolicy,
    Finding,
    RequirementRef,
    classify_findings,
    render_authoritative_table,
    render_gap_report,
)
from testagent.pipeline.executability import grade_case, placeholder_closure_metrics
from testagent.pipeline.normalization import needs_reask, normalize_case, semantic_validation
from testagent.pipeline.obligations import (
    BindingBasis,
    ObligationRegistry,
    register_spec_obligations,
)
from testagent.pipeline.scenario import dedup_cases, render_dedup_report

logger = logging.getLogger(__name__)

#: Keys that may wrap a test-case list when JSON mode is enabled.
_TEST_CASES_WRAPPER_KEYS = ("test_cases", "cases", "data")

#: Max parse retries per batch (guardrails-style targeted re-ask).
MAX_PARSE_RETRIES = 3

#: Requirements are fanned out one concurrent call per requirement (keeps each
#: LLM response small and enables real concurrency even when requirements are
#: not grouped by module). Endpoints still batch in phase 2.

#: Max endpoints per batch in phase 2.
MAX_ENDPOINTS_PER_BATCH = 2

#: Default cross-validation rounds when review is enabled.
DEFAULT_REVIEW_MAX_ROUNDS = 2

#: Default max number of batches generated concurrently in the async path
#: (``agenerate``). The semaphore cap prevents a flood of simultaneous LLM
#: calls (which would trigger 429s). Overridable via ``OPENAI_MAX_CONCURRENCY``.
DEFAULT_MAX_CONCURRENCY = 5

#: CSV column order.
CSV_COLUMNS = [
    "id",
    "title",
    "description",
    "endpoint",
    "test_type",
    "priority",
    "preconditions",
    "steps",
    "expected_results",
    "tags",
    # T8/T10 quality contracts (fix-plan §3.2/§3.4)
    "scenario_operation",
    "scenario_scene",
    "scenario_variant",
    "equivalence_class",
    "covers_obligations",
    "binds",
    "executability",
]


def _item_scope_key(item: dict[str, Any]) -> str:
    """GenericHooks.scope_key adapter: the item's DECLARED endpoint key.

    ``""`` when the item declares none — the engine then attributes it to
    the run's primary scope (legacy converter parity).
    """
    return str(item.get("endpoint", "") or "")


def _engine_dedup_key(item: dict[str, Any], scope_key: str) -> str:
    """GenericHooks.dedup_key adapter — byte-equal to the legacy TestCase
    key (``title|endpoint.full_path|test_type.value``, lowercased title /
    endpoint, test_type normalized through the enum with a functional
    fallback). The resolved scope key replaces the legacy post-conversion
    ``endpoint.full_path`` (same attribution rule)."""
    title = str(item.get("title", "")).strip().lower()
    try:
        test_type = TestType(item.get("test_type", "functional")).value
    except ValueError:
        test_type = TestType.FUNCTIONAL.value
    return f"{title}|{scope_key.lower()}|{test_type}"


def _scope_item_key(endpoint: APIEndpoint) -> str:
    """GenericHooks.scope_item_key adapter (B6a-2): the endpoint's
    ``full_path`` is the batch scope key — the engine itself never reads
    scope-item attributes."""
    return endpoint.full_path


class TestCaseGenerator(BaseGenerator[TestCaseGenInput, list[TestCase]]):
    """Generate test cases from requirements (and optionally API endpoints)."""

    __test__ = False

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_builder: PromptBuilder,
        review_enabled: bool = False,
        review_llm_client: LLMClient | None = None,
        review_max_rounds: int = DEFAULT_REVIEW_MAX_ROUNDS,
        output_language: str = "english",
        json_mode: bool = False,
        max_concurrency: int | None = None,
        verify_model: bool = True,
        truncation_policy: TruncationPolicy | None = None,
        audit_dump_enabled: bool = False,
        audit_dump_dir: str | None = None,
        conflict_policy: str = "strict",
        cases_budget: int = 60,
    ) -> None:
        self._llm = llm_client
        self._prompt_builder = prompt_builder
        self._review_enabled = review_enabled
        # Review client defaults to the primary client when not provided.
        # The container should pass a secondary (non-primary) client so that
        # cross-validation actually alternates between two models.
        self._review_llm: LLMClient = review_llm_client or llm_client
        # Clamp to at least 1 round; 0 effectively disables review even if
        # review_enabled is true (we log a warning in that case).
        self._review_max_rounds = max(1, review_max_rounds)
        self._output_language = output_language
        # When true, every generation/review call requests OpenAI JSON mode and
        # the model wraps its cases in {"test_cases": [...]}. Opt-in only — see
        # settings.OPENAI_JSON_MODE; disabled backends (non-OpenAI compatible)
        # must keep this False.
        self._json_mode = json_mode
        # Bounded concurrency for the async batch fan-out (``agenerate``).
        # None / <=0 falls back to the module default so the generator is safe
        # to construct without explicit settings.
        self._max_concurrency = (
            max_concurrency if max_concurrency and max_concurrency > 0 else DEFAULT_MAX_CONCURRENCY
        )
        # When true, run a zero-token model-availability pre-flight
        # (``self._llm.verify`` / ``averify``) before generation so a bad
        # API key / base_url / model name fails fast instead of hanging for
        # minutes on a generation that can never succeed. Default on.
        self._verify_model = verify_model
        # Correlation id for this run. Set at generate/agenerate start (or
        # passed in for resume) and pushed to the LLM client so every log line
        # and the saved session record share it. Enables resume-by-id.
        self._session_id: str | None = None
        # JSON mode sends response_format={"type":"json_object"} to the LLM.
        # Endpoints that silently ignore this param (some "OpenAI-compatible"
        # proxies, qwen/glm, etc.) will NOT be forced into JSON and may return
        # free-form text -> parse failures. The caller must confirm the backend
        # genuinely enforces JSON mode before enabling it.
        if self._json_mode:
            logger.warning(
                "OPENAI_JSON_MODE is enabled: every generation/review call will "
                "send response_format={'type':'json_object'} to the LLM. Only "
                "backends that truly enforce JSON mode (e.g. real OpenAI) are "
                "safe here. OpenAI-compatible proxies that silently ignore this "
                "param (qwen/glm, some gateways) may return non-JSON and fail to "
                "parse. Keep OPENAI_JSON_MODE=false on such endpoints."
            )

        # Detect whether review will actually use a different model.
        self._review_is_cross_model = self._review_llm is not self._llm
        if review_enabled and not self._review_is_cross_model:
            logger.warning(
                "Review is enabled but the review LLM client is the same as the "
                "primary client. Cross-validation will be single-model only. "
                "Configure multiple models in OPENAI_MODEL to enable true "
                "multi-model cross-validation."
            )

        # Shared review loop (plan v2 §4.3): round alternation, per-round
        # fallback and counters live in engine/review.py; this generator
        # contributes prompt building and its retry-heavy LLM call as hooks.
        self._review_loop = ReviewLoop[list[TestCase]](
            primary_llm=llm_client,
            review_llm=self._review_llm,
            prompt_builder=prompt_builder,
            max_rounds=self._review_max_rounds,
        )

        # Truncation-aware generation loop (plan v6 + v10 B, B6a-1 generalized).
        # The engine owns the loop mechanics and produces plain dict items;
        # this generator is the DOMAIN ADAPTER: it injects the data hooks
        # (extract / salvage / scope / dedup / continue + re-ask) and converts
        # the engine's returned dicts to TestCase objects afterwards
        # (``_to_test_cases`` is no longer an engine hook).
        self._truncation_policy = truncation_policy or TruncationPolicy()
        self._engine = TruncationEngine(
            self._truncation_policy,
            self._json_mode,
            GenericHooks(
                extract=self._extract_json,
                salvage=self._salvage_truncated_json,
                scope_key=_item_scope_key,
                dedup_key=_engine_dedup_key,
                scope_item_key=_scope_item_key,
                build_reask=self._build_reask_prompt,
                build_continue_context=self._build_continue_context,
                expected_for=self._expected_for,
            ),
            raw_sink=self._emit_raw_record,
        )
        # T1 (fix-plan §3.6): raw-response audit. ``audit_dump_dir`` comes
        # from the container (settings.output_dir); direct constructions in
        # tests omit it, so dumping stays off there by default.
        self._audit_dump_enabled = audit_dump_enabled
        self._audit_dump_dir = audit_dump_dir
        self._raw_dumper: RawResponseDumper | None = None
        # T9 (fix-plan §3.3): conflict adjudication policy + per-session
        # authoritative value table shared by generation AND review prompts.
        self._conflict_policy = conflict_policy
        self._conflict_table = ""
        self._conflict_findings: list[Finding] = []
        # T7 (fix-plan §3.5): obligation-driven floor + session cap. The
        # registry holds the coverage state; the floor feeds the engine's
        # ``expected_for`` hook and the cap trims surplus cases WITH per-item
        # accounting (never a silent drop).
        self._cases_budget = cases_budget
        self._obligation_registry: ObligationRegistry | None = None
        self._case_obligations: dict[str, list[str]] = {}
        self._session_case_count = 0
        self._budget_trim_report = ""
        self._dedup_removed: list[dict[str, Any]] = []
        self._dedup_missing: list[str] = []

    def _emit_raw_record(self, record: dict[str, Any]) -> None:
        """Engine ``raw_sink`` adapter: route records to the active dump."""
        dumper = self._raw_dumper
        if dumper is not None:
            dumper.sink(record)

    def _start_raw_audit(self) -> None:
        """Create the per-session dumper when T1 auditing is on.

        Session layout follows the fix-plan convention
        ``<audit_dump_dir>/sessions/<sid>/``.
        """
        base = self._audit_dump_dir
        sid = self._session_id
        if self._audit_dump_enabled and base and sid:
            self._raw_dumper = RawResponseDumper(Path(base) / "sessions", sid)
        else:
            self._raw_dumper = None

    def _start_conflict_context(
        self, requirements: list[RequirementItem], endpoints: list[APIEndpoint]
    ) -> None:
        """T4/T9: run the deterministic consistency checks once per session,
        adjudicate by ``CONFLICT_POLICY`` and share the authoritative value
        table with generation AND review prompts. The gap report lands in
        the session audit directory (when dumping is on) and its summary in
        the log — findings never silently disappear.
        """
        try:
            policy = ConflictPolicy(self._conflict_policy)
        except ValueError:
            raise ValueError(
                f"invalid CONFLICT_POLICY: {self._conflict_policy!r} "
                "(expected strict | spec_first | requirement_first)"
            ) from None
        refs = [
            RequirementRef(
                id=req.id or f"REQ-{i}",
                text=RequirementParser.requirements_to_text([req]),
            )
            for i, req in enumerate(requirements, 1)
        ]
        self._conflict_findings = classify_findings(refs, endpoints)
        self._conflict_table = (
            render_authoritative_table(self._conflict_findings, policy)
            if self._conflict_findings
            else ""
        )
        if self._conflict_findings:
            logger.warning(
                "Consistency check: %d finding(s) under policy=%s",
                len(self._conflict_findings),
                policy.value,
            )
            if self._raw_dumper is not None:
                self._raw_dumper.write_report(
                    "consistency_report.md", render_gap_report(self._conflict_findings)
                )

    def _finish_raw_audit(self, artifacts: list[TestCase]) -> None:
        """Session teardown: write the T1 reconciliation table, the T5
        obligation report, the T7 budget-trim report and the T8 dedup
        report (when auditing is on), then clear session contexts."""
        self._conflict_table = ""
        self._conflict_findings = []
        dumper = self._raw_dumper
        if dumper is None:
            return
        self._raw_dumper = None
        if self._obligation_registry is not None:
            dumper.write_report("obligation_report.md", self._obligation_registry.render_report())
            logger.info("Obligation coverage: %s", self._obligation_registry.summary())
            self._obligation_registry = None
        if self._budget_trim_report:
            dumper.write_report("budget_report.md", self._budget_trim_report)
            self._budget_trim_report = ""
        if self._dedup_removed or self._dedup_missing:
            dumper.write_report(
                "dedup_report.md",
                render_dedup_report(self._dedup_removed, self._dedup_missing),
            )
        dedup_removed = len(self._dedup_removed)
        self._dedup_removed = []
        self._dedup_missing = []
        # Full-chain reconciliation (T1): engine merges minus deterministic
        # removals (T8 dedup, T7 budget trim) must equal the artifact count.
        trimmed = self._budget_trim_report.count("- TC-") if self._budget_trim_report else 0
        path = dumper.write_reconciliation(
            len(artifacts), dedup_removed=dedup_removed, budget_trimmed=trimmed
        )
        logger.info("Raw audit dump written: %s", path)

    def _prompt_extra(self, historical_text: str = "") -> dict[str, str] | None:
        """extra_context for prompt builders: historical baseline + the
        authoritative value table (empty pieces stay absent)."""
        extra: dict[str, str] = {}
        if historical_text:
            extra["historical_cases"] = historical_text
        if self._conflict_table:
            extra["authoritative_table"] = self._conflict_table
        return extra or None

    def _start_obligation_context(
        self, requirements: list[RequirementItem], endpoints: list[APIEndpoint]
    ) -> None:
        """T5/T7: register requirement-AC and spec obligations for this
        session; the uncovered ones drive the per-endpoint quota floor."""
        registry = ObligationRegistry()
        # T6 binding chain (defect ⑤-b: was never invoked, so requirement
        # obligations carried no endpoint_bindings and the T7 floor was
        # permanently zero on the requirement side).
        refs = [
            RequirementRef(
                id=req.id or f"REQ-{i}",
                text=(req.description or "") + " " + " ".join(req.acceptance_criteria or []),
            )
            for i, req in enumerate(requirements, 1)
        ]
        bindings = bind_requirements(refs, endpoints)
        bindings_by_req = {b.requirement_id: b for b in bindings}
        for req in requirements:
            if not req.acceptance_criteria:
                continue
            req_id = req.id or "REQ"
            binding = bindings_by_req.get(req_id)
            registry.register_requirement_acs(
                req_id,
                list(req.acceptance_criteria),
                endpoint_bindings=binding.endpoints if binding and binding.endpoints else (),
                binding_basis=binding.basis if binding else BindingBasis.KEYWORD,
            )
        registry.register_many(register_spec_obligations(endpoints))
        self._obligation_registry = registry

    def _expected_for(self, scope: list[str]) -> dict[str, int]:
        """GenericHooks.expected_for: obligation floor + session budget cap.

        Floor per endpoint = uncovered obligations bound to it (each needs at
        least one case); endpoints without obligations fall back to the
        policy default. The session cap (``CASES_BUDGET``) clamps the total:
        ``0`` disables the cap entirely (legacy behaviour).
        """
        default = self._truncation_policy.default_expected_cases_per_endpoint
        registry = self._obligation_registry
        floor: dict[str, int] = {key: default for key in scope}
        if registry is not None:
            per_endpoint: dict[str, int] = {}
            for state in registry.uncovered():
                for binding in state.obligation.endpoint_bindings:
                    per_endpoint[binding] = per_endpoint.get(binding, 0) + 1
            for key in scope:
                bound = per_endpoint.get(key, 0)
                if bound:
                    floor[key] = bound
        budget = self._cases_budget
        if budget > 0:
            remaining = budget - self._session_case_count
            if remaining <= 0:
                return {key: 0 for key in scope}
            total = sum(floor.values())
            while total > remaining:
                # Trim the largest quota first (deterministic tie-break by key).
                key = max(sorted(floor), key=lambda k: floor[k])
                if floor[key] <= 0:
                    break
                floor[key] -= 1
                total -= 1
        return floor

    @staticmethod
    def _covered_identities(cases: list[TestCase]) -> list[dict[str, str]]:
        return [
            {
                "operation": tc.scenario_operation,
                "scene": tc.scenario_scene,
                "variant": tc.scenario_variant,
            }
            for tc in cases
            if tc.scenario_operation and tc.scenario_scene
        ]

    def _global_identity_dedup(self, cases: list[TestCase]) -> list[TestCase]:
        """T8 wiring 3/3: final global identity dedup over merged cases."""
        raw = [
            {
                "id": tc.id,
                "scenario_operation": tc.scenario_operation,
                "scenario_scene": tc.scenario_scene,
                "scenario_variant": tc.scenario_variant,
                "expected_results": tc.expected_results,
            }
            for tc in cases
        ]
        kept, removed = dedup_cases(raw, phase="global")
        if not removed:
            return cases
        self._dedup_removed.extend(removed)
        kept_ids = {str(c["id"]) for c in kept}
        return [tc for tc in cases if tc.id in kept_ids]

    def _run_executability_gates(self, cases: list[TestCase], endpoints: list[APIEndpoint]) -> None:
        """T10: run Gate-A/Gate-B post-generation, write grades onto the
        artifacts and the metrics into the session report."""
        raw_cases = [self._testcase_to_dict(tc) for tc in cases]
        metrics = placeholder_closure_metrics(raw_cases, endpoints)
        for tc, raw in zip(cases, raw_cases, strict=False):
            result = grade_case(raw, endpoints)
            tc.executability = result.to_meta()
        grades: dict[str, int] = {}
        for tc in cases:
            grade = str(tc.executability.get("grade", "DRAFT"))
            grades[grade] = grades.get(grade, 0) + 1
        logger.info("Executability grades: %s | metrics: %s", grades, metrics)
        if self._raw_dumper is not None:
            lines = [
                "# Executability report",
                "",
                f"Grades: {grades}",
                f"Metrics: {metrics}",
                "",
            ]
            for tc in cases:
                if tc.executability.get("grade") != "INTEGRATION":
                    lines.append(f"- {tc.id}: {tc.executability}")
            self._raw_dumper.write_report("executability_report.md", "\n".join(lines) + "\n")

    def _account_obligations(self, cases: list[TestCase]) -> None:
        """T5 accounting with FINAL (renumbered) case ids: the model emits a
        literal "TC-XXX" id (template rule 11), so id-keyed accounting at
        conversion time put only "TC-XXX" into covered_by (defect ⑥)."""
        registry = self._obligation_registry
        if registry is None:
            return
        for tc in cases:
            if not tc.covers_obligations:
                continue
            try:
                registry.cover_many(tc.id, tc.covers_obligations)
            except KeyError as exc:
                logger.warning("Case %s declares unknown obligation: %s", tc.id, exc)

    def _enforce_cases_budget(self, cases: list[TestCase]) -> list[TestCase]:
        """T7: session cap on total cases with per-item trim accounting.

        Value ordering keeps obligation-covering cases first (the floor
        contract), then original generation order; every trimmed case is
        logged and reported (never a silent drop). ``CASES_BUDGET=0``
        disables the cap.
        """
        budget = self._cases_budget
        if budget <= 0 or len(cases) <= budget:
            return cases
        # Obligation preference reads the case's OWN covers_obligations
        # (defect ⑥: the LLM emits a literal "TC-XXX" id per the template and
        # real ids only exist after renumbering, so an id-keyed map could
        # never match — obligation-covering cases were never preferred).
        covering = [c for c in cases if c.covers_obligations]
        plain = [c for c in cases if not c.covers_obligations]
        kept = (covering + plain)[:budget]
        trimmed = (covering + plain)[budget:]
        lines = [
            "# Budget trim report",
            "",
            f"CASES_BUDGET={budget}, produced={len(cases)}, kept={len(kept)}, trimmed={len(trimmed)}",
            "",
        ]
        for case in trimmed:
            reason = "surplus beyond CASES_BUDGET"
            if case.covers_obligations:
                reason = "surplus beyond CASES_BUDGET (obligation-covering kept preferentially)"
            lines.append(f"- {case.id} {case.title!r}: {reason}")
            logger.warning("Budget trim: dropped %s %r (%s)", case.id, case.title, reason)
        self._budget_trim_report = "\n".join(lines) + "\n"
        return kept

    def set_review_enabled(self, enabled: bool) -> None:
        """Runtime override of the review switch (CLI ``--review/--no-review``).

        Public setter on purpose: callers (CLI/web) must never reach into
        private attributes to flip runtime behavior.
        """
        self._review_enabled = enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        """Return the session id for the most recent (or current) generation."""
        return self._session_id

    def generate(self, data: TestCaseGenInput, session_id: str | None = None) -> list[TestCase]:
        """Generate test cases using two-phase batch strategy.

        Phase 1: Requirements → module-batched generation.
        Phase 2: Endpoints (if any) → endpoint-batched API-specific cases.

        When ``data.historical_cases`` is non-empty, the historical cases are
        used as a baseline: the LLM is asked to produce **only net-new or
        updated** cases for the new requirements (the historical context is
        injected into the prompt so the LLM avoids duplicating coverage).
        Historical cases are then merged with the new cases and re-numbered.
        """
        requirements = data.requirements
        endpoints = data.endpoints
        historical_cases = data.historical_cases

        # Session correlation: set (or accept an externally provided) id and
        # push it to the LLM client so every log line for this run shares it.
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._llm.set_session_id(self._session_id)
        logger.info("Session %s started", self._session_id)
        self._start_raw_audit()
        self._start_conflict_context(requirements, endpoints)
        self._start_obligation_context(requirements, endpoints)

        all_cases: list[TestCase] = []
        phase1_cases: list[TestCase] = []

        # Zero-token pre-flight: fail fast on a bad key / endpoint / model name
        # before spending minutes on a generation that can never succeed.
        if self._verify_model:
            self._llm.verify()

        # --- Phase 1: Requirements-driven generation ---
        if requirements:
            phase1_cases = self._generate_from_requirements(
                requirements, endpoints, historical_cases
            )
            all_cases.extend(phase1_cases)
        elif endpoints:
            # No requirements but have endpoints — generate from endpoints alone
            phase1_cases = self._generate_from_endpoints(endpoints, "")
            all_cases.extend(phase1_cases)
        else:
            logger.warning("No requirements or endpoints provided; nothing to generate.")
            self._finish_raw_audit(all_cases)
            return []

        # --- Phase 2: API-specific enhancement (only if both req + endpoints) ---
        if requirements and endpoints:
            # Feed Phase 1 coverage into Phase 2 so it does NOT regenerate the
            # same scenarios (kills cross-phase duplication / inconsistency).
            covered_text = self._historical_cases_to_text(phase1_cases)
            # T8 wiring 2/3: STRUCTURED covered-identity list (not a text
            # summary) so Phase 2 avoids regenerating the same identity.
            covered_identities = self._covered_identities(phase1_cases)
            if covered_identities:
                covered_text = (
                    f"{covered_text}\n\nAlready-covered scenario identities "
                    "(do NOT regenerate these operation+scene+variant combinations):\n"
                    + json.dumps(covered_identities, ensure_ascii=False)
                )
            api_cases = self._generate_api_specific(
                endpoints, requirements, already_covered=covered_text
            )
            all_cases.extend(api_cases)

        # Boundary cleanup of LLM output (degenerate stubs) — run BEFORE the
        # historical merge so user-provided baseline cases are never dropped,
        # and before review so the reviewer sees a smaller, cleaner list.
        all_cases = self._normalize_cases(all_cases)

        # --- Merge historical cases (baseline) with newly generated cases ---
        if historical_cases:
            merged = self._merge_historical_cases(historical_cases, all_cases)
            all_cases = merged

        # T8 wiring 3/3: global identity dedup before the budget cap.
        all_cases = self._global_identity_dedup(all_cases)
        self._dedup_missing = [
            tc.id
            for tc in all_cases
            if not (tc.scenario_operation and tc.scenario_scene and tc.scenario_variant)
        ]

        # T7: session-wide CASES_BUDGET cap with per-item trim accounting.
        all_cases = self._enforce_cases_budget(all_cases)

        # T10: executability gates (grades land on the artifacts).
        self._run_executability_gates(all_cases, endpoints)

        # Re-number sequentially
        for idx, tc in enumerate(all_cases, 1):
            tc.id = f"TC-{idx:03d}"
        self._account_obligations(all_cases)

        logger.info("Total: %d test cases", len(all_cases))

        # T13: review is no longer the sole owner of dedup/conflict resolution
        # (T8/T9 do it deterministically) - still, flag the combination where
        # a semantic second pass adds value and the user opted out.
        if not self._review_enabled and requirements and len(requirements) > 1 and endpoints:
            logger.info(
                "Hint: multi-requirement + multi-endpoint two-phase generation "
                "ran WITHOUT review; deterministic gates (dedup/consistency/"
                "executability) already applied. Enable --review for an extra "
                "semantic-enhancement pass."
            )

        if self._review_enabled and all_cases:
            all_cases = self._review_and_refine(all_cases, endpoints, requirements)
            all_cases = self._normalize_cases(all_cases)

        self._finish_raw_audit(all_cases)
        return all_cases

    # ------------------------------------------------------------------
    # Async mirror (P1): concurrent batch fan-out via asyncio
    # ------------------------------------------------------------------

    async def agenerate(
        self, data: TestCaseGenInput, session_id: str | None = None
    ) -> list[TestCase]:
        """Async variant of :meth:`generate`.

        Identical two-phase strategy and output, but Phase 1 / Phase 2 batches
        are generated concurrently (bounded by ``self._max_concurrency``) using
        :func:`gather_with_concurrency`. Per-batch re-ask retries stay serial
        within a batch, and review rounds stay serial (data dependency), exactly
        as in the sync path. Call from an async context (the web API) or via
        ``asyncio.run`` (the CLI) to get a real wall-clock speedup whenever
        there is more than one batch.
        """
        requirements = data.requirements
        endpoints = data.endpoints
        historical_cases = data.historical_cases

        # Session correlation: set (or accept an externally provided) id and
        # push it to the LLM client so every log line for this run shares it.
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._llm.set_session_id(self._session_id)
        logger.info("Session %s started", self._session_id)
        self._start_raw_audit()
        self._start_conflict_context(requirements, endpoints)
        self._start_obligation_context(requirements, endpoints)

        all_cases: list[TestCase] = []
        phase1_cases: list[TestCase] = []

        # Zero-token pre-flight (async): same contract as the sync path.
        if self._verify_model:
            await self._llm.averify()

        if requirements:
            phase1_cases = await self._agenerate_from_requirements(
                requirements, endpoints, historical_cases
            )
            all_cases.extend(phase1_cases)
        elif endpoints:
            phase1_cases = await self._agenerate_from_endpoints(endpoints, "")
            all_cases.extend(phase1_cases)
        else:
            logger.warning("No requirements or endpoints provided; nothing to generate.")
            self._finish_raw_audit(all_cases)
            return []

        if requirements and endpoints:
            covered_text = self._historical_cases_to_text(phase1_cases)
            covered_identities = self._covered_identities(phase1_cases)
            if covered_identities:
                covered_text = (
                    f"{covered_text}\n\nAlready-covered scenario identities "
                    "(do NOT regenerate these operation+scene+variant combinations):\n"
                    + json.dumps(covered_identities, ensure_ascii=False)
                )
            api_cases = await self._agenerate_api_specific(
                endpoints, requirements, already_covered=covered_text
            )
            all_cases.extend(api_cases)

        # Boundary cleanup of LLM output (degenerate stubs) — run BEFORE the
        # historical merge so user-provided baseline cases are never dropped,
        # and before review so the reviewer sees a smaller, cleaner list.
        all_cases = self._normalize_cases(all_cases)

        if historical_cases:
            merged = self._merge_historical_cases(historical_cases, all_cases)
            all_cases = merged

        # T8 wiring 3/3: global identity dedup before the budget cap.
        all_cases = self._global_identity_dedup(all_cases)
        self._dedup_missing = [
            tc.id
            for tc in all_cases
            if not (tc.scenario_operation and tc.scenario_scene and tc.scenario_variant)
        ]

        # T7: session-wide CASES_BUDGET cap with per-item trim accounting.
        all_cases = self._enforce_cases_budget(all_cases)

        # T10: executability gates (grades land on the artifacts).
        self._run_executability_gates(all_cases, endpoints)

        # Re-number sequentially
        for idx, tc in enumerate(all_cases, 1):
            tc.id = f"TC-{idx:03d}"
        self._account_obligations(all_cases)

        logger.info("Total: %d test cases", len(all_cases))

        if not self._review_enabled and requirements and len(requirements) > 1 and endpoints:
            logger.info(
                "Hint: multi-requirement + multi-endpoint two-phase generation "
                "ran WITHOUT review; deterministic gates already applied."
            )

        if self._review_enabled and all_cases:
            all_cases = await self._areview_and_refine(all_cases, endpoints, requirements)
            all_cases = self._normalize_cases(all_cases)

        self._finish_raw_audit(all_cases)
        return all_cases

    async def _fan_out_recover(
        self,
        items: list[Any],
        make_coro: Callable[..., Awaitable[list[TestCase]]],
    ) -> list[TestCase]:
        """Fan out ``make_coro`` coroutines concurrently, then retry any unit
        that returned an empty list *sequentially* (concurrency=1).

        Intermittent empty truncation under a single shared model is almost
        always the provider dropping one of several concurrent streams, not a
        request that is too large (see task.md §13 / experience.md #10+).
        Re-running the failed units one-at-a-time removes that parallel
        pressure and usually recovers them — this is the concrete realization
        of the "lower OPENAI_MAX_CONCURRENCY to recover" guidance.

        Recovery only fires when at least one sibling succeeded (the model is
        reachable); if *every* unit failed the model is genuinely down and a
        retry would just fail identically.
        """
        coros = [make_coro(i, item) for i, item in enumerate(items, 1)]
        results = await gather_with_concurrency(self._max_concurrency, *coros)
        failed = [i for i, cases in enumerate(results) if not cases]
        if not failed or not any(results):
            return [c for cases in results for c in cases]
        logger.warning(
            "Generation fan-out: %d/%d units returned empty (concurrent stream "
            "drop suspected). Retrying those %d sequentially (concurrency=1) to "
            "relieve parallel pressure on the single model.",
            len(failed),
            len(items),
            len(failed),
        )
        recovered = await gather_with_concurrency(1, *[make_coro(i + 1, items[i]) for i in failed])
        for idx, new_cases in zip(failed, recovered, strict=True):
            results[idx] = new_cases
        recovered_units = sum(1 for c in recovered if c)
        logger.info(
            "Sequential recovery: %d/%d failed units recovered (%d cases).",
            recovered_units,
            len(failed),
            sum(len(c) for c in recovered),
        )
        return [c for cases in results for c in cases]

    async def _agenerate_from_requirements(
        self,
        requirements: list[RequirementItem],
        endpoints: list[APIEndpoint],
        historical_cases: list[TestCase] | None = None,
    ) -> list[TestCase]:
        """Async mirror of :meth:`_generate_from_requirements` (concurrent).

        Each requirement becomes its own coroutine, so N requirements run in
        parallel (bounded by ``self._max_concurrency``) — this is where the
        async path actually saves wall-clock time. Units that come back empty
        (intermittent concurrent-stream drop) are retried sequentially by
        :meth:`_fan_out_recover`.
        """
        endpoints_text = endpoints_to_rich_signature(endpoints) if endpoints else ""
        historical_text = (
            self._historical_cases_to_text(historical_cases) if historical_cases else ""
        )

        def make_coro(i: int, req: RequirementItem) -> Awaitable[list[TestCase]]:
            async def _one() -> list[TestCase]:
                req_text = RequirementParser.requirements_to_text([req])
                system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
                    endpoints_text=endpoints_text,
                    requirements_text=req_text,
                    output_language=self._output_language,
                    extra_context=self._prompt_extra(historical_text),
                    json_mode=self._json_mode,
                )
                logger.info(
                    "Phase 1 - Requirement %d/%d (module=%s) ...",
                    i,
                    len(requirements),
                    req.module or "default",
                )
                return await self._agenerate_with_retry(
                    system_prompt, user_prompt, endpoints, f"Req {req.id or i}/{len(requirements)}"
                )

            return _one()

        return await self._fan_out_recover(requirements, make_coro)

    async def _agenerate_api_specific(
        self,
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
        already_covered: str = "",
    ) -> list[TestCase]:
        """Async mirror of :meth:`_generate_api_specific` (concurrent)."""
        batches = self._split_endpoint_batches(endpoints)
        req_text = RequirementParser.requirements_to_text(requirements)

        def make_coro(i: int, batch: list[APIEndpoint]) -> Awaitable[list[TestCase]]:
            async def _one() -> list[TestCase]:
                ep_text = endpoints_to_rich_signature(batch)
                system_prompt, user_prompt = self._prompt_builder.build_api_prompt(
                    endpoints_text=ep_text,
                    requirements_text=req_text,
                    output_language=self._output_language,
                    json_mode=self._json_mode,
                    already_covered=already_covered,
                    authoritative_table=self._conflict_table,
                )
                logger.info(
                    "Phase 2 - Batch %d/%d (%d endpoints, API-specific)...",
                    i,
                    len(batches),
                    len(batch),
                )
                return await self._agenerate_with_retry(
                    system_prompt, user_prompt, batch, f"API batch {i}/{len(batches)}"
                )

            return _one()

        return await self._fan_out_recover(batches, make_coro)

    async def _agenerate_from_endpoints(
        self, endpoints: list[APIEndpoint], requirements_text: str
    ) -> list[TestCase]:
        """Async mirror of :meth:`_generate_from_endpoints` (concurrent)."""
        batches = self._split_endpoint_batches(endpoints)

        def make_coro(i: int, batch: list[APIEndpoint]) -> Awaitable[list[TestCase]]:
            async def _one() -> list[TestCase]:
                ep_text = endpoints_to_rich_signature(batch)
                system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
                    endpoints_text=ep_text,
                    requirements_text=requirements_text or "No specific requirements.",
                    output_language=self._output_language,
                    extra_context=self._prompt_extra(),
                    json_mode=self._json_mode,
                )
                logger.info(
                    "Endpoint batch %d/%d (%d endpoints)...",
                    i,
                    len(batches),
                    len(batch),
                )
                return await self._agenerate_with_retry(
                    system_prompt, user_prompt, batch, f"EP batch {i}/{len(batches)}"
                )

            return _one()

        return await self._fan_out_recover(batches, make_coro)

    async def _agenerate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        step_label: str,
        client: LLMClient | None = None,
    ) -> list[TestCase]:
        """Async truncation-aware generation (plan v6) with legacy fallback.

        Delegates to :class:`TruncationEngine.arun`, which drives the
        pending/expected state machine over the rich ``achat_with_meta``
        result (degrading to ``achat`` on legacy clients/mocks). When the
        policy's ``enable_v4_resume`` flag is off, falls back to the fixed
        v2 retry loop (:meth:`_agenerate_v2_legacy`).
        """
        llm = client or self._llm
        if not self._truncation_policy.enable_v4_resume:
            return await self._agenerate_v2_legacy(
                system_prompt, user_prompt, endpoints, step_label, llm
            )
        # B6a-1: the engine returns plain dict items; the adapter converts.
        items = await self._engine.arun(llm, system_prompt, user_prompt, endpoints, step_label)
        return self._to_test_cases(items, endpoints)

    async def _agenerate_v2_legacy(
        self,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        step_label: str,
        llm: LLMClient,
    ) -> list[TestCase]:
        """Legacy v2 fixed-retry loop (kept for rollback / A-B comparison).

        Uses ``await client.achat(...)``; re-ask retries remain serial within
        this batch.
        """
        last_raw = ""
        # See sync mirror: force the "fewer/compact cases" re-ask hint when an
        # LLM call fails (e.g. output truncated to empty beyond the model limit).
        forced_error_type: str | None = None

        for attempt in range(1, MAX_PARSE_RETRIES + 1):
            if attempt == 1:
                effective_prompt = user_prompt
            else:
                error_type = forced_error_type or self._classify_failure(last_raw)
                effective_prompt = self._build_reask_prompt(
                    user_prompt, last_raw, step_label, error_type
                )

            # Retry keeps the SAME token budget. Shrinking max_tokens would make
            # truncation *more* likely (see experience.md #10); the real cause of
            # an empty response is a dropped stream under parallel load, handled
            # by lowering OPENAI_MAX_CONCURRENCY — not by a smaller cap. The
            # compressed-scope re-ask below is what actually reduces per-request
            # size without lowering the budget.
            try:
                raw_response = await llm.achat(
                    system_prompt,
                    effective_prompt,
                    response_format=JSON_OBJECT_FORMAT if self._json_mode else None,
                    max_tokens=None,
                )
            except Exception as exc:
                logger.warning(
                    "%s attempt %d/%d: LLM call failed (%s). Re-asking with a "
                    "compressed/full-regeneration request.",
                    step_label,
                    attempt,
                    MAX_PARSE_RETRIES,
                    exc,
                )
                last_raw = f"<llm error: {exc}>"
                reason = str(exc).lower()
                if "empty" in reason:
                    forced_error_type = "empty"
                elif "truncated" in reason or "too large" in reason or "max_tokens" in reason:
                    forced_error_type = "truncated"
                if attempt < MAX_PARSE_RETRIES:
                    continue
                logger.error(
                    "%s gave up after %d attempts (LLM call errors).",
                    step_label,
                    MAX_PARSE_RETRIES,
                )
                return []

            last_raw = raw_response

            items = self._extract_json(raw_response)
            if items is None:
                items = self._salvage_truncated_json(raw_response)

            if items is not None:
                test_cases = self._to_test_cases(items, endpoints)
                logger.info("%s: %d cases (attempt %d)", step_label, len(test_cases), attempt)
                return test_cases

            logger.warning(
                "%s attempt %d/%d: cannot parse JSON. Preview: %.200s",
                step_label,
                attempt,
                MAX_PARSE_RETRIES,
                raw_response,
            )
            self._dump_debug_response(raw_response)

        logger.error("%s gave up after %d attempts", step_label, MAX_PARSE_RETRIES)
        return []

    async def _areview_and_refine(
        self,
        test_cases: list[TestCase],
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
    ) -> list[TestCase]:
        """Async mirror of :meth:`_review_and_refine`.

        Delegates round control (alternation / fallback / counters) to the
        shared :class:`ReviewLoop`; the LLM call itself stays on this
        generator's retry-heavy ``_agenerate_with_retry`` path. Review rounds
        remain serial — each round consumes the previous round's output.
        """
        build_prompt, _, acall_llm = self._review_hooks(endpoints, requirements)
        result = await self._review_loop.arun(
            test_cases,
            build_prompt=build_prompt,
            acall_llm=acall_llm,
            label="test-cases",
        )
        return self._restore_code_identity(test_cases, result.artifact)

    # ------------------------------------------------------------------
    # Phase 1: Requirements-driven (module-batched)
    # ------------------------------------------------------------------

    def _generate_from_requirements(
        self,
        requirements: list[RequirementItem],
        endpoints: list[APIEndpoint],
        historical_cases: list[TestCase] | None = None,
    ) -> list[TestCase]:
        """Generate test cases from requirements, one call per requirement.

        Each requirement is its own generation unit. This keeps every LLM
        prompt small (avoiding max-token truncation on large specs) and lets
        the async path (``_agenerate_from_requirements``) fan them out
        concurrently — even when requirements are not grouped by module.

        When ``historical_cases`` is provided, a summary of the historical
        coverage is injected into the prompt so the LLM generates only
        net-new or updated cases (avoiding duplicates).
        """
        endpoints_text = endpoints_to_rich_signature(endpoints) if endpoints else ""
        historical_text = (
            self._historical_cases_to_text(historical_cases) if historical_cases else ""
        )

        all_cases: list[TestCase] = []

        for i, req in enumerate(requirements, 1):
            req_text = RequirementParser.requirements_to_text([req])
            system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
                endpoints_text=endpoints_text,
                requirements_text=req_text,
                output_language=self._output_language,
                extra_context=self._prompt_extra(historical_text),
                json_mode=self._json_mode,
            )
            logger.info(
                "Phase 1 - Requirement %d/%d (module=%s) ...",
                i,
                len(requirements),
                req.module or "default",
            )
            cases = self._generate_with_retry(
                system_prompt, user_prompt, endpoints, f"Req {req.id or i}/{len(requirements)}"
            )
            all_cases.extend(cases)

        return all_cases

    # ------------------------------------------------------------------
    # Phase 2: API-specific (endpoint-batched)
    # ------------------------------------------------------------------

    def _generate_api_specific(
        self,
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
        already_covered: str = "",
    ) -> list[TestCase]:
        """Generate API-specific cases (boundary, security, integration).

        ``already_covered`` is a text summary of the Phase 1 cases, injected so
        Phase 2 avoids regenerating scenarios already produced.
        """
        batches = self._split_endpoint_batches(endpoints)
        req_text = RequirementParser.requirements_to_text(requirements)
        all_cases: list[TestCase] = []

        for i, batch in enumerate(batches, 1):
            ep_text = endpoints_to_rich_signature(batch)
            system_prompt, user_prompt = self._prompt_builder.build_api_prompt(
                endpoints_text=ep_text,
                requirements_text=req_text,
                output_language=self._output_language,
                json_mode=self._json_mode,
                already_covered=already_covered,
                authoritative_table=self._conflict_table,
            )
            logger.info(
                "Phase 2 - Batch %d/%d (%d endpoints, API-specific)...",
                i,
                len(batches),
                len(batch),
            )
            cases = self._generate_with_retry(
                system_prompt, user_prompt, batch, f"API batch {i}/{len(batches)}"
            )
            all_cases.extend(cases)

        return all_cases

    def _generate_from_endpoints(
        self, endpoints: list[APIEndpoint], requirements_text: str
    ) -> list[TestCase]:
        """Generate from endpoints only (no requirements)."""
        batches = self._split_endpoint_batches(endpoints)
        all_cases: list[TestCase] = []

        for i, batch in enumerate(batches, 1):
            ep_text = endpoints_to_rich_signature(batch)
            system_prompt, user_prompt = self._prompt_builder.build_testcase_prompt(
                endpoints_text=ep_text,
                requirements_text=requirements_text or "No specific requirements.",
                output_language=self._output_language,
                extra_context=self._prompt_extra(),
                json_mode=self._json_mode,
            )
            logger.info(
                "Endpoint batch %d/%d (%d endpoints)...",
                i,
                len(batches),
                len(batch),
            )
            cases = self._generate_with_retry(
                system_prompt, user_prompt, batch, f"EP batch {i}/{len(batches)}"
            )
            all_cases.extend(cases)

        return all_cases

    # ------------------------------------------------------------------
    # Endpoint batch splitting (requirements fan out per requirement instead)
    # ------------------------------------------------------------------

    @staticmethod
    def _split_endpoint_batches(
        endpoints: list[APIEndpoint],
    ) -> list[list[APIEndpoint]]:
        """Split endpoints into small batches."""
        if not endpoints:
            return []
        return [
            endpoints[i : i + MAX_ENDPOINTS_PER_BATCH]
            for i in range(0, len(endpoints), MAX_ENDPOINTS_PER_BATCH)
        ]

    # ------------------------------------------------------------------
    # LLM call with targeted re-ask (guardrails-inspired)
    # ------------------------------------------------------------------

    def _generate_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        step_label: str,
        client: LLMClient | None = None,
    ) -> list[TestCase]:
        """Sync truncation-aware generation (plan v6) with legacy fallback.

        Delegates to :class:`TruncationEngine.run` (``asyncio.run`` around the
        async-native loop; the sync ``generate`` path never runs inside an
        event loop). Falls back to :meth:`_generate_v2_legacy` when the
        policy's ``enable_v4_resume`` flag is off.

        Args:
            client: Optional LLM client override (used by review rounds to
                alternate between primary and secondary models). Defaults to
                the primary client.
        """
        llm = client or self._llm
        if not self._truncation_policy.enable_v4_resume:
            return self._generate_v2_legacy(system_prompt, user_prompt, endpoints, step_label, llm)
        # B6a-1: the engine returns plain dict items; the adapter converts.
        items = self._engine.run(llm, system_prompt, user_prompt, endpoints, step_label)
        return self._to_test_cases(items, endpoints)

    def _generate_v2_legacy(
        self,
        system_prompt: str,
        user_prompt: str,
        endpoints: list[APIEndpoint],
        step_label: str,
        llm: LLMClient,
    ) -> list[TestCase]:
        """Legacy v2 targeted re-ask loop (guardrails-inspired, kept for rollback).

        Instead of blindly retrying with the same prompt, the re-ask tells
        the LLM exactly what went wrong (parse error / truncation) and
        includes the failed output so it can fix it.
        """
        last_raw = ""
        # When an LLM call itself fails (e.g. output truncated to empty because
        # the request exceeds the model's token limit), force the re-ask to use
        # the "generate fewer / more compact cases" hint rather than the generic
        # "return valid JSON" one.
        forced_error_type: str | None = None

        for attempt in range(1, MAX_PARSE_RETRIES + 1):
            if attempt == 1:
                effective_prompt = user_prompt
            else:
                # Targeted re-ask: classify the failure and tell the LLM
                error_type = forced_error_type or self._classify_failure(last_raw)
                effective_prompt = self._build_reask_prompt(
                    user_prompt, last_raw, step_label, error_type
                )

            # Retry keeps the SAME token budget. Shrinking max_tokens would make
            # truncation *more* likely (see experience.md #10); the real cause of
            # an empty response is a dropped stream under parallel load, handled
            # by lowering OPENAI_MAX_CONCURRENCY — not by a smaller cap. The
            # compressed-scope re-ask below is what actually reduces per-request
            # size without lowering the budget.
            try:
                raw_response = llm.chat(
                    system_prompt,
                    effective_prompt,
                    response_format=JSON_OBJECT_FORMAT if self._json_mode else None,
                    max_tokens=None,
                )
            except Exception as exc:
                logger.warning(
                    "%s attempt %d/%d: LLM call failed (%s). Re-asking with a "
                    "compressed/full-regeneration request.",
                    step_label,
                    attempt,
                    MAX_PARSE_RETRIES,
                    exc,
                )
                last_raw = f"<llm error: {exc}>"
                reason = str(exc).lower()
                if "empty" in reason:
                    forced_error_type = "empty"
                elif "truncated" in reason or "too large" in reason or "max_tokens" in reason:
                    forced_error_type = "truncated"
                if attempt < MAX_PARSE_RETRIES:
                    continue
                logger.error(
                    "%s gave up after %d attempts (LLM call errors).",
                    step_label,
                    MAX_PARSE_RETRIES,
                )
                return []

            last_raw = raw_response

            items = self._extract_json(raw_response)
            if items is None:
                items = self._salvage_truncated_json(raw_response)

            if items is not None:
                test_cases = self._to_test_cases(items, endpoints)
                logger.info("%s: %d cases (attempt %d)", step_label, len(test_cases), attempt)
                return test_cases

            logger.warning(
                "%s attempt %d/%d: cannot parse JSON. Preview: %.200s",
                step_label,
                attempt,
                MAX_PARSE_RETRIES,
                raw_response,
            )
            self._dump_debug_response(raw_response)

        logger.error("%s gave up after %d attempts", step_label, MAX_PARSE_RETRIES)
        return []

    @staticmethod
    def _classify_failure(raw: str) -> str:
        """Classify why the LLM output could not be parsed.

        Returns ``"truncated"`` when the output looks like an incomplete JSON
        array/object (started but never closed), otherwise ``"non_parseable"``
        for outputs with no usable JSON structure. This mirrors guardrails'
        distinction between NonParseableReAsk and truncation handling.
        """
        text = raw.strip()
        has_array_start = "[" in text
        has_array_end = "]" in text
        has_obj_start = "{" in text
        # Heuristic: a JSON array/object was started but never closed
        if (has_array_start and not has_array_end) or (
            has_obj_start and not has_array_end and "}" not in text
        ):
            return "truncated"
        if has_array_start or has_obj_start:
            # Some JSON structure exists but still failed to parse
            return "truncated"
        return "non_parseable"

    def _build_continue_context(self, ctx: EngineContext) -> str:
        """Slim continuation context hook (plan v10 §6 / v8 方案 A).

        Replaces the legacy full-prompt continuation: endpoint signatures +
        requirement summary + fingerprint + pending, WITHOUT the 9KB Rules
        region, worked example or historical cases. Falls back to the legacy
        full-prompt continuation when the requirement section cannot be
        extracted from the rendered prompt (nothing to build a summary from).

        Single-context signature (B6a-3): the five scattered legacy arguments
        arrive via :class:`EngineContext`; ``ctx.scope_items`` are this
        batch's APIEndpoint objects (opaque to the engine).
        """
        char_budget = self._truncation_policy.slim_continue_max_tokens * 2
        summary = extract_requirement_summary(ctx.user_prompt, char_budget)
        if summary is None:
            return build_continue_prompt(ctx.user_prompt, ctx.fingerprint, ctx.label, ctx.pending)
        return self._prompt_builder.build_slim_continue_context(
            endpoints_signature=endpoints_to_signature(ctx.scope_items),
            requirement_summary=summary,
            fingerprint=ctx.fingerprint,
            label=ctx.label,
            pending=ctx.pending,
            max_tokens_budget=self._truncation_policy.slim_continue_max_tokens,
        )

    @staticmethod
    def _build_reask_prompt(
        original_prompt: str,
        failed_output: str,
        label: str,
        error_type: str = "non_parseable",
    ) -> str:
        """Build a targeted re-ask prompt (guardrails pattern).

        Tells the LLM its previous output was not usable, shows the failed
        output, and asks it to fix and return only valid JSON. The hint is
        tailored to the failure type so the LLM knows what to fix.
        """
        # Truncate failed output to avoid token bloat
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

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def _review_hooks(
        self,
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
    ) -> tuple[
        Callable[[list[TestCase], int], tuple[str, str]],
        Callable[[str, str, LLMClient, int], list[TestCase]],
        Callable[[str, str, LLMClient, int], Awaitable[list[TestCase]]],
    ]:
        """Build the artifact-specific hooks for the shared ReviewLoop.

        Returns ``(build_prompt, call_llm, acall_llm)``: prompt construction
        serializes the current cases to the review-prompt JSON contract, and
        the call hooks delegate to this generator's retry-heavy generation
        methods (plan v2 R2: only this generator injects ``call_llm``).
        """
        # Signature format (name/type/required/enum) instead of endpoints_to_text:
        # it is what checklist item "type fidelity" validates against, and it is
        # also smaller — less prompt input lowers review-truncation risk.
        endpoints_text = endpoints_to_signature(endpoints) if endpoints else ""
        requirements_text = RequirementParser.requirements_to_text(requirements)

        def build_prompt(current_cases: list[TestCase], round_idx: int) -> tuple[str, str]:
            current_json = json.dumps(
                [self._testcase_to_dict(tc) for tc in current_cases],
                ensure_ascii=False,
            )
            return self._prompt_builder.build_review_prompt(
                endpoints_text=endpoints_text,
                requirements_text=requirements_text,
                test_cases_json=current_json,
                output_language=self._output_language,
                json_mode=self._json_mode,
                authoritative_table=self._conflict_table,
            )

        def call_llm(
            system_prompt: str, user_prompt: str, client: LLMClient, round_idx: int
        ) -> list[TestCase]:
            return self._generate_with_retry(
                system_prompt,
                user_prompt,
                endpoints,
                f"Review round {round_idx}",
                client=client,
            )

        async def acall_llm(
            system_prompt: str, user_prompt: str, client: LLMClient, round_idx: int
        ) -> list[TestCase]:
            return await self._agenerate_with_retry(
                system_prompt,
                user_prompt,
                endpoints,
                f"Review round {round_idx}",
                client=client,
            )

        return build_prompt, call_llm, acall_llm

    def _review_and_refine(
        self,
        test_cases: list[TestCase],
        endpoints: list[APIEndpoint],
        requirements: list[RequirementItem],
    ) -> list[TestCase]:
        """Run multi-round cross-validation review.

        Round flow (``review_max_rounds`` total):
          - Round 1: secondary model reviews the generated cases.
          - Round 2: primary model reviews round-1 output.
          - Round 3: secondary reviews round-2 output.
          - ... alternating until ``review_max_rounds`` exhausted.

        Round control (alternation, fresh conversation per round, per-round
        fallback on parse failure, original-input fallback when every round
        fails) is implemented once in :class:`ReviewLoop` (plan v2 §4.3).
        """
        build_prompt, call_llm, _ = self._review_hooks(endpoints, requirements)
        result = self._review_loop.run(
            test_cases,
            build_prompt=build_prompt,
            call_llm=call_llm,
            label="test-cases",
        )
        return self._restore_code_identity(test_cases, result.artifact)

    @staticmethod
    def _restore_code_identity(
        original: list[TestCase], reviewed: list[TestCase]
    ) -> list[TestCase]:
        """Carry program-owned quality fields across the review boundary
        (defect ⑦, 2026-09-19 review): the reviewer LLM returns the plain
        legacy JSON contract, so a wholesale replacement zeroed
        binds/executability/scenario_*/covers_obligations. Restoration is
        positional (review preserves order and count via its retention
        guard); unmatchable positions keep the reviewed values as-is."""
        for idx, reviewed_case in enumerate(reviewed):
            if idx >= len(original):
                break
            src = original[idx]
            if not reviewed_case.binds:
                reviewed_case.binds = src.binds
            if not reviewed_case.executability:
                reviewed_case.executability = src.executability
            if not reviewed_case.scenario_operation:
                reviewed_case.scenario_operation = src.scenario_operation
            if not reviewed_case.scenario_scene:
                reviewed_case.scenario_scene = src.scenario_scene
            if not reviewed_case.scenario_variant:
                reviewed_case.scenario_variant = src.scenario_variant
            if not reviewed_case.equivalence_class:
                reviewed_case.equivalence_class = src.equivalence_class
            if not reviewed_case.covers_obligations:
                reviewed_case.covers_obligations = src.covers_obligations
        return reviewed

    # ------------------------------------------------------------------
    # Historical case merging
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_cases(cases: list[TestCase]) -> list[TestCase]:
        """Boundary cleanup for LLM output before save/review.

        Long generations (and truncated-then-salvaged responses) leak
        degenerate stub cases — empty title and/or no ``expected_results`` —
        which are unexecutable noise the reviewer should not have to spend
        output tokens on. Drop them and renumber the survivors sequentially.
        (Scenario-level deduplication stays the reviewer's job.)
        """
        kept: list[TestCase] = []
        dropped = 0
        for tc in cases:
            if not tc.title.strip() or not tc.expected_results:
                dropped += 1
                continue
            kept.append(tc)
        if dropped:
            logger.info("Normalization dropped %d degenerate case(s); %d kept", dropped, len(kept))
        for idx, tc in enumerate(kept, 1):
            tc.id = f"TC-{idx:03d}"
        return kept

    @staticmethod
    def _historical_cases_to_text(cases: list[TestCase] | None) -> str:
        """Render historical cases as a compact text summary for prompt injection.

        Includes id, title, endpoint, test_type and a one-line description so
        the LLM can see what's already covered and avoid regenerating the same
        scenarios.
        """
        if not cases:
            return ""
        lines = [f"Total existing cases: {len(cases)}", ""]
        for tc in cases:
            line = f"- [{tc.id}] {tc.title} | {tc.endpoint.full_path} | {tc.test_type.value}"
            if tc.description:
                # Truncate long descriptions to keep the prompt compact.
                desc = tc.description[:120]
                if len(tc.description) > 120:
                    desc += "..."
                line += f" | {desc}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _merge_historical_cases(
        historical: list[TestCase], new_cases: list[TestCase]
    ) -> list[TestCase]:
        """Merge historical baseline cases with newly generated cases.

        De-duplicates by a fuzzy key (title + endpoint + test_type) so that if
        the LLM regenerated a case that already exists historically, the
        historical version is kept (preserving its original detail) and the
        duplicate new case is dropped.

        Returns the merged list with historical cases first, then net-new cases.
        """
        merged: list[TestCase] = []
        seen_keys: set[str] = set()

        # Historical cases form the baseline.
        for tc in historical:
            key = TestCaseGenerator._case_dedup_key(tc)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(tc)

        # Append only net-new cases (not already in the baseline).
        new_count = 0
        for tc in new_cases:
            key = TestCaseGenerator._case_dedup_key(tc)
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(tc)
                new_count += 1

        logger.info(
            "Merged %d historical + %d net-new = %d total (dedup removed %d duplicates)",
            len(historical),
            new_count,
            len(merged),
            len(new_cases) - new_count,
        )
        return merged

    @staticmethod
    def _case_dedup_key(tc: TestCase) -> str:
        """Build a fuzzy de-duplication key for a test case.

        Uses lowercased title + endpoint + test_type so minor formatting
        differences (case, trailing spaces) don't cause false duplicates.
        """
        return f"{tc.title.strip().lower()}|{tc.endpoint.full_path.lower()}|{tc.test_type.value}"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, output: list[TestCase], output_path: Path) -> Path:
        """Save to JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [self._testcase_to_dict(tc) for tc in output]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d test cases to %s", len(output), output_path)
        return output_path

    def save_csv(self, output: list[TestCase], output_path: Path) -> Path:
        """Save to CSV (UTF-8 BOM for Excel)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for tc in output:
                row = self._testcase_to_dict(tc)
                for key in ("preconditions", "steps", "expected_results", "tags"):
                    row[key] = "; ".join(str(v) for v in row[key])
                row["covers_obligations"] = "; ".join(row["covers_obligations"])
                row["binds"] = json.dumps(row["binds"], ensure_ascii=False)
                row["executability"] = json.dumps(row["executability"], ensure_ascii=False)
                writer.writerow(row)
        logger.info("Saved %d test cases to %s (csv)", len(output), output_path)
        return output_path

    @staticmethod
    def load_historical_cases(path: str | Path) -> list[TestCase]:
        """Load previously generated test cases from a JSON file.

        Supports the JSON array format produced by :meth:`save`. Each element
        must have at least ``id``, ``title``, ``endpoint``, ``test_type`` and
        ``priority``; missing optional fields default to empty lists.

        Args:
            path: Path to the historical test cases JSON file.

        Returns:
            List of :class:`TestCase` objects. Returns an empty list when the
            file cannot be parsed (with a warning logged).
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.warning("Historical test cases file not found: %s", path)
            return []
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse historical cases from %s: %s", path, exc)
            return []
        if not isinstance(data, list):
            logger.warning("Historical cases file %s is not a JSON array", path)
            return []

        cases: list[TestCase] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            tc = TestCaseGenerator._dict_to_testcase(item)
            if tc is not None:
                cases.append(tc)
        logger.info("Loaded %d historical test cases from %s", len(cases), path)
        return cases

    @staticmethod
    def _dict_to_testcase(item: dict[str, Any]) -> TestCase | None:
        """Convert a dict (from JSON) back to a TestCase object.

        Returns ``None`` when required fields are missing or invalid.
        """
        try:
            endpoint_str = str(item.get("endpoint", "N/A N/A"))
            parts = endpoint_str.split(None, 1)
            method = parts[0] if len(parts) >= 1 else "N/A"
            path = parts[1] if len(parts) >= 2 else "N/A"
            endpoint = APIEndpoint(method=method, path=path)

            test_type_str = str(item.get("test_type", "functional")).lower()
            try:
                test_type = TestType(test_type_str)
            except ValueError:
                test_type = TestType.FUNCTIONAL

            priority_str = str(item.get("priority", "medium")).lower()
            try:
                priority = TestPriority(priority_str)
            except ValueError:
                priority = TestPriority.MEDIUM

            return TestCase(
                id=str(item.get("id", "")),
                title=str(item.get("title", "")),
                description=str(item.get("description", "")),
                endpoint=endpoint,
                test_type=test_type,
                priority=priority,
                preconditions=list(item.get("preconditions", [])),
                steps=list(item.get("steps", [])),
                expected_results=list(item.get("expected_results", [])),
                tags=list(item.get("tags", [])),
            )
        except (KeyError, TypeError) as exc:
            logger.warning("Failed to convert dict to TestCase: %s", exc)
            return None

    # ------------------------------------------------------------------
    # JSON parsing & salvage
    # ------------------------------------------------------------------

    def _extract_json(self, raw: str) -> list[Any] | None:
        """Extract JSON array from raw LLM response."""
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            parsed = json.loads(text)
            # Unify the bare-array and {"test_cases": [...]} envelope contracts
            # so both default and JSON-mode output normalize to a flat list.
            return self._unwrap_test_cases(parsed)
        except json.JSONDecodeError:
            pass

        for open_ch, close_ch in (("[", "]"), ("{", "}")):
            start = text.find(open_ch)
            end = text.rfind(close_ch)
            if start != -1 and end > start:
                try:
                    parsed = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
                return self._unwrap_test_cases(parsed)
        return None

    @staticmethod
    def _unwrap_test_cases(parsed: Any) -> list[Any] | None:
        """Normalize a parsed JSON value into a list of test-case dicts.

        Handles both the bare-array contract (default) and the
        ``{"test_cases": [...]}`` envelope produced when JSON mode is enabled.
        A single bare object (legacy/edge case) is wrapped in a list.
        """
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for key in _TEST_CASES_WRAPPER_KEYS:
                if key in parsed and isinstance(parsed[key], list):
                    cases: list[Any] = parsed[key]
                    return cases
            return [parsed]
        return None

    @staticmethod
    def _salvage_truncated_json(raw: str) -> list[Any] | None:
        """Salvage a truncated JSON array by closing brackets."""
        text = raw.strip()
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        bracket_start = text.find("[")
        if bracket_start == -1:
            return None

        depth = 0
        last_complete_obj_end = -1
        in_string = False
        escape = False

        for i in range(bracket_start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_complete_obj_end = i
            elif ch == "]" and depth == 0:
                break

        if last_complete_obj_end == -1:
            return None

        salvaged = text[: last_complete_obj_end + 1] + "]"
        salvaged = re.sub(r",\s*\]$", "]", salvaged)
        try:
            parsed = json.loads(salvaged)
            if isinstance(parsed, list) and len(parsed) > 0:
                logger.info("Salvaged %d cases from truncated response", len(parsed))
                return parsed
        except json.JSONDecodeError:
            pass
        return None

    @staticmethod
    def _dump_debug_response(raw: str) -> None:
        """Persist unparseable response for debugging."""
        try:
            debug_path = Path("output/.debug_last_llm_response.txt")
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text(raw, encoding="utf-8")
            logger.info("Raw response saved to %s", debug_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def _to_test_cases(self, items: list[Any], endpoints: list[APIEndpoint]) -> list[TestCase]:
        """Convert parsed JSON items into TestCase objects."""
        # T8 wiring 1/3: per-batch identity dedup (kept/removed both logged).
        items, removed = dedup_cases([it for it in items if isinstance(it, dict)])
        if removed:
            self._dedup_removed.extend(removed)
            for entry in removed:
                logger.info(
                    "Scenario dedup: removed %s (key=%s) kept %s [%s]",
                    entry["removed_id"],
                    entry["key"],
                    entry["kept_id"],
                    entry["reason"],
                )
        # T11a: deterministic normalization (never calls the LLM).
        items = [normalize_case(it) if isinstance(it, dict) else it for it in items]
        endpoint_map = {ep.full_path: ep for ep in endpoints}
        # Fallback endpoint for requirement-only cases (no API spec)
        fallback_ep = endpoints[0] if endpoints else APIEndpoint(method="N/A", path="N/A")

        test_cases: list[TestCase] = []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            ep_key = item.get("endpoint", "")
            endpoint = endpoint_map.get(ep_key, fallback_ep)
            try:
                test_type = TestType(item.get("test_type", "functional"))
            except ValueError:
                test_type = TestType.FUNCTIONAL
            try:
                priority = TestPriority(item.get("priority", "medium"))
            except ValueError:
                priority = TestPriority.MEDIUM

            case = TestCase(
                id=str(item.get("id", f"TC-{idx:03d}")),
                title=item.get("title", ""),
                description=item.get("description", ""),
                endpoint=endpoint,
                test_type=test_type,
                priority=priority,
                preconditions=item.get("preconditions", []),
                steps=item.get("steps", []),
                expected_results=item.get("expected_results", []),
                tags=item.get("tags", []),
                binds=item.get("binds") or {},
                scenario_operation=str(item.get("scenario_operation", "") or ""),
                scenario_scene=str(item.get("scenario_scene", "") or ""),
                scenario_variant=str(item.get("scenario_variant", "") or ""),
                equivalence_class=str(item.get("equivalence_class", "") or ""),
                covers_obligations=[str(c) for c in item.get("covers_obligations") or []],
            )
            test_cases.append(case)
        self._session_case_count += len(test_cases)
        # T11b: semantic validation (structural issues already normalized).
        known: set[str] | None = None
        if self._obligation_registry is not None:
            known = {st.obligation.id for st in self._obligation_registry.states()}
        for case in test_cases:
            raw = self._testcase_to_dict(case)
            gaps = semantic_validation(raw, known_obligations=known)
            if gaps and needs_reask(gaps):
                logger.warning("Case %s semantic gaps (re-ask eligible): %s", case.id, gaps)
            elif gaps:
                logger.info("Case %s semantic notes: %s", case.id, gaps)
        return test_cases

    @staticmethod
    def _testcase_to_dict(tc: TestCase) -> dict[str, Any]:
        """Convert TestCase to serializable dict."""
        return {
            "id": tc.id,
            "title": tc.title,
            "description": tc.description,
            "endpoint": tc.endpoint.full_path,
            "test_type": tc.test_type.value,
            "scenario_operation": tc.scenario_operation,
            "scenario_scene": tc.scenario_scene,
            "scenario_variant": tc.scenario_variant,
            "equivalence_class": tc.equivalence_class,
            "covers_obligations": tc.covers_obligations,
            "binds": tc.binds,
            "executability": tc.executability,
            "priority": tc.priority.value,
            "preconditions": tc.preconditions,
            "steps": tc.steps,
            "expected_results": tc.expected_results,
            "tags": tc.tags,
        }
