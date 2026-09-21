"""The case quality line (fix-plan T1~T13) as ONE implementation.

Why this module exists: every quality capability (T1 raw audit + reconciliation,
T4/T9 conflict adjudication, T5/T11 normalization + semantic validation, T6
binding, T7 obligation floor + session budget, T8 scenario dedup, T10
executability grading) was built into the LEGACY host adapter
``generators/testcase_generator.py``. The task-package chain produces artifacts
without it, so a naive migration onto the pipeline would silently ship hollow
columns (``executability={}``, no dedup, no obligation accounting) — "tests
green, semantics changed".

Extracting the bodies here means the legacy generator keeps only a thin
delegation and the pipeline calls the same functions: old-chain vs new-chain
equivalence holds BY CONSTRUCTION, and the deletion gate (B6b.5) removes a
wrapper rather than a capability.

State discipline: one instance per run (the obligation registry, dedup ledger
and raw dumper are per-session by nature — see defect ⑨), and no settings
singleton reads: everything comes in through the constructor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from testagent.config.models import APIEndpoint, RequirementItem, TestCase, TestPriority, TestType
from testagent.engine.raw_dump import RawResponseDumper
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
from testagent.pipeline.scenario import covered_identities, dedup_cases, render_dedup_report
from testagent.pipeline.testcase_adapter import renumber, testcase_to_full_dict

logger = logging.getLogger(__name__)

__all__ = ["QualityPass", "QualityRunConfig"]


class QualityRunConfig:
    """The knobs a quality run needs, resolved by the composition root.

    Deliberately a plain object (not ``Settings``): the pipeline must be able
    to run with an injected, overridable configuration, and tests must be able
    to build one without touching ``.env``.
    """

    __slots__ = (
        "audit_dump_dir",
        "audit_dump_enabled",
        "cases_budget",
        "conflict_policy",
        "default_expected_per_endpoint",
    )

    @classmethod
    def from_settings(
        cls, settings: Any, *, default_expected_per_endpoint: int
    ) -> QualityRunConfig:
        """Resolve the run knobs from an INJECTED Settings.

        Missing attributes fall back to the inert values (budget 0 = no cap,
        auditing off) rather than crashing, so a partial double in a test stays
        usable; the real Settings always carries all four (single holder in
        ``config/settings.py``).
        """
        return cls(
            default_expected_per_endpoint=default_expected_per_endpoint,
            cases_budget=int(getattr(settings, "cases_budget", 0) or 0),
            conflict_policy=str(getattr(settings, "conflict_policy", "strict")),
            audit_dump_enabled=bool(getattr(settings, "audit_dump_enabled", False)),
            audit_dump_dir=str(getattr(settings, "output_dir", "") or ""),
        )

    def __init__(
        self,
        *,
        default_expected_per_endpoint: int,
        cases_budget: int = 0,
        conflict_policy: str = "strict",
        audit_dump_enabled: bool = False,
        audit_dump_dir: str = "",
    ) -> None:
        self.default_expected_per_endpoint = default_expected_per_endpoint
        self.cases_budget = cases_budget
        self.conflict_policy = conflict_policy
        self.audit_dump_enabled = audit_dump_enabled
        self.audit_dump_dir = audit_dump_dir


class QualityPass:
    """Per-run quality line: pre-flight context, per-batch conversion, and the
    post-merge passes in the order the legacy chain ran them."""

    def __init__(self, config: QualityRunConfig, *, session_id: str = "") -> None:
        self._config = config
        self._session_id = session_id
        self._raw_dumper: RawResponseDumper | None = None
        self._conflict_table = ""
        self._conflict_findings: list[Finding] = []
        self._obligation_registry: ObligationRegistry | None = None
        self._session_case_count = 0
        self._budget_trim_report = ""
        self._dedup_removed: list[dict[str, Any]] = []
        self._dedup_missing: list[str] = []

    # -- session pre/post ---------------------------------------------------

    def start_session(
        self, requirements: list[RequirementItem], endpoints: list[APIEndpoint]
    ) -> None:
        """Open the raw audit, adjudicate spec/requirement conflicts, and
        register the obligation ledger (T1 + T4/T9 + T5/T6/T7)."""
        self._start_raw_audit()
        self._start_conflict_context(requirements, endpoints)
        self._start_obligation_context(requirements, endpoints)

    def finish_session(self, artifacts: list[TestCase]) -> None:
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

    @property
    def conflict_table(self) -> str:
        """Authoritative value table for prompt injection ('' when silent)."""
        return self._conflict_table

    @property
    def obligation_registry(self) -> ObligationRegistry | None:
        return self._obligation_registry

    def emit_raw_record(self, record: dict[str, Any]) -> None:
        """Engine ``raw_sink`` adapter: route records to the active dump."""
        if self._raw_dumper is not None:
            self._raw_dumper.sink(record)

    def _start_raw_audit(self) -> None:
        base = self._config.audit_dump_dir
        if self._config.audit_dump_enabled and base and self._session_id:
            self._raw_dumper = RawResponseDumper(Path(base) / "sessions", self._session_id)
        else:
            self._raw_dumper = None

    def _start_conflict_context(
        self, requirements: list[RequirementItem], endpoints: list[APIEndpoint]
    ) -> None:
        """T4/T9: deterministic consistency checks once per session, adjudicated
        by ``CONFLICT_POLICY``; the table is shared with generation AND review
        prompts and the gap report lands in the session directory."""
        try:
            policy = ConflictPolicy(self._config.conflict_policy)
        except ValueError:
            raise ValueError(
                f"invalid CONFLICT_POLICY: {self._config.conflict_policy!r} "
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

    def _start_obligation_context(
        self, requirements: list[RequirementItem], endpoints: list[APIEndpoint]
    ) -> None:
        """T5/T7: register requirement-AC and spec obligations for this session;
        the uncovered ones drive the per-endpoint quota floor (T6 binding gives
        requirement obligations their real endpoints — without them the floor
        was permanently zero)."""
        registry = ObligationRegistry()
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

    # -- engine hook --------------------------------------------------------

    def expected_for(self, scope: list[str]) -> dict[str, int]:
        """GenericHooks.expected_for: obligation floor + session budget cap.

        Floor per endpoint = uncovered obligations bound to it (each needs at
        least one case); endpoints without obligations fall back to the policy
        default. The session cap (``CASES_BUDGET``) clamps the total: ``0``
        disables the cap entirely.
        """
        default = self._config.default_expected_per_endpoint
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
        budget = self._config.cases_budget
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

    # -- per-batch conversion ----------------------------------------------

    def to_test_cases(self, items: list[Any], endpoints: list[APIEndpoint]) -> list[TestCase]:
        """Convert parsed JSON items into TestCase objects (T8 per-batch dedup,
        T11a normalization, endpoint scoping, T11b semantic validation)."""
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

            test_cases.append(
                TestCase(
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
                    # path_id / source_stage are deliberately NOT read from the
                    # model: they are program-stamped (v15 §6.1), so a
                    # self-reported one must not survive this boundary. The
                    # links pass records the attempt from the raw items.
                )
            )
        self._session_case_count += len(test_cases)
        # Degenerate-stub cleanup (empty title / no expected results). The
        # legacy chain ran it once on the whole pre-merge list; per batch the
        # FILTER is equivalent, but the renumber is not: unit ids are prompt
        # material for the next phase ("already covered" lists them), and the
        # legacy chain showed the model's own ids there. The H1 merge renumbers
        # globally afterwards either way, and user-provided historical cases
        # never pass through here, so the baseline cannot be filtered away.
        test_cases = self.normalize_cases(test_cases, renumber_ids=False)
        known: set[str] | None = None
        if self._obligation_registry is not None:
            known = {st.obligation.id for st in self._obligation_registry.states()}
        for case in test_cases:
            gaps = semantic_validation(testcase_to_full_dict(case), known_obligations=known)
            if gaps and needs_reask(gaps):
                logger.warning("Case %s semantic gaps (re-ask eligible): %s", case.id, gaps)
            elif gaps:
                logger.info("Case %s semantic notes: %s", case.id, gaps)
        return test_cases

    # -- post-merge passes --------------------------------------------------

    @staticmethod
    def normalize_cases(cases: list[TestCase], *, renumber_ids: bool = True) -> list[TestCase]:
        """Boundary cleanup before save/review: drop degenerate stub cases
        (empty title and/or no expected results) and renumber survivors.

        Scenario-level deduplication stays a separate pass (``global_dedup``).
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
        if renumber_ids:
            for idx, tc in enumerate(kept, 1):
                tc.id = f"TC-{idx:03d}"
        return kept

    @staticmethod
    def covered_identities(cases: list[TestCase]) -> list[dict[str, str]]:
        return covered_identities(cases)

    def global_dedup(self, cases: list[TestCase]) -> list[TestCase]:
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

    def run_executability_gates(self, cases: list[TestCase], endpoints: list[APIEndpoint]) -> None:
        """T10: Gate-A/Gate-B after generation; grades land on the artifacts and
        metrics in the session report."""
        raw_cases = [testcase_to_full_dict(tc) for tc in cases]
        metrics = placeholder_closure_metrics(raw_cases, endpoints)
        for tc, raw in zip(cases, raw_cases, strict=False):
            tc.executability = grade_case(raw, endpoints).to_meta()
        grades: dict[str, int] = {}
        for tc in cases:
            grade = str(tc.executability.get("grade", "DRAFT"))
            grades[grade] = grades.get(grade, 0) + 1
        logger.info("Executability grades: %s | metrics: %s", grades, metrics)
        if self._raw_dumper is not None:
            lines = ["# Executability report", "", f"Grades: {grades}", f"Metrics: {metrics}", ""]
            for tc in cases:
                if tc.executability.get("grade") != "INTEGRATION":
                    lines.append(f"- {tc.id}: {tc.executability}")
            self._raw_dumper.write_report("executability_report.md", "\n".join(lines) + "\n")

    def account_obligations(self, cases: list[TestCase]) -> None:
        """T5 accounting with FINAL (renumbered) case ids: the model emits a
        literal "TC-XXX" id, so id-keyed accounting at conversion time never
        matched the real ids the ledger should carry."""
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

    def enforce_budget(self, cases: list[TestCase]) -> list[TestCase]:
        """T7: session cap with per-item accounting — obligation-covering cases
        first, then generation order; every trim is logged and reported."""
        budget = self._config.cases_budget
        if budget <= 0 or len(cases) <= budget:
            return cases
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

    # -- the whole post-merge sequence, in legacy order ----------------------

    def post_merge(self, cases: list[TestCase], endpoints: list[APIEndpoint]) -> list[TestCase]:
        """The passes the legacy chain ran on the merged, pre-review artifact.

        Order is load-bearing (T8 dedup before the T7 cap, grades before the
        final renumber, accounting with the FINAL ids) — see
        ``TestCaseGenerator.generate`` for the original sequence.
        """
        cases = self.global_dedup(cases)
        self._dedup_missing = [
            tc.id
            for tc in cases
            if not (tc.scenario_operation and tc.scenario_scene and tc.scenario_variant)
        ]
        cases = self.enforce_budget(cases)
        self.run_executability_gates(cases, endpoints)
        renumber(cases)
        self.account_obligations(cases)
        return cases

    def after_review(self, cases: list[TestCase]) -> list[TestCase]:
        """T13: review output is not trusted blind — the same degenerate-stub
        cleanup (with renumber) runs on whatever the reviewer returned."""
        return self.normalize_cases(cases)
