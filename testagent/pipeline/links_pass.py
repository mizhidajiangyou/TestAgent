"""LINK-S5b / S6b + T12b: the per-run links pass on the task-package chain.

What gets wired where (v15 §5-§7, T12b):

- ``start``      : R6 edges from requirement prose → graph → path planning →
                   context builder, all ONCE per run (the L0 index and the
                   planning result are immutable inputs to every later step).
- ``context_block`` : the L0 index / L1 neighbours appended to phase-1/2 unit
                   prompts (T12b: batching material, not a template rewrite).
- ``clusters_for`` : L3a lifecycle clusters that replace blind endpoint
                   batching for phase 2 (v15 §5.3).
- ``l3b_bundles``  : one unit per selected path contract, consumed by the
                   ``l3b`` stage (``split: per_input`` over this list).
- ``stamp``      : program identity fields (``path_id`` / ``source_stage``) —
                   the model never writes them (v15 §6.1 "keep path_id out of
                   your output").
- ``run_gates``  : Gate 1/2/3 over the merged artifact + the S7 metrics.

Layering: no settings-singleton reads (config arrives injected), no bare
``getattr`` on shared columns (``links_fields`` accessors), and everything is
per-run state on this object — never on a shared provider (defect ⑨).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from testagent.pipeline.context_builder import ContextBuilder, assemble_l3b_prompts
from testagent.pipeline.linkcheck import DECLARED_LEGEND, check_contract_cases
from testagent.pipeline.links_fields import case_view_from_dict
from testagent.pipeline.links_graph import build_graph
from testagent.pipeline.links_prompt_contract import (
    BINDS_V2_CONTRACT,
    PATH_CONTRACT_INSTRUCTION,
)
from testagent.pipeline.links_r6 import extract_r6_edges
from testagent.pipeline.pathplanner import plan_paths

if TYPE_CHECKING:
    from testagent.config.models import APIEndpoint, RequirementItem

logger = logging.getLogger(__name__)

__all__ = ["LinksPass", "LinksRunConfig"]

#: Manifest stage name -> the shared ``source_stage`` value domain (v15 §3.1).
#: A stage rename must not silently widen the enum the artifact schema pins, so
#: stages outside the pipeline's generation phases carry "" = "not a phase unit".
STAGE_SOURCE: dict[str, str] = {"phase1": "phase1", "phase2_api": "phase2", "l3b": "l3b"}

#: Sidecar format version (v15 §8.3). A reader must refuse an unknown version
#: rather than guess at fields that may mean something else.
REPORT_VERSION = 1


@dataclass
class LinksRunConfig:
    """The 11 ``LINKS_*`` knobs, resolved by the composition root."""

    enabled: bool = True
    prose_enabled: bool = True
    l0_max_chars: int = 8000
    max_neighbors: int = 4
    neighbor_chars: int = 1200
    cluster_size: int = 6
    max_planned: int = 8
    max_hops: int = 3
    l3b_max_calls: int = 3
    seeding: str = "priority"
    r6_min_score: float = 1.0

    @classmethod
    def from_settings(cls, settings: Any) -> LinksRunConfig:
        return cls(
            enabled=bool(getattr(settings, "links_enabled", True)),
            prose_enabled=bool(getattr(settings, "links_prose_enabled", True)),
            l0_max_chars=int(getattr(settings, "links_l0_max_chars", 8000)),
            max_neighbors=int(getattr(settings, "links_max_neighbors", 4)),
            neighbor_chars=int(getattr(settings, "links_neighbor_chars", 1200)),
            cluster_size=int(getattr(settings, "links_cluster_size", 6)),
            max_planned=int(getattr(settings, "links_max_planned_paths", 8)),
            max_hops=int(getattr(settings, "links_max_hops", 3)),
            l3b_max_calls=int(getattr(settings, "links_l3b_max_calls", 3)),
            seeding=str(getattr(settings, "links_seeding", "priority")),
            r6_min_score=float(getattr(settings, "links_r6_min_score", 1.0)),
        )


@dataclass
class L3bBundle:
    """One L3b unit: the path contract plus the prompt material for it."""

    path_id: str
    static_class: str
    endpoints: list[APIEndpoint]
    block: str

    def as_dict(self) -> dict[str, Any]:
        """Template view (Jinja reads plain keys; the endpoint list stays
        objects so the same renderers work as for phase 2)."""
        return {
            "path_id": self.path_id,
            "static_class": self.static_class,
            "endpoints": self.endpoints,
            "endpoints_text": "\n".join(f"- {ep.method} {ep.path}" for ep in self.endpoints),
            "contract_block": self.block,
            "path_contract_instruction": PATH_CONTRACT_INSTRUCTION,
            "binds_contract": BINDS_V2_CONTRACT,
        }


class LinksPass:
    """Per-run links state + the four touch points the pipeline calls."""

    def __init__(
        self,
        config: LinksRunConfig,
        *,
        signature_fn: Any = None,
        session_id: str = "",
    ) -> None:
        self._config = config
        self._signature_fn = signature_fn
        self._session_id = session_id
        self.started = False
        self.l0 = ""
        self.planning: dict[str, Any] = {}
        self.bundles: list[L3bBundle] = []
        self.abstentions: list[dict[str, str]] = []
        #: Model-written identity columns that the program overwrote (v15 §6.1).
        self.forgeries: list[dict[str, str]] = []
        #: Identity values an external baseline carried, before they were
        #: cleared for this run (v15 §7.3). Audit only, never provenance.
        self.history_audits: list[dict[str, Any]] = []
        self.r6_edges = 0
        self.gate_report: dict[str, Any] = {}
        self._builder: ContextBuilder | None = None
        self._endpoint_index: dict[str, APIEndpoint] = {}

    # -- preparation -------------------------------------------------------

    def start(self, requirements: list[RequirementItem], endpoints: list[APIEndpoint]) -> None:
        """Build graph → plan → context once; never re-plan per unit."""
        if self._config.l3b_max_calls < 1:
            raise ValueError("LINKS_L3B_MAX_CALLS must be >= 1 (degrade to 1, never 0)")
        self._endpoint_index = {f"{ep.method} {ep.path}": ep for ep in endpoints}
        relations: list[tuple[str, str, str]] = []
        edges: list[Any] = []
        if self._config.prose_enabled and requirements:
            text = "\n\n".join(
                "\n".join(
                    filter(
                        None,
                        [
                            req.id or "",
                            req.description or "",
                            *[f"- {ac}" for ac in req.acceptance_criteria or []],
                        ],
                    )
                )
                for req in requirements
            )
            edges, self.abstentions = extract_r6_edges(text, min_score=self._config.r6_min_score)
            self.r6_edges = len(edges)
            relations = [(e.source, e.target, e.kind) for e in edges]

        graph = build_graph(endpoints, r6_edges=tuple(edges))
        self.planning = plan_paths(
            graph,
            endpoints,
            max_hops=self._config.max_hops,
            max_planned=self._config.max_planned,
            l3b_budget=self._config.l3b_max_calls,
            seeding=self._config.seeding,
        )
        self._builder = ContextBuilder(
            endpoints,
            relations=relations,
            max_chars=self._config.l0_max_chars,
            max_neighbors=self._config.max_neighbors,
            l1_chars=self._config.neighbor_chars,
            max_cluster=self._config.cluster_size,
            signature_fn=self._signature_fn,
        )
        self.l0 = self._builder.build_l0()
        self.bundles = self._assemble_bundles()
        self.started = True
        logger.info(
            "[links] %s paths=%d candidate=%d selected=%d r6=%d abstain=%d l3b=%d",
            self._session_id or "-",
            len(self.planning.get("planned", [])),
            len(self.planning.get("candidate", [])),
            len(self.planning.get("selected", [])),
            self.r6_edges,
            len(self.abstentions),
            len(self.bundles),
        )

    def _assemble_bundles(self) -> list[L3bBundle]:
        if self._builder is None:
            return []
        out: list[L3bBundle] = []
        for item in assemble_l3b_prompts(self.planning, self._builder):
            contract = next(
                (c for c in self.planning["selected"] if c.path_id == item["path_id"]), None
            )
            keys = list(contract.endpoints) if contract is not None else []
            eps = [
                self._endpoint_index[k] for k in dict.fromkeys(keys) if k in self._endpoint_index
            ]
            out.append(
                L3bBundle(
                    path_id=item["path_id"],
                    static_class=item["static_class"],
                    endpoints=eps,
                    block=(
                        f"[path_id={item['path_id']} class={item['static_class']}]\n"
                        f"{item['l0']}\n\n{item['l2']}\n\n"
                        f"{PATH_CONTRACT_INSTRUCTION}\n{BINDS_V2_CONTRACT}"
                    ),
                )
            )
        return out[: self._config.l3b_max_calls]

    # -- touch points ------------------------------------------------------

    def context_block(self, stage: str, unit_ctx: dict[str, Any]) -> str:
        """L0 (+ L1 neighbours of this unit's batch) for a phase-1/2 unit.

        Returns "" when there is nothing to add, so the caller can append
        unconditionally and links-off prompts stay byte-identical.
        """
        if self._builder is None or not self.l0:
            return ""
        batch = unit_ctx.get("_unit_batch") or []
        keys = [f"{ep.method} {ep.path}" for ep in batch if hasattr(ep, "method")]
        if not keys:
            single: Any = unit_ctx.get("_unit_item")
            if hasattr(single, "method"):
                keys = [f"{single.method} {single.path}"]
        parts = [f"[L0 endpoint index]\n{self.l0}"]
        if keys:
            l1 = self._builder.build_l1(keys)
            if l1.strip():
                parts.append(f"[L1 related endpoints]\n{l1}")
        return "\n\n".join(parts)

    def clusters_for(self, items: list[Any]) -> list[list[Any]]:
        """L3a lifecycle clusters replacing blind batches (v15 §5.3).

        Anything the clusters do not cover (endpoints outside every lifecycle)
        is appended in its own deterministic group, so no endpoint is dropped.
        """
        if self._builder is None:
            return [items]
        index = {f"{ep.method} {ep.path}": ep for ep in items}
        used: set[str] = set()
        clusters: list[list[Any]] = []
        for group in self._builder.split_clusters():
            members = [index[k] for k in group if k in index and k not in used]
            for k in group:
                used.add(k)
            if members:
                clusters.append(members)
        leftovers = [ep for key, ep in index.items() if key not in used]
        for start in range(0, len(leftovers), self._config.cluster_size):
            clusters.append(leftovers[start : start + self._config.cluster_size])
        return clusters or [items]

    def l3b_units(self) -> list[dict[str, Any]]:
        return [bundle.as_dict() for bundle in self.bundles]

    def note_forgeries(self, raw_items: list[Any], stage: str) -> None:
        """Record identity columns the MODEL tried to write (v15 §6.1).

        Reading happens on the raw unit output because that is the only place
        the claim exists: the quality conversion already refuses to copy the
        columns onto a case, so by stamp time the artifact looks clean and the
        attempt would go unrecorded.
        """
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            forged = {k: item[k] for k in ("path_id", "source_stage") if item.get(k)}
            if not forged:
                continue
            self.forgeries.append(
                {
                    "stage": stage,
                    "case_id": str(item.get("id", "")),
                    **{k: str(v) for k, v in forged.items()},
                }
            )
            logger.warning(
                "[links] %s model-forged identity on %s (%s); ignored",
                self._session_id or "-",
                item.get("id", "?"),
                ", ".join(f"{k}={v!r}" for k, v in forged.items()),
            )

    def stamp(self, cases: list[dict[str, Any]], *, stage: str, path_id: str = "") -> None:
        """Program-owned identity fields, written from the unit's sidecar."""
        for case in cases:
            if not isinstance(case, dict):
                continue
            case["path_id"] = path_id
            case["source_stage"] = STAGE_SOURCE.get(stage, "")
            ex = case.get("executability")
            if isinstance(ex, dict):
                # v15 §3.1 called this column a string; the shipped model, the
                # tasks/testcase JSON schema and the CSV writer all use the T10
                # dict. The links class therefore lives UNDER that dict.
                ex["links"] = {"static_class": self._static_class_for(path_id)}
            elif ex in (None, ""):
                case["executability"] = {"links": {"static_class": self._static_class_for(path_id)}}

    def normalize_history(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """v15 §7.3: strip this-run identity claims from an external baseline.

        Copies rather than mutates — the caller's file may be re-read by the
        session writer, and the original values stay in ``history_audits``.
        """
        out: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                out.append(item)
                continue
            audit = {
                column: item[column]
                for column in ("path_id", "source_stage", "executability")
                if item.get(column)
            }
            if audit:
                self.history_audits.append({"id": str(item.get("id", "")), **audit})
            out.append({**item, "path_id": "", "source_stage": "", "executability": {}})
        return out

    def _static_class_for(self, path_id: str) -> str:
        for bundle in self.bundles:
            if bundle.path_id == path_id:
                return bundle.static_class
        return ""

    def run_gates(
        self, cases: list[dict[str, Any]], endpoints: list[APIEndpoint], attempted: list[str]
    ) -> dict[str, Any]:
        """Gate 1/2/3 over the merged artifact (never a silent pass on empty)."""
        contracts = self.planning.get("candidate", [])
        # Gate identities come from the artifact id, so an id-less case cannot
        # be checked. It is counted rather than dropped quietly.
        views: list[Any] = []
        skipped = 0
        for case in cases:
            if not isinstance(case, dict) or not str(case.get("id", "") or ""):
                skipped += 1
                continue
            views.append(case_view_from_dict(case))
        outcome_report = check_contract_cases(
            contracts, views, endpoints, attempted=tuple(attempted)
        )
        self.gate_report = outcome_report.metrics()
        self.gate_report["cases_without_id"] = skipped
        self.gate_report["model_forged_identity"] = len(self.forgeries)
        self.gate_report["history_identity_cleared"] = len(self.history_audits)
        self.gate_report.setdefault("planned_paths", len(self.planning.get("planned", [])))
        self.gate_report.setdefault("selected_paths", len(self.planning.get("selected", [])))
        self.gate_report.setdefault("r6_edges", self.r6_edges)
        self.gate_report.setdefault("r6_abstentions", len(self.abstentions))
        # S7: rejected_rate must be readable per source_stage (v15 §8.1) — a
        # single run-wide rate hides whether the L3b path is the failing one.
        stage_of = {view.case_id: (view.source_stage or "unspecified") for view in views}
        by_stage: dict[str, dict[str, Any]] = {}
        for case_id, outcome in outcome_report.outcomes.items():
            bucket = by_stage.setdefault(stage_of.get(case_id, "unspecified"), {})
            grade = str(outcome.grade)
            bucket[grade] = int(bucket.get(grade, 0)) + 1
        self.gate_report["grades_by_source_stage"] = {
            stage: {
                **counts,
                "total": sum(int(v) for v in counts.values()),
                "rejected_rate": round(
                    int(counts.get("REJECTED", 0)) / max(1, sum(int(v) for v in counts.values())), 4
                ),
            }
            for stage, counts in by_stage.items()
        }
        return self.gate_report

    def attempted_keys(self) -> list[str]:
        return [c.path_id for c in self.planning.get("candidate", [])]

    def config_digest(self) -> str:
        """Hash of the links knobs this run planned and graded under."""
        payload = json.dumps(self._config.__dict__, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def run_document(
        self,
        *,
        cases: list[dict[str, Any]],
        pre_review_hash: str,
        final_hash: str,
        input_digest: str,
        config_digest: str,
        code_version: str,
        code_sha: str,
    ) -> dict[str, Any]:
        """The links sidecar (v15 §8.3): everything a later ``links-check``
        needs to replay this run WITHOUT re-planning it.

        Re-planning on replay is the trap this document exists to close: the
        planner is deterministic today, but a changed threshold or a new edge
        rule would silently move the denominators and a coverage number would
        stop meaning "what this run was measured against".
        """
        candidates = self.planning.get("candidate", [])
        selected_ids = {c.path_id for c in self.planning.get("selected", [])}
        return {
            "report_version": REPORT_VERSION,
            "run_id": self._session_id,
            # Provenance, not verification: a reader must know WHICH code said
            # these numbers, and an empty sha means "not recorded" (warn) rather
            # than "verified signature" (v15 §8.3 says local consistency only).
            "code_version": code_version,
            "code_sha": code_sha,
            "input_digest": input_digest,
            "config_digest": config_digest,
            "candidate_contracts": [
                {
                    "path_id": c.path_id,
                    # The canonical payload IS the contract: a replay rebuilds
                    # from it and refuses if it no longer hashes to path_id.
                    "payload": dict(c.payload),
                    "canonical_sha256": hashlib.sha256(c.canonical_bytes).hexdigest(),
                    "endpoints": list(c.endpoints),
                    # Full pair records (pair_key / kind / required_binding):
                    # Gate 2 reads them as-is, so the replay needs no re-derivation.
                    "required_pairs": [dict(p) for p in c.required_pairs],
                    "priority": c.priority,
                    "static_class": c.static_class,
                    "selected": c.path_id in selected_ids,
                }
                for c in candidates
            ],
            "pool_counts": {
                "planned": len(self.planning.get("planned", [])),
                "candidate": len(candidates),
                "selected": len(self.planning.get("selected", [])),
                "l3b_units": len(self.bundles),
                "r6_edges": self.r6_edges,
                "r6_abstentions": len(self.abstentions),
            },
            "attempt_ledger": {
                # Attempted = this run actually dispatched a unit for the path,
                # not merely that the path exists in the candidate pool.
                c.path_id: c.path_id in {bundle.path_id for bundle in self.bundles}
                for c in candidates
            },
            "case_provenance": [
                {
                    "case_id": str(case.get("id", "")),
                    "path_id": str(case.get("path_id", "") or ""),
                    "source_stage": str(case.get("source_stage", "") or ""),
                }
                for case in cases
                if isinstance(case, dict)
            ],
            "artifact_hashes": {"pre_review": pre_review_hash, "final": final_hash},
            "gates": dict(self.gate_report),
            "legend": DECLARED_LEGEND,
            "model_forged_identity": list(self.forgeries),
            "history_identity_cleared": list(self.history_audits),
        }
