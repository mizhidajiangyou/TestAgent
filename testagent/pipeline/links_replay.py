"""Offline replay of a links run report (LINK-S7, v15 §8.3).

``testagent links-check`` answers one question about an artifact file: *do the
cases actually fulfil the contracts, and do the provenance columns tell the
truth?* Two modes, because the honest answer differs:

- **with the run sidecar**: the contracts are the ones the run planned. The
  planner is NOT re-run — re-planning would move the denominators whenever a
  threshold changes, and then "62% pair coverage" would silently describe a
  different run than the one that produced the file.
- **without it**: only the facts visible in the text can be checked (closure,
  bind legality). Selection is reported as ``unknown`` and contract coverage as
  ``n/a``; an existing ``case.path_id`` is never promoted to "this run selected
  that path", because a model-written or stale id is not provenance.

Exit codes (documented, not decoration): ``0`` clean, ``1`` violations or
unfulfilled candidates, ``2`` the inputs/contract/hashes do not add up.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from testagent.pipeline.linkcheck import gate1_closure, gate3_binds
from testagent.pipeline.links_fields import CaseView, case_view_from_dict
from testagent.pipeline.pathplanner import (
    PathContract,
    canonical_payload_bytes,
    contract_from_payload,
    path_id_of,
)

__all__ = ["CheckReport", "ReportError", "load_document", "observe", "replay"]

REPORT_VERSION = 1


class ReportError(ValueError):
    """The document or the artifact it is checked against does not add up."""


@dataclass
class CheckReport:
    """Outcome of one offline check, in the shape the CLI prints verbatim."""

    mode: str
    run_id: str = ""
    errors: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    per_case: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        if self.errors:
            return 2
        return 1 if self.findings else 0


def hash_document(value: Any) -> str:
    """The same content hash the pipeline sidecar records."""
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def load_document(raw: str | dict[str, Any]) -> dict[str, Any]:
    doc = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if doc.get("report_version") != REPORT_VERSION:
        raise ReportError(f"unsupported report_version {doc.get('report_version')!r}")
    for key in ("run_id", "candidate_contracts", "artifact_hashes", "gates"):
        if key not in doc:
            raise ReportError(f"sidecar is missing {key!r}")
    return doc


def _views(cases: list[dict[str, Any]]) -> list[CaseView]:
    views: list[CaseView] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ReportError("artifact items must be objects")
        case_id = str(case.get("id", "") or "")
        if not case_id:
            raise ReportError("every artifact case needs a unique id (empty id found)")
        if case_id in seen:
            raise ReportError(f"duplicate case id {case_id!r} in the artifact")
        seen.add(case_id)
        views.append(case_view_from_dict(case))
    return views


def _contract_from_document(entry: dict[str, Any]) -> PathContract:
    """Rebuild the contract EXACTLY as the run saw it, or refuse.

    ``path_id`` is derived from the canonical payload, so a document whose
    payload was edited after the run fails here instead of silently changing
    what the coverage number measures.
    """
    payload = entry.get("payload")
    if not isinstance(payload, dict):
        raise ReportError(f"contract entry without payload: {entry.get('path_id')!r}")
    canonical = canonical_payload_bytes(payload)
    if hashlib.sha256(canonical).hexdigest() != str(entry.get("canonical_sha256", "")):
        raise ReportError(f"canonical payload changed for {entry.get('path_id')!r}")
    derived = path_id_of(payload)
    if derived != str(entry.get("path_id", "")):
        raise ReportError(
            f"path_id {entry.get('path_id')!r} does not match its payload ({derived})"
        )
    return contract_from_payload(
        payload,
        required_pairs=[dict(p) for p in entry.get("required_pairs") or []],
        priority=int(entry.get("priority", 0)),
        static_class=str(entry.get("static_class", "")),
    )


def replay(
    document: dict[str, Any], artifact: list[dict[str, Any]], endpoints: list[Any]
) -> CheckReport:
    """Check an artifact against the contracts the run actually planned."""
    report = CheckReport(mode="replay", run_id=str(document.get("run_id", "")))
    recorded = document.get("artifact_hashes") or {}
    final_hash = str(recorded.get("final", ""))
    if final_hash and final_hash != hash_document(artifact):
        report.errors.append(
            "artifact does not match the run's recorded final hash "
            f"({final_hash[:12]}… expected) — checking the wrong file?"
        )
        return report

    try:
        views = _views(artifact)
    except ReportError as exc:
        report.errors.append(str(exc))
        return report

    candidates = document.get("candidate_contracts") or []
    try:
        contracts = [_contract_from_document(entry) for entry in candidates]
    except ReportError as exc:
        report.errors.append(str(exc))
        return report

    candidate_ids = {c.path_id for c in contracts}
    attempted = {k: bool(v) for k, v in (document.get("attempt_ledger") or {}).items()}
    selected = {str(entry.get("path_id")) for entry in candidates if entry.get("selected")}
    for path_id in attempted:
        if path_id not in candidate_ids:
            report.errors.append(f"attempt ledger names an unknown path {path_id!r}")
    provenance = document.get("case_provenance") or []
    by_id = {str(v.case_id): v for v in views}
    for row in provenance:
        case_id, path_id = str(row.get("case_id", "")), str(row.get("path_id", ""))
        if path_id and path_id not in candidate_ids:
            report.errors.append(f"{case_id}: path_id {path_id!r} is not a candidate of this run")
        live = by_id.get(case_id)
        if live is not None and path_id != live.path_id:
            report.errors.append(
                f"{case_id}: artifact path_id {live.path_id!r} differs from the run report {path_id!r}"
            )
    if report.errors:
        return report

    from testagent.pipeline.linkcheck import check_contract_cases

    outcome = check_contract_cases(
        contracts, views, endpoints, attempted=tuple(k for k, v in attempted.items() if v)
    )
    report.metrics = outcome.metrics()
    report.metrics["selected_paths"] = len(selected)
    report.metrics["attempted_paths"] = sum(1 for v in attempted.values() if v)
    report.per_case = {
        case_id: {
            "grade": res.grade,
            "orphans": list(res.orphans),
            "forward_references": list(res.forward_references),
            "bind_violations": list(res.bind_violations),
            "fulfillment_violations": list(res.fulfillment_violations),
        }
        for case_id, res in outcome.outcomes.items()
    }
    for path_id, coverage in outcome.paths.items():
        if coverage.state != "COVERED" and attempted.get(path_id):
            report.findings.append(
                f"{path_id}: {coverage.state} "
                f"(pairs {len(coverage.covered_pairs & coverage.required_pairs)}"
                f"/{len(coverage.required_pairs)})"
            )
    for case_id, row in report.per_case.items():
        if row["grade"] == "REJECTED":
            report.findings.append(f"{case_id}: REJECTED {json.dumps(row, ensure_ascii=False)}")
    unattempted = [p for p in sorted(candidate_ids - selected) if not attempted.get(p)]
    if unattempted:
        # v15 §8.1: candidate coverage keeps the denominator — an unattempted
        # candidate is a finding, not a smaller universe.
        report.findings.append(
            f"{len(unattempted)} candidate path(s) never attempted: {', '.join(unattempted)}"
        )
    return report


def observe(artifact: list[dict[str, Any]], endpoints: list[Any]) -> CheckReport:
    """No sidecar: only what the text itself proves (selection unknown)."""
    report = CheckReport(mode="observe")
    try:
        views = _views(artifact)
    except ReportError as exc:
        report.errors.append(str(exc))
        return report
    for view in views:
        facts = gate1_closure(view)
        violations = gate3_binds(view, endpoints)
        problems = [
            *(f"orphan:{o}" for o in facts["orphans"]),
            *(f"forward_reference:{f}" for f in facts["forward_references"]),
            *(f"ambiguous_producer:{a}" for a in facts["ambiguous_producers"]),
            *(f"bind:{v}" for v in violations),
        ]
        report.per_case[str(view.case_id)] = {"grade": "unchecked", "problems": problems}
        report.findings.extend(f"{view.case_id}: {p}" for p in problems)
    report.metrics = {
        "selection": "unknown",
        "contract_coverage": "n/a",
        "cases": len(views),
        "note": "no run metadata: nothing is claimed about path fulfilment",
    }
    return report
