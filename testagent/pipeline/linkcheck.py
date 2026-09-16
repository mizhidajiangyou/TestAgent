"""LinkCheck gates (LINK-S4, plan-links-v15 §6).

Pure, zero-LLM, post-generation verification over CaseView artifacts:

- **Gate1 lifecycle facts** (§6.3): placeholder occurrences with roles;
  production via the finite verb templates / assignment / binds; closure
  classification (orphan / forward_reference / ambiguous_producer /
  producer_source_conflict / external).
- **Gate3 declaration legality** (§6.4): binds v2 structure, endpoint
  existence, field addresses against the spec (known-missing vs
  unresolvable).
- **Gate2 fulfillment** (§6.5): per-case endpoint sequence, bridge steps,
  required bindings with actual consumption on the target request field;
  observable assertions for non-value relations; pair/path coverage with
  REJECTED > DRAFT > INTEGRATION precedence.
- **Metrics** (§8.1): denominators fixed to candidate contracts; empty
  denominators render ``"n/a"``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from testagent.pipeline.links_fields import CaseView, FieldAddress, request_fields
from testagent.pipeline.links_graph import BindingCandidate
from testagent.pipeline.pathplanner import PathContract

__all__ = [
    "DECLARED_LEGEND",
    "GateOutcome",
    "PathCoverage",
    "RunReport",
    "check_contract_cases",
    "gate1_closure",
    "gate3_binds",
    "render_run_report",
]

DECLARED_LEGEND = "结构合法但业务语义未经验证，不等同于业务绑定成立"

_EXTERNAL_IDS = frozenset({"TOKEN", "BEARER_TOKEN", "API_KEY", "BASE_URL", "SESSION_ID"})
_PLACEHOLDER_RE = re.compile(r"<([A-Z][A-Z0-9_]{0,30})>")
#: Production verb templates (§6.3): verb ... <ID> (no line crossing, <=80 chars).
_PRODUCER_VERB_RE = re.compile(
    r"(?:store|save|record|capture|extract|keep|note|remember|grab|retain|collect|保存|记录|提取|捕获|记下|留存|记为|作为|拿到|获得|返回)[^\n]{0,80}?<([A-Z][A-Z0-9_]{0,30})>",
    re.IGNORECASE,
)
_ASSIGN_RE = re.compile(r"<([A-Z][A-Z0-9_]{0,30})>\s*:?=\s*\S+")
_ENDPOINT_TOKEN_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(/\S+)")
_BIND_ENDPOINT_RE = re.compile(
    r"^(?P<method>GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(?P<path>/\S+)\s+"
    r"(?P<loc>response|body|params)\.(?P<field>[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)$"
)
_ASSERT_RE = re.compile(
    r"(?:assert|expect|验证|断言)\s+(?:response\.)?([\w.]+)\s*(==|!=|>=|<=|>|<|contains)\s*(\S+)",
    re.IGNORECASE,
)


@dataclass
class GateOutcome:
    """Per-case Gate results (REJECTED > DRAFT > INTEGRATION)."""

    case_id: str
    grade: str = "INTEGRATION"
    orphans: list[str] = field(default_factory=list)
    forward_references: list[str] = field(default_factory=list)
    ambiguous_producers: list[str] = field(default_factory=list)
    producer_source_conflicts: list[str] = field(default_factory=list)
    external_ids: list[str] = field(default_factory=list)
    bind_violations: list[dict[str, str]] = field(default_factory=list)
    fulfillment_violations: list[str] = field(default_factory=list)
    missing_bind_declarations: list[str] = field(default_factory=list)
    covered_pairs: set[tuple[str, int]] = field(default_factory=set)

    def finalize(self) -> str:
        if (
            self.bind_violations
            or self.fulfillment_violations
            or self.missing_bind_declarations
            or self.producer_source_conflicts
        ):
            return "REJECTED"
        if self.orphans or self.forward_references or self.ambiguous_producers:
            return "DRAFT"
        return "INTEGRATION"


# ----------------------------------------------------------------------
# Endpoint matcher (§6.2)
# ----------------------------------------------------------------------


def _strip_wrappers(token: str) -> str:
    token = token.strip()
    prev: str | None = None
    while prev != token:
        prev = token
        # paired wrappers first (e.g. ("/orders")); braces are NOT wrappers
        # here — template placeholders `{id}` are part of the path (v15 §6.2)
        for pair in (("(", ")"), ("[", "]"), ('"', '"'), ("'", "'")):
            if token.startswith(pair[0]) and token.endswith(pair[1]) and len(token) > 1:
                token = token[1:-1]
        for ch in "，。、）】》」』；；，,.;:)\"'`()[]":
            token = token.strip(ch)
    return token


def match_endpoint(text: str, spec_paths: dict[str, set[str]]) -> tuple[str, str] | None:
    """Scan adjacent method/path tokens; template-exact path match; wrapper
    characters may be stripped. Returns (method, path) or None."""
    for m in re.finditer(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s*([^\n]{0,24})", text):
        method = m.group(1).upper()
        rest = m.group(2)
        # Take progressive prefixes of the rest and try to strip wrappers
        # into a spec path.
        for width in range(len(rest), 0, -1):
            candidate = _strip_wrappers(rest[:width])
            if (
                candidate.startswith("/")
                and candidate in spec_paths
                and method in spec_paths[candidate]
            ):
                return (method, candidate)
            # stop early once a bare path prefix can no longer grow into a
            # spec path (all spec paths that share this prefix are gone)
            if candidate.startswith("/") and not any(p.startswith(candidate) for p in spec_paths):
                break
    return None


# ----------------------------------------------------------------------
# Gate1 (§6.3)
# ----------------------------------------------------------------------


def gate1_closure(case: CaseView) -> dict[str, Any]:
    """Classify every placeholder occurrence over the four sections."""
    producers: dict[str, list[int]] = {}
    consumers: dict[str, list[tuple[int, str]]] = {}
    external: list[str] = []
    conflicts: list[str] = []

    steps = list(case.steps)
    # binds-declared producer endpoints override text attribution when legal.
    bind_producer_step: dict[str, int] = {}
    for name, spec in case.binds.items():
        if isinstance(spec, dict) and isinstance(spec.get("producer"), str):
            bind_producer_step[str(name)] = -1  # resolved after source-call scan
    # find the first step whose single canonical endpoint == bind producer
    for name, spec in case.binds.items():
        producer = spec.get("producer") if isinstance(spec, dict) else None
        if not isinstance(producer, str):
            continue
        m = re.match(r"^([A-Za-z]+)\s+(/\S+)\s+response\.", producer)
        if not m:
            continue
        wanted = (m.group(1).upper(), m.group(2))
        for index, step in enumerate(steps):
            hits = [h for h in _ENDPOINT_TOKEN_RE.finditer(step)]
            canonical = [(h.group(1).upper(), h.group(2)) for h in hits]
            if (wanted in canonical) and len(set(canonical)) == 1:
                bind_producer_step[name] = index
                break

    for index, step in enumerate(steps):
        text = step
        for m in _ASSIGN_RE.finditer(text):
            name = m.group(1)
            producers.setdefault(name, []).append(index)
        for m in _PRODUCER_VERB_RE.finditer(text):
            name = m.group(1)
            producers.setdefault(name, []).append(index)
        assignment_spans = [m.span(1) for m in _ASSIGN_RE.finditer(text)]
        for m in _PLACEHOLDER_RE.finditer(text):
            if any(start <= m.start(1) < end for start, end in assignment_spans):
                continue
            consumers.setdefault(m.group(1), []).append((index, "steps"))

    # expected_results / cleanup only consume.
    for section_name, section in (("expected", case.expected_results), ("cleanup", case.cleanup)):
        for item in section:
            for m in _PLACEHOLDER_RE.finditer(item):
                consumers.setdefault(m.group(1), []).append((-1, section_name))

    precondition_provided: set[str] = set()
    for name in _PLACEHOLDER_RE.findall(" ".join(case.preconditions)):
        precondition_provided.add(name)
        external.append(name)

    orphans: list[str] = []
    forwards: list[str] = []
    ambiguous: list[str] = []
    for name, occurrences in consumers.items():
        if name in _EXTERNAL_IDS:
            continue
        produced_at = producers.get(name) or (
            [bind_producer_step[name]] if bind_producer_step.get(name, -1) >= 0 else []
        )
        if name in precondition_provided:
            continue  # value provided by precondition: closure available
        if len(set(produced_at)) > 1:
            ambiguous.append(name)
            continue
        if not produced_at:
            orphans.append(name)
            continue
        first = min(produced_at)
        if any(step_index < first for step_index, _ in occurrences if step_index >= 0):
            forwards.append(name)
    for name in producers:
        if name not in consumers and name not in _EXTERNAL_IDS:
            continue  # producer-only: audit-only, not a violation
    for name, step_index in bind_producer_step.items():
        if step_index == -1 and case.binds.get(name, {}).get("producer"):
            consumers.setdefault(name, [])
            if name not in producers:
                conflicts.append(f"producer_step_not_found:{name}")
    return {
        "orphans": sorted(set(orphans)),
        "forward_references": sorted(set(forwards)),
        "ambiguous_producers": sorted(set(ambiguous)),
        "conflicts": conflicts,
        "external": sorted(set(external)),
        "producers": {k: v for k, v in producers.items()},
    }


# ----------------------------------------------------------------------
# Gate3 (§6.4)
# ----------------------------------------------------------------------


def _spec_index(endpoints: Sequence[Any]) -> dict[str, Any]:
    return {f"{ep.method.upper()} {ep.path}": ep for ep in endpoints}


def gate3_binds(case: CaseView, endpoints: Sequence[Any]) -> list[dict[str, str]]:
    """binds v2 legality: keys, syntax, endpoint existence, field addresses."""
    violations: list[dict[str, str]] = []
    index = _spec_index(endpoints)
    key_re = re.compile(r"^[A-Z][A-Z0-9_]{0,30}$")
    for name, spec in case.binds.items():
        if not key_re.match(str(name)):
            violations.append(
                {"class": "MALFORMED_BIND", "key": str(name), "detail": "bad identifier"}
            )
            continue
        if not isinstance(spec, dict):
            violations.append(
                {"class": "MALFORMED_BIND", "key": name, "detail": "bind value not an object"}
            )
            continue
        extra = set(spec) - {"producer", "consumer", "reason"}
        if extra:
            violations.append(
                {"class": "MALFORMED_BIND", "key": name, "detail": f"unknown keys {sorted(extra)}"}
            )
            continue
        producer = spec.get("producer")
        if not isinstance(producer, str) or not producer.strip():
            violations.append(
                {"class": "MALFORMED_BIND", "key": name, "detail": "producer required"}
            )
            continue
        for side in ("producer", "consumer"):
            raw = spec.get(side)
            if raw is None:
                if side == "producer":
                    continue
                continue
            if not isinstance(raw, str) or not raw.strip():
                violations.append(
                    {"class": "MALFORMED_BIND", "key": name, "detail": f"{side} empty"}
                )
                continue
            m = _BIND_ENDPOINT_RE.match(raw.strip())
            if not m:
                violations.append(
                    {
                        "class": "MALFORMED_BIND",
                        "key": name,
                        "detail": f"unparseable {side}: {raw!r}",
                    }
                )
                continue
            method, path = m.group("method").upper(), m.group("path")
            endpoint = index.get(f"{method} {path}")
            if endpoint is None:
                violations.append(
                    {
                        "class": "unknown_endpoint",
                        "key": name,
                        "detail": f"{side} endpoint {method} {path}",
                    }
                )
                continue
            address = FieldAddress.parse(f"{m.group('loc')}.{m.group('field')}")
            if address is None:
                violations.append(
                    {"class": "MALFORMED_BIND", "key": name, "detail": f"bad field {raw!r}"}
                )
                continue
            if address.location == "response":
                fields, known = _known_response_fields(endpoint)
                if not known:
                    continue  # schema-blind: not validated, not a violation
                last = address.path.split(".")[-1]
                if last not in fields:
                    violations.append(
                        {
                            "class": "unknown_response_field",
                            "key": name,
                            "detail": f"{address.render()} not in schema",
                        }
                    )
            else:
                fields, known = _known_request_fields(endpoint)
                if not known:
                    continue
                last = address.path.split(".")[-1]
                if last not in fields:
                    violations.append(
                        {
                            "class": "unknown_request_field",
                            "key": name,
                            "detail": f"{address.render()} not in schema",
                        }
                    )
    return violations


def _known_response_fields(endpoint: Any) -> tuple[set[str], bool]:
    raw_schemas = None
    if hasattr(endpoint, "response_schemas"):
        raw_schemas = endpoint.response_schemas
    schemas = raw_schemas or {}
    if not schemas:
        return set(), False
    fields: set[str] = set()
    for schema in schemas.values():
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        if isinstance(props, dict):
            fields.update(str(k) for k in props)
    return fields, True


def _known_request_fields(endpoint: Any) -> tuple[set[str], bool]:
    views = request_fields(endpoint)
    if not views:
        return set(), False
    fields = set()
    for view in views:
        fields.add(view.address.split(".", 1)[1])
    return fields, True


# ----------------------------------------------------------------------
# Gate2 (§6.5)
# ----------------------------------------------------------------------


def check_contract_cases(
    contracts: Sequence[PathContract],
    cases: Sequence[CaseView],
    endpoints: Sequence[Any],
    *,
    attempted: Sequence[str] = (),
) -> RunReport:
    """Run Gate1+Gate3 per case, Gate2 per contract, and aggregate the
    run report (denominators = candidate contracts)."""
    outcomes: dict[str, GateOutcome] = {}
    spec_paths: dict[str, set[str]] = {}
    for ep in endpoints:
        spec_paths.setdefault(str(ep.path), set()).add(str(ep.method).upper())
    index = _spec_index(endpoints)

    # only this-run L3b cases may fulfill: path_id non-empty AND staged l3b
    fulfill_cases = [c for c in cases if c.path_id]
    by_path: dict[str, list[CaseView]] = {}
    for case in fulfill_cases:
        by_path.setdefault(case.path_id, []).append(case)

    covered_pairs: dict[str, set[int]] = {}
    best_case_grade: dict[str, str] = {}
    pair_states: dict[tuple[str, int], dict[str, str]] = {}

    for contract in contracts:
        path_cases = by_path.get(contract.path_id, [])
        path_grades: list[str] = []
        for case in path_cases:
            if case.case_id in outcomes:
                outcome = outcomes[case.case_id]
            else:
                outcome = GateOutcome(case_id=case.case_id)
                facts = gate1_closure(case)
                outcome.orphans = facts["orphans"]
                outcome.forward_references = facts["forward_references"]
                outcome.ambiguous_producers = facts["ambiguous_producers"]
                outcome.producer_source_conflicts = facts["conflicts"]
                outcome.external_ids = facts["external"]
                outcome.bind_violations = gate3_binds(case, endpoints)
                outcomes[case.case_id] = outcome
            # Gate2 per hop
            violations = _gate2_case(contract, case, index, spec_paths, outcome)
            outcome.fulfillment_violations.extend(violations)
            if not outcome.fulfillment_violations and not outcome.bind_violations:
                for pair in contract.required_pairs:
                    _, hop_index = pair["pair_key"]
                    covered_pairs.setdefault(contract.path_id, set()).add(hop_index)
                    pair_states.setdefault((contract.path_id, hop_index), {})
                    if pair["kind"] == "DATA_FLOW" and pair["required_binding"]:
                        binding = pair["required_binding"]
                        state = "EXPLICIT" if binding["mode"] == "EXACT" else "DECLARED"
                        pair_states[(contract.path_id, hop_index)]["binding"] = state
                    else:
                        pair_states[(contract.path_id, hop_index)]["binding"] = "DECLARED"
            outcome.grade = outcome.finalize()
            path_grades.append(outcome.grade)
        best_case_grade[contract.path_id] = (
            "INTEGRATION"
            if "INTEGRATION" in path_grades
            else ("DRAFT" if "DRAFT" in path_grades else ("REJECTED" if path_grades else ""))
        )

    attempted_set = set(attempted)
    paths: dict[str, PathCoverage] = {}
    for contract in contracts:
        pairs = {pair["pair_key"][1] for pair in contract.required_pairs}
        covered = covered_pairs.get(contract.path_id, set())
        attempted_flag = contract.path_id in attempted_set
        if covered == pairs and pairs:
            state = "COVERED"
        elif best_case_grade.get(contract.path_id) or attempted_flag:
            state = (
                "FAILED"
                if attempted_flag or best_case_grade.get(contract.path_id)
                else "UNATTEMPTED"
            )
        else:
            state = "UNATTEMPTED"
        paths[contract.path_id] = PathCoverage(
            path_id=contract.path_id,
            state=state,
            attempted=attempted_flag,
            required_pairs=pairs,
            covered_pairs=covered,
            has_integration=best_case_grade.get(contract.path_id) == "INTEGRATION",
            pair_states={k[1]: v for k, v in pair_states.items() if k[0] == contract.path_id},
        )
    return RunReport(outcomes=outcomes, paths=paths)


def _gate2_case(
    contract: PathContract,
    case: CaseView,
    index: dict[str, Any],
    spec_paths: dict[str, set[str]],
    outcome: GateOutcome,
) -> list[str]:
    """Fulfill every endpoint in order + every required binding on this
    single case (no cross-case stitching)."""
    violations: list[str] = []
    steps = list(case.steps)
    executed: list[tuple[str, str]] = []
    for step in steps:
        hits = [(m.group(1).upper(), m.group(2)) for m in _ENDPOINT_TOKEN_RE.finditer(step)]
        for method, raw in hits:
            candidate = _strip_wrappers(raw)
            executed.append((method, candidate))

    endpoint_seq = list(contract.endpoints)
    # endpoint sequence check (ordered subsequence match with exact templates)
    it = iter(executed)
    for expected in endpoint_seq:
        method, _, path = expected.partition(" ")
        found = False
        for em, ep in it:
            if em == method and ep == path:
                found = True
                break
        if not found:
            violations.append(f"endpoint_sequence_missing:{expected}")
            break

    for pair in contract.required_pairs:
        hop = contract.business_hops[pair["pair_key"][1]]
        if hop.kind == "DATA_FLOW" and hop.candidate:
            binding = hop.candidate
            if binding.consumer and not _consumes_value(case, binding, index):
                violations.append(
                    f"binding_not_fulfilled:{binding.consumer} (required by {contract.path_id})"
                )
                outcome.missing_bind_declarations.append(binding.consumer)
        else:
            # non-value relation: target step needs an observable assertion
            if not _target_step_has_assertion(case, hop.target, spec_paths):
                violations.append(f"no_target_assertion:{hop.target}")

    # binds declared producer must actually run (§6.1 producer-only rule)
    for name, spec in case.binds.items():
        if not isinstance(spec, dict):
            continue
        producer = spec.get("producer")
        m = re.match(r"^([A-Za-z]+)\s+(/\S+)\s+response\.", str(producer or ""))
        if not m:
            continue
        wanted = (m.group(1).upper(), m.group(2))
        if not any(em == wanted[0] and ep == wanted[1] for em, ep in executed):
            violations.append(f"producer_not_executed:{name}")
    return violations


def _consumes_value(case: CaseView, binding: BindingCandidate, index: dict[str, Any]) -> bool:
    """Actual consumption: dot form or JSON body assignment of the target
    request field with the SAME identifier (§6.5)."""
    consumer = binding.consumer
    if not consumer:
        return True
    identifier = None
    for name, spec in case.binds.items():
        if not isinstance(spec, dict):
            continue
        raw = str(spec.get("consumer", ""))
        # The bind stores "METHOD /path location.field"; the contract's
        # consumer is the canonical address ("location.field"). Match on
        # the trailing address segment.
        if raw == consumer or raw.endswith(f" {consumer}"):
            identifier = str(name)
            break
    if identifier is None:
        return False
    consumer_address = FieldAddress.parse(consumer.replace("request.", "body.", 1))
    if consumer_address is None:
        return False
    last_field = consumer_address.path.split(".")[-1]
    for step in case.steps:
        if re.search(rf"{re.escape(last_field)}\s*=\s*<{identifier}>", step):
            return True
        if re.search(rf"{re.escape(last_field)}\s*=\s*\"?<{identifier}>\"?", step):
            return True
        # JSON form: "field": "<ID>" (placeholder keeps its angle brackets
        # inside the JSON string value, v15 §6.5)
        if re.search(rf'"{re.escape(last_field)}"\s*:\s*"<{re.escape(identifier)}>"', step):
            return True
        # params form
        if consumer_address.location == "params" and re.search(
            rf"params\.{re.escape(last_field)}=<{identifier}>", step
        ):
            return True
    return False


def _target_step_has_assertion(
    case: CaseView, target: str, spec_paths: dict[str, set[str]]
) -> bool:
    for step in case.steps:
        hits = [(m.group(1).upper(), m.group(2)) for m in _ENDPOINT_TOKEN_RE.finditer(step)]
        if not any(
            method == target.split(" ")[0] and path == target.split(" ")[1] for method, path in hits
        ):
            continue
        for item in (*case.expected_results, step):
            if _ASSERT_RE.search(item):
                return True
    return False


@dataclass
class PathCoverage:
    path_id: str
    state: str  # COVERED | FAILED | UNATTEMPTED
    attempted: bool
    required_pairs: set[int]
    covered_pairs: set[int]
    has_integration: bool
    pair_states: dict[int, dict[str, str]]


@dataclass
class RunReport:
    outcomes: dict[str, GateOutcome]
    paths: dict[str, PathCoverage]

    def grades(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for outcome in self.outcomes.values():
            counts[outcome.grade] = counts.get(outcome.grade, 0) + 1
        return counts

    def metrics(self) -> dict[str, Any]:
        """§8.1 metrics with n/a on empty denominators."""
        total_pairs = sum(len(p.required_pairs) for p in self.paths.values())
        covered = sum(len(p.covered_pairs & p.required_pairs) for p in self.paths.values())
        attempted_paths = [p for p in self.paths.values() if p.attempted]
        attempted_pairs = sum(len(p.required_pairs) for p in attempted_paths)
        covered_attempted = sum(len(p.covered_pairs & p.required_pairs) for p in attempted_paths)
        covered_paths = sum(1 for p in self.paths.values() if p.has_integration)

        def rate(numer: int, denom: int) -> str | float:
            return "n/a" if denom == 0 else round(numer / denom, 4)

        return {
            "required_pairs_total": total_pairs,
            "covered_pairs": covered,
            "pair_rate": rate(covered, total_pairs),
            "pair_attempted_rate": rate(covered_attempted, attempted_pairs),
            "path_rate": rate(covered_paths, len(self.paths)),
            "path_attempted_rate": rate(
                sum(1 for p in attempted_paths if p.has_integration), len(attempted_paths)
            ),
            "candidate_paths": len(self.paths),
            "grades": self.grades(),
        }


def render_run_report(report: RunReport) -> str:
    metrics = report.metrics()
    lines = [
        "# Links run report",
        "",
        f"Grades: {report.grades()}",
        f"Metrics: {json.dumps(metrics, ensure_ascii=False)}",
        "",
        f"DECLARED_LEGEND: {DECLARED_LEGEND}",
        "",
    ]
    for path_id, coverage in report.paths.items():
        lines.append(
            f"- {path_id}: {coverage.state} "
            f"(pairs {len(coverage.covered_pairs)}/{len(coverage.required_pairs)}, "
            f"integration={coverage.has_integration})"
        )
    lines.append("")
    return "\n".join(lines)
