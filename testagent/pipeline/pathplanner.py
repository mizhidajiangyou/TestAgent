"""Path planner (LINK-S3, plan-links-v15 §5).

Enumerates cross-module business paths from the LinkGraph (DATA_FLOW /
PRECONDITION / SIDE_EFFECT hops only; LIFECYCLE is bridge material),
assigns collision-free canonical identities, ranks deterministically, and
splits the pool into candidate / not_materialized / selected / deferred
with fixed reasons.

Pure: no LLM, no I/O, no input-order dependence (rank keys use canonical
bytes; MAX_ENUMERATED_VARIANTS guards runaway enumeration).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from testagent.pipeline.links_graph import BindingCandidate, LinkGraph

__all__ = [
    "PathContract",
    "PathIdentityCollision",
    "PathIdentityCollisionError",
    "PlanningLimitExceeded",
    "PlanningLimitExceededError",
    "canonical_payload_bytes",
    "plan_paths",
]

MAX_ENUMERATED_VARIANTS = 10000

#: Business-priority table (v15 §5.2).
HOP_PRIORITY = {
    "EXACT_DATA_FLOW": 30,
    "DECLARED_DATA_FLOW": 20,
    "PRECONDITION": 10,
    "SIDE_EFFECT": 5,
}

_WRITE = {"POST", "PUT", "PATCH", "DELETE"}


class PlanningLimitExceededError(Exception):
    """Plan name: PlanningLimitExceeded (v15 §5.1)."""

    """Enumeration protection fired (v15 §5.1) — fail loudly, never
    silently truncate and claim full coverage."""


class PathIdentityCollisionError(Exception):
    """Plan name: PathIdentityCollision (v15 §5.2)."""

    """Two different canonical payloads hash to the same short path_id
    (v15 §5.2) — hard fail, no auto-suffixing."""


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    """Fixed canonical serialization (v15 §5.2)."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


@dataclass(frozen=True)
class _Hop:
    source: str
    target: str
    kind: str
    candidate: BindingCandidate | None
    bridge_before: tuple[str, ...]


@dataclass
class PathContract:
    """One planned path with its frozen identity and binding contract."""

    path_id: str
    payload: dict[str, Any]
    canonical_bytes: bytes
    endpoints: tuple[str, ...]
    business_hops: tuple[_Hop, ...]
    priority: int
    static_class: str = ""  # executable | spec_blind (INVALID never lands here)
    required_pairs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rank_key(self) -> tuple[int, int, bytes]:
        return (-self.priority, len(self.business_hops), self.canonical_bytes)

    @property
    def module_flow(self) -> list[str]:
        flow: list[str] = []
        for hop in self.business_hops:
            for endpoint in (*hop.bridge_before, hop.source, hop.target):
                module = endpoint.split(" ", 1)[1].strip("/").split("/")[0]
                if not flow or flow[-1] != module:
                    flow.append(module)
        return flow


def plan_paths(
    graph: LinkGraph,
    endpoints: Sequence[Any],
    *,
    max_hops: int = 3,
    max_planned: int = 8,
    l3b_budget: int = 3,
    seeding: str = "priority",
) -> dict[str, Any]:
    """Full planning pipeline (v15 §5.1-§5.4). Returns the pool dict with
    contracts, counts and fixed reasons."""
    payloads: dict[str, bytes] = {}
    planned: list[PathContract] = []
    invalid_attempts: dict[str, int] = {}
    enumerated = 0

    business_edges = [e for e in graph.edges if e.kind != "LIFECYCLE"]
    lifecycle = graph.lifecycle_adjacency()

    def _bridge(src: str, dst: str) -> tuple[str, ...] | None:
        """Shortest R3/R4 path within one module (R5); None when unreachable."""
        if src == dst:
            return ()
        same_module = (
            src.split(" ", 1)[1].strip("/").split("/")[0]
            == dst.split(" ", 1)[1].strip("/").split("/")[0]
        )
        if not same_module:
            return None
        # BFS over lifecycle adjacency, lexicographic tie-break.
        from collections import deque

        queue: deque[tuple[str, tuple[str, ...]]] = deque([(src, ())])
        visited = {src}
        while queue:
            node, path = queue.popleft()
            for nxt in sorted(lifecycle.get(node, ())):
                if nxt == dst:
                    return (*path, node)
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, (*path, node)))
        return None

    def _endpoint_ok(identity: str) -> bool:
        return any(f"{ep.method.upper()} {ep.path}" == identity for ep in endpoints)

    def _emit(hops: list[_Hop]) -> None:
        nonlocal enumerated
        enumerated += 1
        if enumerated > MAX_ENUMERATED_VARIANTS:
            raise PlanningLimitExceeded(f"enumeration exceeded {MAX_ENUMERATED_VARIANTS} variants")
        endpoint_seq: list[str] = []
        for hop in hops:
            for b in hop.bridge_before:
                if b not in endpoint_seq:
                    endpoint_seq.append(b)
            if hop.source not in endpoint_seq:
                endpoint_seq.append(hop.source)
            if hop.target not in endpoint_seq:
                endpoint_seq.append(hop.target)
        for e in endpoint_seq:
            if not _endpoint_ok(e):
                invalid_attempts["unknown_endpoint"] = (
                    invalid_attempts.get("unknown_endpoint", 0) + 1
                )
                return
        # DELETE only at the very end.
        for _i, e in enumerate(endpoint_seq[:-1]):
            if e.startswith("DELETE "):
                invalid_attempts["delete_not_terminal"] = (
                    invalid_attempts.get("delete_not_terminal", 0) + 1
                )
                return
        modules_in_order = [e.split(" ", 1)[1].strip("/").split("/")[0] for e in endpoint_seq]
        # no module revisits
        seen_modules: set[str] = set()
        for m in modules_in_order:
            if m in seen_modules:
                invalid_attempts["module_revisit"] = invalid_attempts.get("module_revisit", 0) + 1
                return
            seen_modules.add(m)
        payload = _payload(hops, endpoint_seq)
        data = canonical_payload_bytes(payload)
        path_id = "P" + hashlib.sha256(data).hexdigest()[:12]
        if path_id in payloads and payloads[path_id] != data:
            raise PathIdentityCollision(f"path_id collision: {path_id}")
        payloads[path_id] = data
        contract = PathContract(
            path_id=path_id,
            payload=payload,
            canonical_bytes=data,
            endpoints=tuple(endpoint_seq),
            business_hops=tuple(hops),
            priority=max(HOP_PRIORITY.get(_hop_class(h), 0) for h in hops),
        )
        planned.append(contract)

    def _payload(hops: list[_Hop], endpoint_seq: list[str]) -> dict[str, Any]:
        return {
            "version": 1,
            "endpoints": list(endpoint_seq),
            "hops": [
                {
                    "source": h.source,
                    "target": h.target,
                    "kind": h.kind,
                    "binding": (
                        {
                            "producer": h.candidate.producer,
                            "consumer": h.candidate.consumer,
                            "mode": h.candidate.mode,
                        }
                        if h.candidate and (h.candidate.producer or h.candidate.consumer)
                        else None
                    ),
                    "bridge_before": list(h.bridge_before),
                }
                for h in hops
            ],
        }

    def _walk(
        hops: list[_Hop], visited_endpoints: frozenset[str], visited_modules: frozenset[str]
    ) -> None:
        if hops:
            _emit(hops)
        if len(hops) >= max_hops:
            return
        last_target = hops[-1].target if hops else None
        last_module = last_target.split(" ", 1)[1].strip("/").split("/")[0] if last_target else None
        for edge in sorted(business_edges, key=lambda e: (e.source, e.target, e.kind)):
            if last_target is None:
                start = edge.source
            else:
                # continue only from the previous target (same endpoint or bridged)
                if edge.source == last_target:
                    start = edge.source
                else:
                    same_module = (
                        edge.source.split(" ", 1)[1].strip("/").split("/")[0] == last_module
                    )
                    if not same_module:
                        continue
                    bridge = _bridge(last_target, edge.source)
                    if bridge is None or (bridge and bridge[-1:] == ()):
                        if edge.source == last_target:
                            pass
                        else:
                            continue
                    else:
                        start = edge.source
                if edge.source != last_target and last_target is not None:
                    bridge = _bridge(last_target, edge.source)
                    if bridge is None:
                        continue
            if start in visited_endpoints or edge.target in visited_endpoints:
                continue
            target_module = edge.target.split(" ", 1)[1].strip("/").split("/")[0]
            if target_module in visited_modules and target_module != last_module:
                continue
            if last_target is not None and edge.source != last_target:
                bridge = _bridge(last_target, edge.source)
                if not bridge:  # None or empty
                    continue
                bridge_before = tuple(bridge)
            else:
                bridge_before = ()
            for candidate in edge.binding_candidates:
                _walk(
                    [
                        *hops,
                        _Hop(
                            source=edge.source,
                            target=edge.target,
                            kind=edge.kind,
                            candidate=candidate,
                            bridge_before=bridge_before,
                        ),
                    ],
                    visited_endpoints | {edge.source, edge.target},
                    visited_modules | {target_module},
                )

    _walk([], frozenset(), frozenset())

    # dedup identical path_ids (same payload re-found via another candidate
    # ordering) — canonical bytes are the identity.
    unique: dict[str, PathContract] = {}
    for contract in planned:
        unique.setdefault(contract.path_id, contract)
    planned = sorted(unique.values(), key=lambda c: c.rank_key)
    for contract in planned:
        contract.static_class = _static_class(contract)
        contract.required_pairs = _required_pairs(contract)

    candidate = planned[:max_planned]
    not_materialized = [
        {"path_id": c.path_id, "reason": "candidate_cap"} for c in planned[max_planned:]
    ]
    if seeding == "priority":
        with_consumer = [c for c in candidate if _has_concrete_consumer(c)]
        rest = [c for c in candidate if c not in with_consumer]
        ordered = with_consumer + rest
    else:
        ordered = list(candidate)
    selected = ordered[:l3b_budget]
    deferred = [{"path_id": c.path_id, "reason": "l3b_unit_budget"} for c in ordered[l3b_budget:]]

    return {
        "planned": planned,
        "candidate": candidate,
        "selected": selected,
        "not_materialized": not_materialized,
        "deferred": deferred,
        "invalid_attempts_by_reason": invalid_attempts,
        "counts": {
            "planned_paths": len(planned),
            "candidate_paths": len(candidate),
            "selected_paths": len(selected),
        },
    }


def _hop_class(hop: _Hop) -> str:
    if hop.kind == "DATA_FLOW":
        return f"{hop.candidate.mode if hop.candidate else 'DECLARED'}_DATA_FLOW"
    return hop.kind


def _has_concrete_consumer(contract: PathContract) -> bool:
    return any(
        h.kind == "DATA_FLOW" and h.candidate is not None and h.candidate.consumer
        for h in contract.business_hops
    )


def _static_class(contract: PathContract) -> str:
    """executable only when EVERY business hop has EXACT field evidence;
    any DECLARED hop (including R6 no-value) -> spec_blind (v15 §5.4)."""
    if not contract.business_hops:
        return "spec_blind"
    classes = []
    for hop in contract.business_hops:
        if hop.kind == "DATA_FLOW":
            classes.append(
                "executable" if (hop.candidate and hop.candidate.mode == "EXACT") else "spec_blind"
            )
        else:
            classes.append("executable")  # non-value relations are structurally executable
    return "executable" if all(c == "executable" for c in classes) else "spec_blind"


def _required_pairs(contract: PathContract) -> list[dict[str, Any]]:
    """One required pair per business hop (pair_key = path_id + hop index)."""
    pairs: list[dict[str, Any]] = []
    for index, hop in enumerate(contract.business_hops):
        if hop.kind == "DATA_FLOW":
            required_binding = (
                {
                    "identifier": None,
                    "producer": hop.candidate.producer,
                    "consumer": hop.candidate.consumer,
                    "mode": hop.candidate.mode,
                }
                if hop.candidate
                else None
            )
            pairs.append(
                {
                    "pair_key": (contract.path_id, index),
                    "kind": "DATA_FLOW",
                    "required_binding": required_binding,
                }
            )
        else:
            pairs.append(
                {"pair_key": (contract.path_id, index), "kind": hop.kind, "required_binding": None}
            )
    return pairs


#: Plan-facing aliases (v15 §5.1/§5.2 names without the Error suffix).
PlanningLimitExceeded = PlanningLimitExceededError
PathIdentityCollision = PathIdentityCollisionError
