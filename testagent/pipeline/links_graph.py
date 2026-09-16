"""LinkGraph: spec-derived relations R1-R4 + R6 ingestion (LINK-S2b,
plan-links-v15 §4.1).

Pure functions from parsed endpoints to a relation graph. Evidence carries
rule id, trust and provenance; identical topology relations merge evidence
(but DATA_FLOW keeps every binding candidate — no "strongest edge wins").
Trust separation: relation trust and field-binding trust are independent;
prose (R6) is at most MEDIUM and can never upgrade a spec-derived candidate.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from testagent.pipeline.links_fields import request_fields, response_fields
from testagent.pipeline.links_r6 import R6Edge, ground_r6_edge

__all__ = [
    "EXCLUDED_FIELDS",
    "BindingCandidate",
    "Edge",
    "LinkGraph",
    "RelationKind",
    "Trust",
    "build_graph",
]

#: R1 excludes fixed generic envelope fields (v15 §4.1).
EXCLUDED_FIELDS = frozenset(
    {
        "code",
        "message",
        "msg",
        "success",
        "data",
        "result",
        "list",
        "items",
        "total",
        "page",
        "limit",
        "timestamp",
    }
)

_ID_SUFFIX_RE = re.compile(r"(?:^|_)([a-z]+)(?:_?id)$", re.IGNORECASE)


class RelationKind:
    DATA_FLOW = "DATA_FLOW"
    LIFECYCLE = "LIFECYCLE"
    PRECONDITION = "PRECONDITION"
    SIDE_EFFECT = "SIDE_EFFECT"


class Trust:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class BindingCandidate:
    """One field-level binding hypothesis on a DATA_FLOW edge."""

    producer: str  # canonical address ("response.id") or "" for R6 no-value
    consumer: str  # canonical address ("body.productId") or ""
    mode: str  # EXACT | DECLARED
    trust: str
    rule: str  # R1 | R2 | R6
    producer_endpoint: str = ""  # "POST /users" when known
    consumer_endpoint: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.producer, self.consumer, self.mode)


@dataclass
class Edge:
    """One topology relation: ``(source, target, kind)`` identity."""

    source: str
    target: str
    kind: str
    evidence: list[dict[str, str]] = field(default_factory=list)
    binding_candidates: list[BindingCandidate] = field(default_factory=list)

    @property
    def relation_key(self) -> tuple[str, str, str]:
        return (self.source, self.target, self.kind)

    def trust(self) -> str:
        return max((e.get("trust", Trust.LOW) for e in self.evidence), default=Trust.LOW)


def _endpoint_identity(ep: Any) -> str:
    return f"{ep.method.upper()} {ep.path}"


def _module_of(ep: Any) -> str:
    tags = getattr(ep, "tags", None) or []
    if tags and str(tags[0]).strip():
        return str(tags[0]).strip().casefold()
    segments = [s for s in str(ep.path).strip("/").split("/") if s and not s.startswith("{")]
    return segments[0].casefold() if segments else "root"


def _singular(word: str) -> str:
    """Fixed English plural normalization (v15 §4.1: 末尾 s/ies only)."""
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _resource_of(ep: Any) -> str:
    segments = [s for s in str(ep.path).strip("/").split("/") if s and not s.startswith("{")]
    return _singular(segments[-1].casefold()) if segments else "root"


def _response_field_map(ep: Any) -> dict[str, str]:
    """field name (top-level + envelope) -> declared type, status annotated
    by the caller; excluded generics removed."""
    out: dict[str, str] = {}
    for view in response_fields(ep):
        name = view.address.split(".", 1)[1]
        if name in EXCLUDED_FIELDS:
            continue
        out[name] = view.type
    return out


def _request_field_map(ep: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for view in request_fields(ep):
        address = view.address.split(".", 1)[1]
        if "." in address:
            continue  # body envelope inner fields handled at top level only
        if address in EXCLUDED_FIELDS and view.address.startswith("body."):
            continue
        out[address] = view.type
    return out


def _compatible(t1: str, t2: str) -> bool:
    if t1 == "?" or t2 == "?":
        return True  # unknown types never block reachability, but never EXACT
    return t1 == t2 or {t1, t2} == {"integer", "number"}


def _r1_data_flow(endpoints: Sequence[Any]) -> list[Edge]:
    """R1: same-name type-compatible business field in one endpoint's
    response and another's request -> DATA_FLOW, EXACT candidate."""
    edges: dict[tuple[str, str, str], Edge] = {}
    field_cache = [
        (_endpoint_identity(ep), _response_field_map(ep), _request_field_map(ep))
        for ep in endpoints
    ]
    for src_id, resp_fields, _ in field_cache:
        for dst_id, _, req_fields in field_cache:
            if src_id == dst_id:
                continue
            for name, rtype in resp_fields.items():
                if name not in req_fields:
                    continue
                qtype = req_fields[name]
                mode = (
                    "EXACT"
                    if (rtype != "?" and qtype != "?" and _compatible(rtype, qtype))
                    else "DECLARED"
                )
                candidate = BindingCandidate(
                    producer=f"response.{name}",
                    consumer=f"body.{name}",
                    mode=mode,
                    trust=Trust.HIGH,
                    rule="R1",
                    producer_endpoint=src_id,
                    consumer_endpoint=dst_id,
                )
                key = (src_id, dst_id, RelationKind.DATA_FLOW)
                edge = edges.setdefault(
                    key, Edge(source=src_id, target=dst_id, kind=RelationKind.DATA_FLOW)
                )
                edge.binding_candidates.append(candidate)
                edge.evidence.append(
                    {
                        "rule": "R1",
                        "trust": Trust.HIGH,
                        "field": name,
                        "producer": src_id,
                        "consumer": dst_id,
                        "limitation": "same-name+type proves address reachability, not business identity",
                    }
                )
    return list(edges.values())


def _r2_declared(endpoints: Sequence[Any]) -> list[Edge]:
    """R2: request field ``<resource>Id`` + a POST creator of that resource
    -> DATA_FLOW DECLARED (consumer address known; producer is a hint)."""
    edges: dict[tuple[str, str, str], Edge] = {}
    creators: dict[str, str] = {}
    for ep in endpoints:
        if ep.method.upper() == "POST":
            creators.setdefault(_resource_of(ep), _endpoint_identity(ep))
    for ep in endpoints:
        for name in _request_field_map(ep):
            m = _ID_SUFFIX_RE.search(name)
            if not m:
                continue
            resource = m.group(1)
            creator = creators.get(resource)
            if not creator or creator == _endpoint_identity(ep):
                continue
            candidate = BindingCandidate(
                producer="response.id",
                consumer=f"body.{name}",
                mode="DECLARED",
                trust=Trust.HIGH,
                rule="R2",
                producer_endpoint=creator,
                consumer_endpoint=_endpoint_identity(ep),
            )
            key = (creator, _endpoint_identity(ep), RelationKind.DATA_FLOW)
            edge = edges.setdefault(
                key,
                Edge(source=creator, target=_endpoint_identity(ep), kind=RelationKind.DATA_FLOW),
            )
            edge.binding_candidates.append(candidate)
            edge.evidence.append(
                {
                    "rule": "R2",
                    "trust": Trust.HIGH,
                    "field": name,
                    "producer_hint": creator,
                    "note": "producer address is a hint, not a proven field",
                }
            )
    return list(edges.values())


def _r3_r4_lifecycle(endpoints: Sequence[Any]) -> list[Edge]:
    """R3 (same path) and R4 (child path to nearest existing ancestor)
    -> undirected LIFECYCLE adjacency, normalized as sorted endpoint pairs."""
    edges: dict[tuple[str, str, str], Edge] = {}
    by_path: dict[str, list[Any]] = {}
    for ep in endpoints:
        by_path.setdefault(str(ep.path), []).append(ep)

    def add(a: Any, b: Any, rule: str) -> None:
        ia, ib = _endpoint_identity(a), _endpoint_identity(b)
        if ia == ib:
            return
        lo, hi = sorted((ia, ib))
        key = (lo, hi, RelationKind.LIFECYCLE)
        edge = edges.setdefault(key, Edge(source=lo, target=hi, kind=RelationKind.LIFECYCLE))
        edge.evidence.append({"rule": rule, "trust": Trust.HIGH})

    for eps in by_path.values():
        for other in eps[1:]:
            add(eps[0], other, "R3")
    path_set = set(by_path)
    for path in path_set:
        segments = [s for s in path.strip("/").split("/") if s]
        # Walk from the full path upward: nearest existing ancestor wins.
        for i in range(len(segments) - 1, 0, -1):
            ancestor = "/" + "/".join(segments[:i])
            if ancestor in path_set:
                for child in by_path[path]:
                    for parent in by_path[ancestor]:
                        add(child, parent, "R4")
                break
    return list(edges.values())


def ingest_r6(
    graph: LinkGraph,
    edges: Sequence[R6Edge],
    endpoints: Sequence[Any],
) -> None:
    """Merge R6 prose relations: ground to endpoint pairs (module direction
    preserved), always DECLARED, trust MEDIUM."""
    module_map: dict[str, list[Any]] = {}
    for ep in endpoints:
        module_map.setdefault(_module_of(ep), []).append(ep)
    for r6 in edges:
        pairs = ground_r6_edge(r6, module_map)
        for src, dst in pairs:
            graph.add_edge(
                Edge(
                    source=src,
                    target=dst,
                    kind=r6.kind,
                    evidence=[
                        {
                            "rule": "R6",
                            "trust": Trust.MEDIUM,
                            "clause": r6.clause_id,
                            "text": r6.evidence,
                        }
                    ],
                    binding_candidates=[
                        BindingCandidate(
                            producer="",
                            consumer="",
                            mode="DECLARED",
                            trust=Trust.MEDIUM,
                            rule="R6",
                            producer_endpoint=src,
                            consumer_endpoint=dst,
                        )
                    ],
                )
            )


class LinkGraph:
    """Merged relation graph (v15 §4.1 invariants)."""

    def __init__(self, endpoints: Sequence[Any]) -> None:
        self._endpoints = list(endpoints)
        self._edges: dict[tuple[str, str, str], Edge] = {}

    def add_edge(self, edge: Edge) -> None:
        existing = self._edges.get(edge.relation_key)
        if existing is None:
            self._edges[edge.relation_key] = edge
            return
        # Same topology: merge evidence, dedup identical candidates but keep
        # every DISTINCT field candidate (no strongest-edge override).
        existing.evidence.extend(edge.evidence)
        seen = {c.key for c in existing.binding_candidates}
        for candidate in edge.binding_candidates:
            if candidate.key not in seen:
                existing.binding_candidates.append(candidate)
                seen.add(candidate.key)

    @property
    def edges(self) -> list[Edge]:
        return list(self._edges.values())

    def data_flow_edges(self) -> list[Edge]:
        return [e for e in self._edges.values() if e.kind == RelationKind.DATA_FLOW]

    def lifecycle_adjacency(self) -> dict[str, set[str]]:
        """Undirected LIFECYCLE adjacency (R3/R4 only) for L3a/bridges."""
        adjacency: dict[str, set[str]] = {}
        for edge in self._edges.values():
            if edge.kind != RelationKind.LIFECYCLE:
                continue
            adjacency.setdefault(edge.source, set()).add(edge.target)
            adjacency.setdefault(edge.target, set()).add(edge.source)
        return adjacency

    def module_pairs(self) -> set[tuple[str, str]]:
        """Undirected module pairs (benchmark layer 1)."""
        modules = {_module_of(ep): None for ep in self._endpoints}
        ep_module = {_endpoint_identity(ep): _module_of(ep) for ep in self._endpoints}
        pairs: set[tuple[str, str]] = set()
        for edge in self._edges.values():
            if edge.kind == RelationKind.LIFECYCLE:
                continue
            sm = ep_module.get(edge.source, "")
            tm = ep_module.get(edge.target, "")
            if sm and tm and sm != tm:
                lo, hi = sorted((sm, tm))
                pairs.add((lo, hi))
        _ = modules
        return pairs


def build_graph(
    endpoints: Sequence[Any],
    r6_edges: Sequence[R6Edge] = (),
) -> LinkGraph:
    """Deterministic graph build: R1/R2/R3/R4 from the spec + optional R6
    ingestion. Endpoint identity is ``full_path``; duplicates are an input
    error (§4.1)."""
    seen: set[str] = set()
    for ep in endpoints:
        identity = _endpoint_identity(ep)
        if identity in seen:
            raise ValueError(f"duplicate endpoint identity: {identity}")
        seen.add(identity)
    graph = LinkGraph(endpoints)
    for edge in _r1_data_flow(endpoints):
        graph.add_edge(edge)
    for edge in _r2_declared(endpoints):
        graph.add_edge(edge)
    for edge in _r3_r4_lifecycle(endpoints):
        graph.add_edge(edge)
    if r6_edges:
        ingest_r6(graph, r6_edges, endpoints)
    return graph
