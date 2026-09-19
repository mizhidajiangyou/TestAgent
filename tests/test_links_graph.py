"""S2b tests: LinkGraph R1-R4 rules, candidate merging, R6 ingestion."""

import pytest

from testagent.config.models import APIEndpoint
from testagent.pipeline.links_graph import (
    EXCLUDED_FIELDS,
    RelationKind,
    Trust,
    build_graph,
)
from testagent.pipeline.links_r6 import R6Edge


def _ep(
    method: str, path: str, *, tags=None, responses=None, body=None, params=None, rschemas=None
):
    request_body = None
    if body is not None:
        request_body = {"media_type": "application/json", "schema": body}
    return APIEndpoint(
        method=method,
        path=path,
        tags=tags or [],
        responses=responses or [],
        parameters=params or [],
        request_body=request_body,
        response_schemas=rschemas or {},
    )


_USERS_SPEC = [
    _ep(
        "POST",
        "/users",
        tags=["users"],
        responses=["201"],
        body={
            "type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            "required": ["name"],
        },
        rschemas={
            "201": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "profile": {"type": "object"}},
            }
        },
    ),
    _ep(
        "DELETE",
        "/users/{id}",
        tags=["users"],
        responses=["204"],
        params=[{"name": "id", "in": "path", "schema": {"type": "integer"}}],
    ),
    _ep(
        "POST",
        "/orders",
        tags=["orders"],
        responses=["201"],
        body={
            "type": "object",
            "properties": {"user_id": {"type": "integer"}, "total": {"type": "number"}},
        },
    ),
    _ep("GET", "/orders", tags=["orders"], responses=["200"]),
]


class TestR1:
    def test_exact_candidate_on_compatible_field(self) -> None:
        """POST /users response.id (integer) meets POST /orders request
        total? No — the shared business field here is none; id vs user_id
        is R2 territory. R1 needs the SAME name: add an endpoint whose body
        declares `id`."""
        spec = [
            *_USERS_SPEC,
            _ep(
                "POST",
                "/reviews",
                tags=["reviews"],
                body={"type": "object", "properties": {"id": {"type": "integer"}}},
            ),
        ]
        graph = build_graph(spec)
        edge = next(
            e
            for e in graph.data_flow_edges()
            if e.source == "POST /users" and e.target == "POST /reviews"
        )
        candidate = edge.binding_candidates[0]
        assert candidate.mode == "EXACT"
        assert candidate.producer == "response.id"
        assert candidate.consumer == "body.id"
        assert candidate.trust == Trust.HIGH

    def test_generic_fields_excluded(self) -> None:
        assert "total" in EXCLUDED_FIELDS
        spec = [
            *_USERS_SPEC,
            _ep(
                "POST",
                "/x",
                tags=["x"],
                body={"type": "object", "properties": {"total": {"type": "number"}}},
            ),
        ]
        graph = build_graph(spec)
        assert not any(
            e.source == "POST /orders" and e.target == "POST /x" for e in graph.data_flow_edges()
        )

    def test_multiple_candidates_all_kept(self) -> None:
        """Same topology edge with two different field candidates: both
        survive (no strongest-edge override)."""
        spec = [
            _ep(
                "POST",
                "/a",
                tags=["a"],
                rschemas={
                    "201": {
                        "type": "object",
                        "properties": {"xid": {"type": "integer"}, "yid": {"type": "integer"}},
                    }
                },
            ),
            _ep(
                "POST",
                "/b",
                tags=["b"],
                body={
                    "type": "object",
                    "properties": {"xid": {"type": "integer"}, "yid": {"type": "integer"}},
                },
            ),
        ]
        graph = build_graph(spec)
        edge = next(
            e for e in graph.data_flow_edges() if e.source == "POST /a" and e.target == "POST /b"
        )
        assert len(edge.binding_candidates) == 2


class TestR2:
    def test_creator_to_consumer_declared(self) -> None:
        graph = build_graph(_USERS_SPEC)
        edge = next(
            e
            for e in graph.data_flow_edges()
            if e.source == "POST /users" and e.target == "POST /orders"
        )
        candidate = edge.binding_candidates[0]
        assert candidate.rule == "R2"
        assert candidate.mode == "DECLARED"
        assert candidate.consumer == "body.user_id"
        assert candidate.producer == "response.id"  # hint, not proven


class TestR3R4:
    def test_same_path_lifecycle(self) -> None:
        graph = build_graph(_USERS_SPEC)
        adjacency = graph.lifecycle_adjacency()
        # GET /orders and POST /orders share the path -> R3 adjacency.
        assert "POST /users" not in adjacency.get("GET /orders", set())
        assert "POST /orders" in adjacency.get("GET /orders", set())
        pairs = {
            frozenset((e.source, e.target)) for e in graph.edges if e.kind == RelationKind.LIFECYCLE
        }
        # GET /orders and POST /orders share the path (R3).
        assert frozenset(("GET /orders", "POST /orders")) in pairs

    def test_nested_ancestor(self) -> None:
        spec = [
            *_USERS_SPEC,
            _ep("GET", "/users", tags=["users"]),
            _ep("GET", "/users/{id}/orders", tags=["orders"]),
        ]
        graph = build_graph(spec)
        pairs = {
            frozenset((e.source, e.target)) for e in graph.edges if e.kind == RelationKind.LIFECYCLE
        }
        # /users/{id}/orders child of /users/{id} (nearest EXISTING
        # ancestor — /users is not the nearest since /users/{id} exists).
        assert frozenset(("DELETE /users/{id}", "GET /users/{id}/orders")) in pairs


class TestR6Ingestion:
    def test_grounding_preserves_module_direction(self) -> None:
        r6 = R6Edge(
            source="users",
            target="orders",
            kind="DATA_FLOW",
            clause_id="C1",
            evidence="订单使用用户",
            score=2.0,
        )
        graph = build_graph(_USERS_SPEC, r6_edges=[r6])
        edge = next(
            e
            for e in graph.data_flow_edges()
            if e.source == "POST /users" and e.target == "POST /orders"
        )
        r6_candidates = [c for c in edge.binding_candidates if c.rule == "R6"]
        assert r6_candidates
        assert r6_candidates[0].mode == "DECLARED"
        assert r6_candidates[0].trust == Trust.MEDIUM
        assert r6_candidates[0].producer == "" and r6_candidates[0].consumer == ""

    def test_evidence_merged_candidates_distinct(self) -> None:
        """Re-adding the same topology edge merges evidence but keeps
        distinct candidates."""
        r6a = R6Edge(
            source="users",
            target="orders",
            kind="DATA_FLOW",
            clause_id="C1",
            evidence="a",
            score=2.0,
        )
        r6b = R6Edge(
            source="users",
            target="orders",
            kind="DATA_FLOW",
            clause_id="C2",
            evidence="b",
            score=2.0,
        )
        graph = build_graph(_USERS_SPEC, r6_edges=[r6a, r6b])
        edge = next(
            e
            for e in graph.data_flow_edges()
            if e.source == "POST /users" and e.target == "POST /orders"
        )
        r6_evidence = [ev for ev in edge.evidence if ev.get("rule") == "R6"]
        assert len(r6_evidence) == 2  # both clauses kept
        r6_candidates = [c for c in edge.binding_candidates if c.rule == "R6"]
        assert len(r6_candidates) == 1  # identical R6 no-value candidate merged


class TestIdentity:
    def test_duplicate_endpoint_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            build_graph([_ep("GET", "/users"), _ep("GET", "/users")])
