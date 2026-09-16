"""S3 tests: path enumeration, identity, pool, static class (§5)."""

import pytest

from testagent.config.models import APIEndpoint
from testagent.pipeline.links_graph import BindingCandidate, Edge, build_graph
from testagent.pipeline.pathplanner import (
    PathIdentityCollision,
    PlanningLimitExceeded,
    canonical_payload_bytes,
    plan_paths,
)


def _ep(method: str, path: str, *, tags=None, rschemas=None, body=None):
    request_body = None
    if body is not None:
        request_body = {"media_type": "application/json", "schema": body}
    return APIEndpoint(
        method=method,
        path=path,
        tags=tags or [],
        response_schemas=rschemas or {},
        request_body=request_body,
    )


def _two_module_spec():
    spec = [
        _ep(
            "POST",
            "/users",
            tags=["users"],
            rschemas={"201": {"type": "object", "properties": {"id": {"type": "integer"}}}},
        ),
        _ep(
            "POST",
            "/orders",
            tags=["orders"],
            body={"type": "object", "properties": {"id": {"type": "integer"}}},
        ),
        _ep("GET", "/orders", tags=["orders"]),
        _ep("GET", "/users", tags=["users"]),
    ]
    return spec


class TestEnumeration:
    def test_cross_module_path_found(self) -> None:
        spec = _two_module_spec()
        graph = build_graph(spec)
        result = plan_paths(graph, spec)
        assert result["counts"]["planned_paths"] >= 1
        path = next(
            c
            for c in result["planned"]
            if "POST /users" in c.endpoints and "POST /orders" in c.endpoints
        )
        assert path.business_hops[0].kind == "DATA_FLOW"
        assert path.business_hops[0].candidate is not None

    def test_same_module_edges_are_not_business_hops(self) -> None:
        spec = _two_module_spec()
        graph = build_graph(spec)
        result = plan_paths(graph, spec)
        for contract in result["planned"]:
            for hop in contract.business_hops:
                assert hop.kind != "LIFECYCLE"


class TestIdentity:
    def test_path_id_stable_and_order_independent(self) -> None:
        """Reversing the input container order must not change path_ids."""
        spec = _two_module_spec()
        r1 = plan_paths(build_graph(spec), spec)
        r2 = plan_paths(build_graph(list(reversed(spec))), list(reversed(spec)))
        ids1 = {c.path_id for c in r1["planned"]}
        ids2 = {c.path_id for c in r2["planned"]}
        assert ids1 == ids2

    def test_different_binding_different_id(self) -> None:
        """Same endpoints/kind but different candidate -> different
        path_id (v15 §5.2)."""
        spec = [
            _ep(
                "POST",
                "/a",
                tags=["a"],
                rschemas={
                    "201": {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    }
                },
            ),
            _ep(
                "POST",
                "/b",
                tags=["b"],
                body={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                },
            ),
        ]
        graph = build_graph(spec)
        result = plan_paths(graph, spec)
        two_field_paths = [
            c
            for c in result["planned"]
            if len(c.business_hops) == 1 and c.business_hops[0].kind == "DATA_FLOW"
        ]
        assert len({c.path_id for c in two_field_paths}) == 2  # x and y variants

    def test_canonical_bytes_shape(self) -> None:
        data = canonical_payload_bytes({"b": 1, "a": 2})
        assert data == b'{"a":2,"b":1}'


class TestPool:
    def test_pool_arithmetic(self) -> None:
        spec = _two_module_spec()
        graph = build_graph(spec)
        result = plan_paths(graph, spec, max_planned=1, l3b_budget=1)
        assert result["counts"]["candidate_paths"] == 1
        assert all(d["reason"] == "candidate_cap" for d in result["not_materialized"])
        assert len(result["selected"]) == 1
        assert all(d["reason"] == "l3b_unit_budget" for d in result["deferred"])

    def test_priority_seeding_prefers_concrete_consumer(self) -> None:
        spec = [
            # R6-style no-value edge (via manual edge) and R1-style concrete edge
            _ep(
                "POST",
                "/a",
                tags=["a"],
                rschemas={"201": {"type": "object", "properties": {"k": {"type": "integer"}}}},
            ),
            _ep(
                "POST",
                "/b",
                tags=["b"],
                body={"type": "object", "properties": {"k": {"type": "integer"}}},
            ),
            _ep(
                "POST",
                "/c",
                tags=["c"],
                body={"type": "object", "properties": {"other": {"type": "string"}}},
            ),
        ]
        graph = build_graph(spec)
        # manual no-value DATA_FLOW a->c (simulating R6 ungrounded values)
        graph.add_edge(
            Edge(
                source="POST /a",
                target="POST /c",
                kind="DATA_FLOW",
                evidence=[{"rule": "R6"}],
                binding_candidates=[
                    BindingCandidate(
                        producer="", consumer="", mode="DECLARED", trust="MEDIUM", rule="R6"
                    )
                ],
            )
        )
        result = plan_paths(graph, spec, max_planned=8, l3b_budget=1, seeding="priority")
        selected = result["selected"][0]
        hops = [h.kind for h in selected.business_hops]
        assert "DATA_FLOW" in hops
        # the selected path's first hop must have a concrete consumer
        first_df = next(h for h in selected.business_hops if h.kind == "DATA_FLOW")
        assert first_df.candidate is not None and first_df.candidate.consumer

    def test_delete_only_terminal(self) -> None:
        spec = [
            _ep("DELETE", "/users/{id}", tags=["users"]),
            _ep(
                "POST",
                "/orders",
                tags=["orders"],
                body={"type": "object", "properties": {"user_id": {"type": "integer"}}},
            ),
        ]
        graph = build_graph(spec)
        # R2 binds POST /users creator — absent; DELETE has request field
        # user_id? No. Force an edge DELETE -> POST to test terminal rule.
        graph.add_edge(
            Edge(
                source="DELETE /users/{id}",
                target="POST /orders",
                kind="DATA_FLOW",
                evidence=[{"rule": "R6"}],
                binding_candidates=[
                    BindingCandidate(
                        producer="", consumer="", mode="DECLARED", trust="MEDIUM", rule="R6"
                    )
                ],
            )
        )
        result = plan_paths(graph, spec)
        assert result["planned"] == []
        assert result["invalid_attempts_by_reason"].get("delete_not_terminal", 0) >= 1


class TestStaticClass:
    def test_exact_all_hops_executable(self) -> None:
        spec = _two_module_spec()
        graph = build_graph(spec)
        result = plan_paths(graph, spec)
        contract = next(
            c for c in result["planned"] if c.endpoints == ("POST /users", "POST /orders")
        )
        assert contract.static_class == "executable"

    def test_declared_hop_spec_blind(self) -> None:
        spec = [
            _ep("POST", "/a", tags=["a"]),
            _ep("POST", "/b", tags=["b"]),
        ]
        graph = build_graph(spec)
        graph.add_edge(
            Edge(
                source="POST /a",
                target="POST /b",
                kind="DATA_FLOW",
                evidence=[{"rule": "R6"}],
                binding_candidates=[
                    BindingCandidate(
                        producer="", consumer="", mode="DECLARED", trust="MEDIUM", rule="R6"
                    )
                ],
            )
        )
        result = plan_paths(graph, spec)
        assert result["planned"], "schema-missing spec still plans (INVALID ≠ schema-missing)"
        assert all(c.static_class == "spec_blind" for c in result["planned"])


class TestLimits:
    def test_collision_hard_fail(self) -> None:
        """Two distinct payloads mapping to the same short id must raise."""
        from testagent.pipeline import pathplanner as pp

        original = pp.hashlib.sha256

        class _FakeDigest:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def hexdigest(self) -> str:
                return ("deadbeef" * 8)[:64] if b"endpoints" in self._data else "0000000" + "0" * 57

        def fake_sha256(data: bytes):
            return original(data)  # keep real behavior; collision test via monkeypatch below

        # Simulate collision by forcing the digest function to a constant.
        def constant_sha256(data: bytes):
            class D:
                def hexdigest(self) -> str:
                    return "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

            return D()

        pp.hashlib = type("h", (), {"sha256": staticmethod(constant_sha256)})
        try:
            spec = [
                _ep(
                    "POST",
                    "/a",
                    tags=["a"],
                    rschemas={
                        "201": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        }
                    },
                ),
                _ep(
                    "POST",
                    "/b",
                    tags=["b"],
                    body={
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                ),
            ]
            graph = build_graph(spec)
            with pytest.raises(PathIdentityCollision):
                plan_paths(graph, spec)
        finally:
            pp.hashlib = type("h", (), {"sha256": staticmethod(original)})

    def test_enum_guard(self) -> None:
        from testagent.pipeline import pathplanner as pp

        original = pp.MAX_ENUMERATED_VARIANTS
        pp.MAX_ENUMERATED_VARIANTS = 1
        try:
            spec = [
                _ep(
                    "POST",
                    "/a",
                    tags=["a"],
                    rschemas={
                        "201": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                        }
                    },
                ),
                _ep(
                    "POST",
                    "/b",
                    tags=["b"],
                    body={
                        "type": "object",
                        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                    },
                ),
            ]
            graph = build_graph(spec)
            with pytest.raises(PlanningLimitExceeded):
                plan_paths(graph, spec)
        finally:
            pp.MAX_ENUMERATED_VARIANTS = original
