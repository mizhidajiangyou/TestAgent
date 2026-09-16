"""T12a tests: L0/L1 assembly + lifecycle clustering (interface frozen)."""

from testagent.config.models import APIEndpoint
from testagent.pipeline.context_builder import ContextBuilder


def _eps() -> list[APIEndpoint]:
    return [
        APIEndpoint(method="GET", path="/users"),
        APIEndpoint(method="POST", path="/users"),
        APIEndpoint(method="DELETE", path="/users/{id}"),
        APIEndpoint(method="GET", path="/users/{id}/orders"),
        APIEndpoint(method="GET", path="/products"),
        APIEndpoint(method="POST", path="/products"),
        APIEndpoint(method="DELETE", path="/products/{id}"),
        APIEndpoint(method="GET", path="/orders"),
    ]


class TestL0:
    def test_detailed_signature_under_budget(self) -> None:
        l0 = ContextBuilder(_eps(), max_chars=8000).build_l0()
        assert "GET /users" in l0
        assert "module index" not in l0

    def test_char_budget_degrades_to_module_index(self) -> None:
        """The RENDERED character count (not endpoint count) is the only
        degradation criterion. The rich renderer is injected (layering),
        so the test supplies a verbose one to force the overflow."""

        def verbose(eps):
            return "\n".join(f"- {ep.method} {ep.path} " + "x" * 120 for ep in eps)

        l0 = ContextBuilder(_eps(), max_chars=400, signature_fn=verbose).build_l0()
        assert "module index" in l0
        assert "GET /users" in l0  # module view still lists keys

    def test_hard_truncation_marker(self) -> None:
        l0 = ContextBuilder(_eps(), max_chars=80).build_l0()
        assert "L0 truncated" in l0
        assert len(l0) <= 80


class TestL1:
    def test_neighbours_capped(self) -> None:
        relations = [
            ("GET /users", "GET /products", "data_flow"),
            ("POST /users", "GET /orders", "data_flow"),
            ("DELETE /users/{id}", "GET /users/{id}/orders", "precondition"),
            ("GET /orders", "POST /products", "side_effect"),
            ("GET /products", "GET /orders", "lifecycle"),
        ]
        l1 = ContextBuilder(_eps(), relations=relations).build_l1(["GET /users"])
        assert l1.count("[L1 neighbours]") <= 4

    def test_char_budget_respected(self) -> None:
        relations = [("GET /users", f"GET /r{i}", "data_flow") for i in range(10)]
        l1 = ContextBuilder(_eps(), relations=relations, l1_chars=120).build_l1(["GET /users"])
        assert len(l1) <= 120

    def test_same_resource_siblings_included(self) -> None:
        l1 = ContextBuilder(_eps()).build_l1(["GET /users"])
        assert "POST /users" in l1


class TestClusters:
    def test_same_resource_stays_together(self) -> None:
        clusters = ContextBuilder(_eps(), max_cluster=6).split_clusters()
        users = [c for c in clusters if any("users" in k for k in c)]
        user_keys = {k for c in users for k in c}
        # GET/POST /users + DELETE /users/{id} + nested orders-of-user all
        # share the /users resource -> one cluster.
        assert "GET /users" in user_keys and "POST /users" in user_keys
        assert "DELETE /users/{id}" in user_keys
        assert "GET /users/{id}/orders" in user_keys

    def test_cluster_size_capped(self) -> None:
        many = [APIEndpoint(method=m, path=f"/res{i}") for i in range(10) for m in ("GET", "POST")]
        clusters = ContextBuilder(many, max_cluster=6).split_clusters()
        assert all(len(c) <= 6 for c in clusters)

    def test_no_blind_split_of_lifecycle(self) -> None:
        """The gate: GET/POST/DELETE on one resource never land in different
        batches the way index slicing would."""
        clusters = ContextBuilder(_eps(), max_cluster=6).split_clusters()
        for cluster in clusters:
            if any(k == "GET /users" for k in cluster):
                assert "POST /users" in cluster
                assert "DELETE /users/{id}" in cluster
                break


class TestS5aAssembly:
    def _plan_two_module(self):
        from testagent.config.models import APIEndpoint
        from testagent.pipeline.links_graph import build_graph
        from testagent.pipeline.pathplanner import plan_paths

        spec = [
            APIEndpoint(
                method="POST",
                path="/users",
                tags=["users"],
                response_schemas={
                    "201": {"type": "object", "properties": {"id": {"type": "integer"}}}
                },
            ),
            APIEndpoint(
                method="POST",
                path="/orders",
                tags=["orders"],
                request_body={
                    "media_type": "application/json",
                    "schema": {"type": "object", "properties": {"id": {"type": "integer"}}},
                },
            ),
            APIEndpoint(method="GET", path="/orders", tags=["orders"]),
            APIEndpoint(method="GET", path="/users", tags=["users"]),
        ]
        graph = build_graph(spec)
        return plan_paths(graph, spec, max_planned=8, l3b_budget=3), spec

    def test_l2_skeleton_and_contract(self) -> None:
        planning, spec = self._plan_two_module()
        builder = ContextBuilder(spec)
        contract = planning["candidate"][0]
        l2 = builder.build_l2(contract)
        assert "[L2 path skeleton]" in l2
        assert all(f"- {e}" in l2 for e in contract.endpoints)
        assert "[L2 binding contract]" in l2
        assert "EXACT" in l2

    def test_l2_limit_recorded_not_sliced(self) -> None:
        planning, spec = self._plan_two_module()
        builder = ContextBuilder(spec)
        contract = planning["candidate"][0]
        # Simulate a >8-endpoint path: rebuild contract endpoints via object
        long_contract = type(contract)(
            path_id=contract.path_id,
            payload=contract.payload,
            canonical_bytes=contract.canonical_bytes,
            endpoints=tuple(f"POST /x{i}" for i in range(10)),
            business_hops=contract.business_hops,
            priority=contract.priority,
        )
        assert builder.build_l2(long_contract).startswith("context_shape_limit")

    def test_l3b_bundles_from_same_planning(self) -> None:
        from testagent.pipeline.context_builder import assemble_l3b_prompts

        planning, spec = self._plan_two_module()
        builder = ContextBuilder(spec)
        bundles = assemble_l3b_prompts(planning, builder)
        assert len(bundles) == len(planning["selected"])
        for bundle in bundles:
            assert bundle["path_id"]
            assert bundle["l0"]
            assert "L2 path skeleton" in bundle["l2"]
