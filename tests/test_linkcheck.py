"""S4 tests: Gate1 closure, Gate3 legality, Gate2 fulfillment, metrics."""

from testagent.config.models import APIEndpoint
from testagent.pipeline.linkcheck import (
    check_contract_cases,
    gate1_closure,
    gate3_binds,
    match_endpoint,
)
from testagent.pipeline.links_fields import case_view_from_dict
from testagent.pipeline.links_graph import BindingCandidate, Edge, build_graph
from testagent.pipeline.pathplanner import plan_paths


def _ep(method, path, *, tags=None, rschemas=None, body=None, params=None):
    request_body = None
    if body is not None:
        request_body = {"media_type": "application/json", "schema": body}
    return APIEndpoint(
        method=method,
        path=path,
        tags=tags or [],
        response_schemas=rschemas or {},
        request_body=request_body,
        parameters=params or [],
    )


_SPEC = [
    _ep(
        "POST",
        "/users",
        tags=["users"],
        rschemas={"201": {"type": "object", "properties": {"id": {"type": "integer"}}}},
        body={"type": "object", "properties": {"name": {"type": "string"}}},
    ),
    _ep(
        "POST",
        "/orders",
        tags=["orders"],
        body={"type": "object", "properties": {"id": {"type": "integer"}}},
    ),
    _ep("GET", "/orders", tags=["orders"]),
]


class TestEndpointMatcher:
    def test_template_exact(self) -> None:
        spec_paths = {"/orders/{id}": {"GET"}, "/orders": {"GET"}}
        assert match_endpoint("GET /orders/{id}", spec_paths) == ("GET", "/orders/{id}")

    def test_no_substring_match(self) -> None:
        spec_paths = {"/orders/{id}": {"GET"}}
        assert match_endpoint("GET /orders-extra", spec_paths) is None
        assert match_endpoint("GET /orders", spec_paths) is None

    def test_keeps_braces(self) -> None:
        spec_paths = {"/orders/{id}": {"GET"}}
        assert match_endpoint("GET /orders/{id}", spec_paths) == ("GET", "/orders/{id}")

    def test_wrappers_stripped(self) -> None:
        spec_paths = {"/orders": {"POST"}}
        assert match_endpoint('POST ("/orders")', spec_paths) == ("POST", "/orders")

    def test_not_from_binds_text_alone(self) -> None:
        """binds/preconditions/expected mentions don't prove execution —
        matcher is used on steps only (enforced by callers)."""


class TestGate1:
    def test_store_then_consume_closed(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-1",
                "preconditions": ["authenticated with <TOKEN>"],
                "steps": [
                    "POST /users; store <USER_ID>",
                    'POST /orders with body {"id": "<USER_ID>"}',
                ],
                "expected_results": ["201"],
            }
        )
        facts = gate1_closure(case)
        assert facts["orphans"] == []
        assert facts["forward_references"] == []

    def test_consume_before_produce_forward(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-2",
                "steps": [
                    "GET /orders/<ORDER_ID>",
                    "POST /orders; store <ORDER_ID>",
                ],
                "expected_results": ["200"],
            }
        )
        facts = gate1_closure(case)
        assert facts["forward_references"] == ["ORDER_ID"]

    def test_orphan(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-3",
                "steps": ["GET /orders"],
                "expected_results": ["returns <ORDER_ID>"],
            }
        )
        facts = gate1_closure(case)
        assert facts["orphans"] == ["ORDER_ID"]

    def test_external_ids_available(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-4",
                "preconditions": ["<TOKEN> ready"],
                "steps": ["GET /orders with Authorization Bearer <TOKEN>"],
                "expected_results": ["200"],
            }
        )
        facts = gate1_closure(case)
        assert facts["orphans"] == []
        assert "TOKEN" in facts["external"]

    def test_precondition_placeholder_is_available(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-5",
                "preconditions": ["existing user <USER_ID>"],
                "steps": ["GET /orders/<USER_ID>"],
                "expected_results": ["200"],
            }
        )
        assert gate1_closure(case)["orphans"] == []


class TestGate3:
    def test_valid_bind_passes(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-6",
                "steps": ["POST /users; store <USER_ID>"],
                "expected_results": ["201"],
                "binds": {
                    "USER_ID": {
                        "producer": "POST /users response.id",
                        "consumer": "POST /orders body.id",
                    }
                },
            }
        )
        assert gate3_binds(case, _SPEC) == []

    def test_unknown_endpoint(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-7",
                "steps": [],
                "expected_results": [],
                "binds": {"X": {"producer": "POST /nope response.id"}},
            }
        )
        violations = gate3_binds(case, _SPEC)
        assert violations[0]["class"] == "unknown_endpoint"

    def test_unknown_response_field_when_schema_known(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-8",
                "steps": [],
                "expected_results": [],
                "binds": {"X": {"producer": "POST /users response.wizard"}},
            }
        )
        assert gate3_binds(case, _SPEC)[0]["class"] == "unknown_response_field"

    def test_malformed_producer(self) -> None:
        case = case_view_from_dict(
            {
                "id": "TC-9",
                "steps": [],
                "expected_results": [],
                "binds": {"X": {"producer": "not-a-binding"}},
            }
        )
        assert gate3_binds(case, _SPEC)[0]["class"] == "MALFORMED_BIND"

    def test_producer_only_is_audit_not_violation(self) -> None:
        """producer-only binds don't fail Gate3 (audit-only statements)."""
        case = case_view_from_dict(
            {
                "id": "TC-10",
                "steps": [],
                "expected_results": [],
                "binds": {"X": {"producer": "POST /users response.id"}},
            }
        )
        assert gate3_binds(case, _SPEC) == []


class TestGate2AndMetrics:
    def _plan(self):
        graph = build_graph(_SPEC)
        return plan_paths(graph, _SPEC, max_planned=8, l3b_budget=3)

    def test_fulfilled_path_covered(self) -> None:
        plan = self._plan()
        contract = plan["candidate"][0]
        assert contract.static_class == "executable"
        case = case_view_from_dict(
            {
                "id": "TC-F1",
                "path_id": contract.path_id,
                "source_stage": "l3b",
                "preconditions": [],
                "steps": [
                    "POST /users; store <USER_ID>",
                    'POST /orders with body {"id": "<USER_ID>"}',
                ],
                "expected_results": ["201"],
                "binds": {
                    "USER_ID": {
                        "producer": "POST /users response.id",
                        "consumer": "POST /orders body.id",
                    }
                },
            }
        )
        report = check_contract_cases(
            plan["candidate"], [case], _SPEC, attempted=[contract.path_id]
        )
        coverage = report.paths[contract.path_id]
        assert coverage.state == "COVERED"
        assert report.metrics()["pair_rate"] == 1.0
        assert report.outcomes["TC-F1"].grade == "INTEGRATION"

    def test_no_case_failed_unattempted(self) -> None:
        plan = self._plan()
        report = check_contract_cases(plan["candidate"], [], _SPEC)
        assert all(p.state == "UNATTEMPTED" for p in report.paths.values())
        assert report.metrics()["pair_rate"] == 0.0

    def test_attempted_without_success_failed(self) -> None:
        plan = self._plan()
        contract = plan["candidate"][0]
        bad_case = case_view_from_dict(
            {
                "id": "TC-B1",
                "path_id": contract.path_id,
                "source_stage": "l3b",
                "steps": ["GET /orders"],  # wrong sequence
                "expected_results": ["200"],
            }
        )
        report = check_contract_cases(
            plan["candidate"], [bad_case], _SPEC, attempted=[contract.path_id]
        )
        assert report.paths[contract.path_id].state == "FAILED"
        assert report.metrics()["pair_rate"] == 0.0

    def test_field_invention_rejected(self) -> None:
        plan = self._plan()
        contract = plan["candidate"][0]
        case = case_view_from_dict(
            {
                "id": "TC-R1",
                "path_id": contract.path_id,
                "source_stage": "l3b",
                "steps": ["POST /users; store <USER_ID>"],
                "expected_results": ["201"],
                "binds": {"USER_ID": {"producer": "POST /users response.wizard_id"}},
            }
        )
        report = check_contract_cases(
            plan["candidate"], [case], _SPEC, attempted=[contract.path_id]
        )
        assert report.outcomes["TC-R1"].grade == "REJECTED"

    def test_half_path_cases_do_not_combine(self) -> None:
        plan = self._plan()
        contract = plan["candidate"][0]
        if len(contract.business_hops) < 2:
            return  # single-hop contract: stitching test not applicable
        producer_case = case_view_from_dict(
            {
                "id": "TC-H1",
                "path_id": contract.path_id,
                "source_stage": "l3b",
                "steps": ["POST /users; store <USER_ID>"],
                "expected_results": ["201"],
            }
        )
        consumer_case = case_view_from_dict(
            {
                "id": "TC-H2",
                "path_id": contract.path_id,
                "source_stage": "l3b",
                "steps": ['POST /orders with body {"id": "<USER_ID>"}'],
                "expected_results": ["201"],
                "binds": {
                    "USER_ID": {
                        "producer": "POST /users response.id",
                        "consumer": "POST /orders body.id",
                    }
                },
            }
        )
        report = check_contract_cases(
            plan["candidate"], [producer_case, consumer_case], _SPEC, attempted=[contract.path_id]
        )
        # Each case alone misses part of the sequence → not COVERED.
        assert report.paths[contract.path_id].state != "COVERED"

    def test_na_metrics_when_no_contracts(self) -> None:
        report = check_contract_cases([], [], _SPEC)
        assert report.metrics()["pair_rate"] == "n/a"
        assert report.metrics()["path_rate"] == "n/a"
