"""T10 tests: Gate-A closure, Gate-B binding contract, grading, metrics."""

from testagent.config.models import APIEndpoint
from testagent.pipeline.executability import (
    gate_a_closure,
    gate_b_bindings,
    grade_case,
    normalize_placeholder,
    placeholder_closure_metrics,
)

_SPEC = [
    APIEndpoint(
        method="POST",
        path="/users",
        responses=["201", "400"],
        request_body={
            "media_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
                "required": ["name"],
            },
        },
        response_schemas={
            "201": {
                "type": "object",
                "properties": {"id": {"type": "integer"}},
                "required": ["id"],
            }
        },
    ),
    APIEndpoint(
        method="DELETE",
        path="/users/{id}",
        responses=["204", "404"],
    ),
]


def _case(**overrides):
    base = {
        "id": "TC-001",
        "title": "integration flow",
        "description": "verify user lifecycle",
        "preconditions": ["User is authenticated with <TOKEN>"],
        "steps": [
            'POST /users with body {"name": "n"}; store as <USER_ID>',
            "DELETE /users/<USER_ID>",
        ],
        "expected_results": ["Status 201", "Status 204"],
        "binds": {
            "USER_ID": {
                "producer": "POST /users response.body.id",
                "consumer": "DELETE /users/{id} path.id",
            }
        },
    }
    base.update(overrides)
    return base


class TestGateA:
    def test_proper_flow_is_closed(self) -> None:
        orphans, forwards, rate = gate_a_closure(_case())
        assert orphans == [] and forwards == []
        assert rate == "1.0"

    def test_orphan_detected(self) -> None:
        case = _case(
            steps=['POST /users with body {"name": "n"}'],
            expected_results=["user <ORDER_ID> created"],
        )
        orphans, _, _ = gate_a_closure(case)
        assert orphans == ["ORDER_ID"]

    def test_forward_reference_detected(self) -> None:
        case = _case(
            steps=[
                "GET /orders/<ORDER_ID> returns the order",
                "POST /orders creates order; store as <ORDER_ID>",
            ],
            expected_results=["Status 200"],
            binds={"ORDER_ID": {"producer": "POST /orders response.body.id"}},
        )
        _, forwards, _ = gate_a_closure(case)
        assert forwards == ["ORDER_ID"]

    def test_exempt_placeholders_available(self) -> None:
        case = _case(
            preconditions=["authenticated via <TOKEN> and <API_KEY>"],
            steps=["GET /users with Authorization: Bearer <TOKEN>"],
            expected_results=["Status 200"],
            binds={},
        )
        orphans, forwards, _ = gate_a_closure(case)
        assert orphans == [] and forwards == []

    def test_assignment_form_produces(self) -> None:
        case = _case(
            steps=[
                "POST /users; <USER_ID> = response.body.id",
                "DELETE /users/<USER_ID>",
            ],
        )
        orphans, forwards, _ = gate_a_closure(case)
        assert orphans == [] and forwards == []

    def test_cleanup_step_cannot_produce(self) -> None:
        """A placeholder first appearing inside a cleanup step is never
        produced (cleanup consumes, it does not create)."""
        case = _case(
            steps=[
                "POST /users creates user",
                "cleanup: DELETE /orders/<ORDER_ID> to reset state",
            ],
            expected_results=["Status 201"],
            binds={},
        )
        orphans, _, _ = gate_a_closure(case)
        assert "ORDER_ID" in orphans

    def test_normalize_placeholder_aliases(self) -> None:
        text = normalize_placeholder("use {{uuid}} and <uuid> and <token>")
        assert "<RUN_ID>" in text and "<TOKEN>" in text


class TestGateB:
    def test_valid_bind_passes(self) -> None:
        violations, blind = gate_b_bindings(_case(), _SPEC)
        assert violations == [] and blind == []

    def test_producer_endpoint_not_found(self) -> None:
        case = _case(binds={"USER_ID": {"producer": "POST /nope response.body.id"}})
        violations, _ = gate_b_bindings(case, _SPEC)
        assert violations[0]["class"] == "producer_endpoint_not_found"

    def test_producer_field_rejected_when_schema_known(self) -> None:
        case = _case(binds={"USER_ID": {"producer": "POST /users response.body.wizard_id"}})
        violations, _ = gate_b_bindings(case, _SPEC)
        assert violations[0]["class"] == "producer_field_rejected"

    def test_spec_blind_not_a_violation(self) -> None:
        """POST /users declares no response schema -> producer side is
        recorded as spec_blind, not rejected; the consumer side (path param
        `id` known from the endpoint path) still validates."""
        blind_spec = [
            APIEndpoint(method="POST", path="/users", responses=["201"]),
            APIEndpoint(method="DELETE", path="/users/{id}", responses=["204"]),
        ]
        case = _case()
        violations, blind = gate_b_bindings(case, blind_spec)
        assert violations == []
        assert blind == ["USER_ID:producer"]

    def test_consumer_endpoint_not_found(self) -> None:
        case = _case(
            binds={
                "USER_ID": {
                    "producer": "POST /users response.body.id",
                    "consumer": "DELETE /nope path.id",
                }
            }
        )
        violations, _ = gate_b_bindings(case, _SPEC)
        assert violations[0]["class"] == "consumer_endpoint_not_found"

    def test_consumer_field_rejected(self) -> None:
        case = _case(
            binds={
                "USER_ID": {
                    "producer": "POST /users response.body.id",
                    "consumer": "DELETE /users/{id} path.wizard",
                }
            }
        )
        violations, _ = gate_b_bindings(case, _SPEC)
        assert violations[0]["class"] == "consumer_field_rejected"

    def test_malformed_bind(self) -> None:
        case = _case(binds={"USER_ID": {"nope": "x"}})
        violations, _ = gate_b_bindings(case, _SPEC)
        assert violations[0]["class"] == "malformed_bind"


class TestGrading:
    def test_integration_grade(self) -> None:
        assert grade_case(_case(), _SPEC).grade == "INTEGRATION"

    def test_orphan_is_draft(self) -> None:
        case = _case(expected_results=["user <ORDER_ID> created"], binds={})
        assert grade_case(case, _SPEC).grade == "DRAFT"

    def test_field_rejection_is_rejected(self) -> None:
        """Consumer field invented against a KNOWN request schema (path
        param `id` is the only request field) -> REJECTED."""
        case = _case(
            binds={
                "USER_ID": {
                    "producer": "POST /users response.body.id",
                    "consumer": "DELETE /users/{id} path.wizard_id",
                }
            }
        )
        assert grade_case(case, _SPEC).grade == "REJECTED"


class TestMetrics:
    def test_empty_denominators_are_na(self) -> None:
        cases = [_case(steps=["GET /users"], binds={})]
        metrics = placeholder_closure_metrics(cases, _SPEC)
        assert metrics["setup_dependency_completeness"] == "n/a"

    def test_violations_counted(self) -> None:
        bad = _case(binds={"USER_ID": {"producer": "POST /nope response.body.id"}})
        metrics = placeholder_closure_metrics([_case(), bad], _SPEC)
        assert metrics["bind_violations"].get("producer_endpoint_not_found") == 1
        assert metrics["orphan_placeholders"] == 0
