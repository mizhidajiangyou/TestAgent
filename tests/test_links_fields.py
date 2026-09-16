"""S1b tests: typed accessors, CaseView contract, field views (§3.1/§3.3)."""

import pytest

from testagent.config.models import APIEndpoint
from testagent.pipeline.links_fields import (
    FieldAddress,
    case_view_from_dict,
    case_view_from_model,
    request_fields,
    response_fields,
)


def _ep(**kwargs) -> APIEndpoint:
    defaults: dict = {"method": "GET", "path": "/x"}
    defaults.update(kwargs)
    return APIEndpoint(**defaults)


class TestCaseView:
    def test_from_model_reads_typed_fields(self) -> None:
        from testagent.config.models import TestCase, TestPriority, TestType

        tc = TestCase(
            id="TC-1",
            title="t",
            description="d",
            endpoint=_ep(),
            test_type=TestType.FUNCTIONAL,
            priority=TestPriority.HIGH,
            preconditions=["p"],
            steps=["s1", "s2"],
            expected_results=["e"],
            path_id="Pabc123",
            source_stage="l3b",
            binds={"ID": {"producer": "POST /x response.id"}},
        )
        view = case_view_from_model(tc)
        assert view.case_id == "TC-1"
        assert view.steps == ("s1", "s2")
        assert view.path_id == "Pabc123"
        assert view.source_stage == "l3b"
        assert view.binds == {"ID": {"producer": "POST /x response.id"}}

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            case_view_from_dict({"id": "", "steps": []})

    def test_wrong_section_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="steps"):
            case_view_from_dict({"id": "X", "steps": "not-a-list"})

    def test_missing_cleanup_is_empty(self) -> None:
        view = case_view_from_dict({"id": "X", "steps": ["s"]})
        assert view.cleanup == ()


class TestFieldAddress:
    def test_request_alias_is_body(self) -> None:
        addr = FieldAddress.parse("request.userId")
        assert addr is not None and addr.location == "body"

    def test_params_distinct_from_body(self) -> None:
        assert FieldAddress.parse("params.id") != FieldAddress.parse("body.id")

    def test_invalid_rejected(self) -> None:
        assert FieldAddress.parse("header.x") is None
        assert FieldAddress.parse("response.") is None
        assert FieldAddress.parse("response.bad-name") is None


class TestResponseFields:
    def test_top_level_and_envelope_expanded(self) -> None:
        ep = _ep(
            response_schemas={
                "200": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "data": {
                            "type": "object",
                            "properties": {"total": {"type": "integer"}},
                        },
                    },
                }
            }
        )
        views = response_fields(ep)
        addresses = {v.address: v for v in views}
        assert "response.id" in addresses
        assert "response.total" in addresses
        assert addresses["response.total"].envelope == "data"
        assert addresses["response.id"].status_source == "200"

    def test_deep_structures_unknown(self) -> None:
        """Nested objects beyond the envelope layer are NOT flattened."""
        ep = _ep(
            response_schemas={
                "200": {
                    "type": "object",
                    "properties": {
                        "meta": {
                            "type": "object",
                            "properties": {"deep": {"type": "string"}},
                        }
                    },
                }
            }
        )
        views = response_fields(ep)
        assert {v.address for v in views} == {"response.meta"}
        # The deep field is not invented.
        assert "response.deep" not in {v.address for v in views}

    def test_empty_schema_is_honest(self) -> None:
        assert response_fields(_ep()) == []


class TestRequestFields:
    def test_body_and_params_both_listed(self) -> None:
        ep = _ep(
            method="POST",
            path="/users",
            request_body={
                "media_type": "application/json",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
            parameters=[{"name": "verbose", "in": "query", "schema": {"type": "boolean"}}],
        )
        views = request_fields(ep)
        addresses = {v.address for v in views}
        assert "body.name" in addresses
        assert "params.verbose" in addresses
