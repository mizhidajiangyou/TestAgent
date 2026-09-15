"""T5 tests: obligation registry, spec-fact registration, hook quota (§3.1)."""

import threading

import pytest

from testagent.config.models import APIEndpoint
from testagent.engine.truncation import GenericHooks, TruncationPolicy
from testagent.pipeline.obligations import (
    BindingBasis,
    Obligation,
    ObligationRegistry,
    ObligationSource,
    register_spec_obligations,
)


def _ep(
    method: str,
    path: str,
    responses: list[str] | None = None,
    body_props: dict | None = None,
    body_required: list[str] | None = None,
) -> APIEndpoint:
    request_body = None
    if body_props is not None:
        request_body = {
            "media_type": "application/json",
            "schema": {
                "type": "object",
                "properties": body_props,
                "required": body_required or [],
            },
        }
    return APIEndpoint(
        method=method, path=path, request_body=request_body, responses=responses or []
    )


class TestRegistry:
    def test_register_and_account(self) -> None:
        reg = ObligationRegistry()
        ids = reg.register_requirement_acs("REQ-001", ["创建成功返回 201", "重名返回 409"])
        assert ids == ["REQ-001-AC1", "REQ-001-AC2"]
        reg.cover("REQ-001-AC1", "TC-001")
        reg.gap_blocked("REQ-001-AC2", "spec_missing_endpoint:/auth/login")
        assert reg.is_complete()
        summary = reg.summary()
        assert summary == {"total": 2, "covered": 1, "gap_blocked": 1, "uncovered": 0}

    def test_uncovered_keeps_incomplete(self) -> None:
        reg = ObligationRegistry()
        reg.register_requirement_acs("REQ-001", ["AC one"])
        assert not reg.is_complete()
        assert reg.summary()["uncovered"] == 1

    def test_gap_blocked_requires_reason(self) -> None:
        reg = ObligationRegistry()
        reg.register_requirement_acs("REQ-001", ["AC one"])
        with pytest.raises(ValueError, match="gap reason"):
            reg.gap_blocked("REQ-001-AC1", "")

    def test_duplicate_id_rejected(self) -> None:
        reg = ObligationRegistry()
        ob = Obligation(
            id="X-1",
            requirement_id=None,
            statement="s",
            source=ObligationSource.SPEC,
        )
        reg.register(ob)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(ob)

    def test_unknown_obligation_rejected(self) -> None:
        reg = ObligationRegistry()
        with pytest.raises(KeyError):
            reg.cover("NOPE", "TC-001")

    def test_concurrent_accounting_is_lock_guarded(self) -> None:
        reg = ObligationRegistry()
        ids = reg.register_requirement_acs("REQ-001", [f"AC {i}" for i in range(50)])

        def worker(k: int) -> None:
            reg.cover(ids[k % len(ids)], f"TC-{k:03d}")

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert reg.summary()["covered"] == 50
        covered_by = [s for s in reg.states() if s.covered_by]
        assert sum(len(s.covered_by) for s in covered_by) == 100

    def test_report_renders_states(self) -> None:
        reg = ObligationRegistry()
        reg.register_requirement_acs("REQ-002", ["登录成功", "连续失败锁定"])
        reg.gap_blocked("REQ-002-AC1", "spec_missing_endpoint:/auth/login")
        report = reg.render_report()
        assert "gap_blocked" in report
        assert "spec_missing_endpoint:/auth/login" in report
        assert "REQ-002-AC2" in report


class TestSpecObligations:
    def test_required_bound_format_error_facts(self) -> None:
        endpoints = [
            _ep(
                "POST",
                "/users",
                responses=["201", "400", "409"],
                body_props={
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "age": {"type": "integer", "minimum": 0},
                },
                body_required=["name", "email"],
            ),
            _ep("GET", "/users", responses=["200"]),
        ]
        obligations = register_spec_obligations(endpoints)
        ids = {o.id for o in obligations}
        assert "SPEC-POST_users-required_name" in ids
        assert "SPEC-POST_users-required_email" in ids
        assert "SPEC-POST_users-format_email" in ids
        assert "SPEC-POST_users-minimum_age" in ids
        assert "SPEC-POST_users-error_400" in ids
        assert "SPEC-POST_users-error_409" in ids
        # GET /users only has a success response -> no facts from it
        assert not any(o.id.startswith("SPEC-GET_users") for o in obligations)
        assert all(o.source is ObligationSource.SPEC for o in obligations)
        assert all(o.binding_basis is BindingBasis.EXPLICIT for o in obligations)

    def test_registry_accepts_spec_batch(self) -> None:
        endpoints = [
            _ep("POST", "/users", responses=["400"], body_props={"name": {"type": "string"}})
        ]
        reg = ObligationRegistry()
        reg.register_many(register_spec_obligations(endpoints))
        assert reg.summary()["total"] >= 1


class TestExpectedForHook:
    def test_hook_overrides_quota(self) -> None:
        """expected_for replaces the flat per-endpoint quota (T5/T7 seam)."""
        calls: list[list[str]] = []

        def expected_for(scope: list[str]) -> dict[str, int]:
            calls.append(list(scope))
            return {"GET /users": 3}

        hooks = GenericHooks(
            extract=lambda raw: None,
            salvage=lambda raw: None,
            scope_key=lambda item: "",
            dedup_key=lambda item, key: key,
            scope_item_key=lambda item: str(item),
            build_reask=lambda a, b, c, d: "reask",
            expected_for=expected_for,
        )
        assert hooks.expected_for is not None
        assert hooks.expected_for(["GET /users", "POST /users"]) == {"GET /users": 3}
        assert calls == [["GET /users", "POST /users"]]
        # Legacy default stays available for hosts without the hook.
        assert TruncationPolicy().default_expected_cases_per_endpoint >= 1

    def test_default_hooks_have_no_hook(self) -> None:
        hooks = GenericHooks(
            extract=lambda raw: None,
            salvage=lambda raw: None,
            scope_key=lambda item: "",
            dedup_key=lambda item, key: key,
            scope_item_key=lambda item: str(item),
            build_reask=lambda a, b, c, d: "reask",
        )
        assert hooks.expected_for is None
