"""T4 tests: deterministic consistency checks (fix-plan §3.3).

Pure functions — requirements text + parsed endpoints in, findings /
rendered reports out. The milestone-level "baseline yields exactly six
findings" result is judged at the quality milestone gate on a real run;
these tests pin the deterministic behavior per check.
"""

from dataclasses import FrozenInstanceError, replace

import pytest

from testagent.config.models import APIEndpoint
from testagent.pipeline.consistency import (
    ConflictPolicy,
    ConsistencyKind,
    Finding,
    RequirementRef,
    adjudicate_conflicts,
    classify_findings,
    extract_bound_rules,
    extract_endpoint_mentions,
    extract_field_mentions,
    extract_status_expectations,
    find_endpoint_gaps,
    find_field_gaps,
    find_status_conflicts,
    find_unsupported_rules,
    render_authoritative_table,
    render_gap_report,
)


def _ep(
    method: str,
    path: str,
    responses: list[str] | None = None,
    params: list[dict] | None = None,
    body_props: dict | None = None,
) -> APIEndpoint:
    request_body = None
    if body_props is not None:
        request_body = {
            "media_type": "application/json",
            "schema": {"type": "object", "properties": body_props},
        }
    return APIEndpoint(
        method=method,
        path=path,
        parameters=params or [],
        request_body=request_body,
        responses=responses or [],
    )


_USERS_SPEC = [
    _ep(
        "GET",
        "/users",
        responses=["200"],
        params=[{"name": "page", "in": "query", "schema": {"type": "integer", "minimum": 1}}],
    ),
    _ep(
        "POST",
        "/users",
        responses=["201", "400"],
        body_props={
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0},
            "email": {"type": "string"},
        },
    ),
]


class TestExtraction:
    def test_endpoint_mentions(self) -> None:
        text = "用户登录走 POST /auth/login，随后 GET /users 校验。"
        assert extract_endpoint_mentions(text) == ["POST /auth/login", "GET /users"]

    def test_field_mentions_backticked_only(self) -> None:
        text = "请求体必须带 `password` 与 `email`，password 不能为空。"
        assert extract_field_mentions(text) == ["password", "email"]

    def test_status_expectations_pair_trigger_with_code(self) -> None:
        text = "重复邮箱时返回 409；未认证返回 401。"
        pairs = extract_status_expectations(text)
        assert ("duplicate_email", "409") in pairs
        assert ("unauthorized", "401") in pairs

    def test_bound_rules_zh_and_en(self) -> None:
        text = "limit 最大 100；age 至少 0；锁定 10 分钟。"
        rules = extract_bound_rules(text)
        assert ("limit", "max", "100") in rules
        assert ("age", "min", "0") in rules
        assert any(kind == "duration" and field == "锁定" for field, kind, _ in rules)


class TestEndpointGaps:
    def test_missing_endpoint_reported_once_per_requirement(self) -> None:
        reqs = [
            RequirementRef("REQ-001", "创建用户 POST /users"),
            RequirementRef("REQ-002", "登录走 POST /auth/login，重复登录 POST /auth/login"),
        ]
        findings = find_endpoint_gaps(reqs, _USERS_SPEC)
        assert len(findings) == 1
        f = findings[0]
        assert f.kind is ConsistencyKind.SPEC_GAP
        assert f.subject == "/auth/login"
        assert f.requirement_ref == "REQ-002"

    def test_path_prefix_match_not_gap(self) -> None:
        """`/users/{id}` mentioned as `/users/42` style is not required here;
        an exact declared endpoint is never a gap."""
        reqs = [RequirementRef("REQ-001", "查询 GET /users")]
        assert find_endpoint_gaps(reqs, _USERS_SPEC) == []


class TestFieldGaps:
    def test_missing_field_reported(self) -> None:
        reqs = [
            RequirementRef(
                "REQ-002",
                "注册表单包含 `email` 和 `password` 字段",
            )
        ]
        findings = find_field_gaps(reqs, _USERS_SPEC)
        assert len(findings) == 1
        assert findings[0].subject == "password"

    def test_declared_field_not_reported(self) -> None:
        reqs = [RequirementRef("REQ-001", "创建用户带 `name` 字段")]
        assert find_field_gaps(reqs, _USERS_SPEC) == []


class TestStatusConflicts:
    def test_spec_vs_contract_conflict(self) -> None:
        """Duplicate email: spec declares 400, contract default is 409,
        requirement asserts 409 -> three-way disagreement."""
        reqs = [RequirementRef("REQ-001", "重复邮箱返回 409。涉及 POST /users")]
        findings = find_status_conflicts(reqs, _USERS_SPEC)
        assert len(findings) == 1
        f = findings[0]
        assert f.kind is ConsistencyKind.SPEC_REQ_CONFLICT
        assert f.subject == "duplicate_email(400 vs 409)"
        assert f.endpoint_ref == "POST /users"

    def test_requirement_vs_contract_conflict(self) -> None:
        """Unauthorized: requirement says 403, contract default 401, spec
        declares nothing for the mentioned endpoint."""
        spec = [_ep("GET", "/users", responses=["200"])]
        reqs = [RequirementRef("REQ-001", "未授权返回 403。涉及 GET /users")]
        findings = find_status_conflicts(reqs, spec)
        assert len(findings) == 1
        assert findings[0].kind is ConsistencyKind.SPEC_REQ_CONFLICT
        assert findings[0].subject == "unauthorized(401 vs 403)"

    def test_contract_only_when_spec_silent(self) -> None:
        spec = [_ep("GET", "/items", responses=["200"])]
        reqs = [RequirementRef("REQ-001", "资源不存在 GET /items 时按契约处理")]
        findings = find_status_conflicts(reqs, spec)
        assert all(f.kind is not ConsistencyKind.SPEC_REQ_CONFLICT for f in findings)

    def test_agreement_is_no_finding(self) -> None:
        spec = [_ep("POST", "/users", responses=["201", "409"])]
        reqs = [RequirementRef("REQ-001", "重复邮箱返回 409。涉及 POST /users")]
        assert find_status_conflicts(reqs, spec) == []


class TestUnsupportedRules:
    def test_unsupported_cap(self) -> None:
        reqs = [RequirementRef("REQ-001", "limit 上限 50")]
        findings = find_unsupported_rules(reqs, _USERS_SPEC)
        assert len(findings) == 1
        assert findings[0].kind is ConsistencyKind.UNSUPPORTED_REQUIREMENT
        assert "limit" in findings[0].subject

    def test_supported_cap_not_reported(self) -> None:
        spec = [
            _ep(
                "GET",
                "/items",
                params=[
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "maximum": 100},
                    }
                ],
            )
        ]
        reqs = [RequirementRef("REQ-001", "limit 最大 100")]
        assert find_unsupported_rules(reqs, spec) == []

    def test_lockout_duration_unsupported(self) -> None:
        reqs = [RequirementRef("REQ-002", "连续失败 5 次锁定 10 分钟")]
        findings = find_unsupported_rules(reqs, _USERS_SPEC)
        assert len(findings) == 1
        assert findings[0].kind is ConsistencyKind.UNSUPPORTED_REQUIREMENT
        assert "10 分钟" in findings[0].subject


class TestClassificationAndRendering:
    def test_classify_combines_all_checks(self) -> None:
        reqs = [
            RequirementRef(
                "REQ-002",
                "登录走 POST /auth/login，表单含 `password`；重复邮箱返回 409；"
                "锁定 10 分钟。涉及 POST /users",
            )
        ]
        findings = classify_findings(reqs, _USERS_SPEC)
        kinds = {f.kind for f in findings}
        assert ConsistencyKind.SPEC_GAP in kinds
        assert ConsistencyKind.SPEC_REQ_CONFLICT in kinds
        assert ConsistencyKind.UNSUPPORTED_REQUIREMENT in kinds

    def test_gap_report_renders_groups(self) -> None:
        reqs = [RequirementRef("REQ-002", "登录走 POST /auth/login")]
        report = render_gap_report(find_endpoint_gaps(reqs, _USERS_SPEC))
        assert "# Consistency gap report" in report
        assert "## spec_gap (1)" in report
        assert "/auth/login" in report

    def test_empty_report(self) -> None:
        assert "agree" in render_gap_report([])

    def test_authoritative_table_strict_marks_unresolved(self) -> None:
        reqs = [RequirementRef("REQ-001", "重复邮箱返回 409。涉及 POST /users")]
        findings = find_status_conflicts(reqs, _USERS_SPEC)
        table = render_authoritative_table(findings, ConflictPolicy.STRICT)
        assert "Policy: strict" in table
        assert "conflict_unresolved:duplicate_email(400 vs 409)" in table
        assert "spec_support=missing" not in table

    def test_adjudicate_policies(self) -> None:
        reqs = [RequirementRef("REQ-001", "重复邮箱返回 409。涉及 POST /users")]
        findings = find_status_conflicts(reqs, _USERS_SPEC)
        strict = adjudicate_conflicts(findings, ConflictPolicy.STRICT)
        spec_first = adjudicate_conflicts(findings, ConflictPolicy.SPEC_FIRST)
        req_first = adjudicate_conflicts(findings, ConflictPolicy.REQUIREMENT_FIRST)
        assert list(strict.values()) == ["conflict_unresolved:duplicate_email(400 vs 409)"]
        assert "spec_wins:400" in spec_first.values()
        assert "requirement_wins:409" in req_first.values()

    def test_finding_immutability(self) -> None:
        """Findings are frozen dataclasses - accidental mutation must fail."""
        f = Finding(kind=ConsistencyKind.SPEC_GAP, subject="x", detail="d")
        try:
            replace(f, subject="y")  # dataclasses.replace is the legal path
        except Exception as exc:  # pragma: no cover - defensive
            raise AssertionError("replace should work on frozen dataclass") from exc
        with pytest.raises(FrozenInstanceError):
            f.subject = "y"  # type: ignore[misc]
