"""T6 tests: deterministic requirement->endpoint binding chain (RC-2)."""

from testagent.config.models import APIEndpoint
from testagent.pipeline.binding import Binding, bind_requirements, extract_explicit_bindings
from testagent.pipeline.consistency import RequirementRef
from testagent.pipeline.obligations import BindingBasis

_SPEC = [
    APIEndpoint(method="GET", path="/users", summary="list"),
    APIEndpoint(
        method="POST",
        path="/users",
        summary="create",
        request_body={
            "media_type": "application/json",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "email": {"type": "string"}},
            },
        },
    ),
    APIEndpoint(method="DELETE", path="/users/{id}", summary="delete"),
    APIEndpoint(method="POST", path="/auth/login", summary="login"),
]


def _ref(rid: str, text: str) -> RequirementRef:
    return RequirementRef(id=rid, text=text)


class TestExplicitOverrides:
    def test_at_annotation_wins_over_heuristics(self) -> None:
        """The gate case: an explicit '@ POST /auth/login' annotation must
        beat any field/verb heuristic (e.g. 'password' field matching)."""
        req = _ref(
            "REQ-002",
            "用户登录 `password` 校验。@ POST /auth/login",
        )
        bindings = bind_requirements([req], _SPEC)
        assert bindings[0].basis is BindingBasis.EXPLICIT
        assert bindings[0].endpoints == ("POST /auth/login",)

    def test_bare_path_annotation_binds_all_methods(self) -> None:
        req = _ref("REQ-001", "用户管理，绑定 @ /users")
        bindings = bind_requirements([req], _SPEC)
        assert bindings[0].basis is BindingBasis.EXPLICIT
        assert set(bindings[0].endpoints) == {"GET /users", "POST /users"}

    def test_plain_mention_counts_as_explicit(self) -> None:
        req = _ref("REQ-001", "删除用户调用 DELETE /users/{id}")
        bindings = bind_requirements([req], _SPEC)
        assert bindings[0].basis is BindingBasis.EXPLICIT
        assert bindings[0].endpoints == ("DELETE /users/{id}",)

    def test_nonexistent_annotation_ignored(self) -> None:
        """An annotation pointing outside the spec must NOT bind (verified
        against the spec) — the chain falls through to lower sources."""
        req = _ref("REQ-002", "登录 @ POST /auth/login（规格缺失）")
        bindings = bind_requirements([req], [])
        assert bindings[0].basis is BindingBasis.NONE
        assert bindings[0].gap is not None


class TestKeywordMatching:
    def test_field_match_binds_endpoint(self) -> None:
        req = _ref("REQ-001", "registration validates the email format and the name length")
        bindings = bind_requirements([req], _SPEC)
        assert bindings[0].basis is BindingBasis.KEYWORD
        assert "POST /users" in bindings[0].endpoints

    def test_verb_noun_match(self) -> None:
        req = _ref("REQ-001", "an admin can delete the user record")
        bindings = bind_requirements([req], _SPEC)
        assert bindings[0].basis is BindingBasis.KEYWORD
        assert "DELETE /users/{id}" in bindings[0].endpoints


class TestSuggestionsAndGaps:
    def test_llm_suggestion_is_candidate_only(self) -> None:
        req = _ref("REQ-001", "平台运营策略相关需求")
        bindings = bind_requirements([req], _SPEC, llm_suggestions={"REQ-001": ["GET /users"]})
        assert bindings[0].basis is BindingBasis.LLM_SUGGESTION
        assert bindings[0].endpoints == ("GET /users",)

    def test_invalid_suggestion_dropped(self) -> None:
        req = _ref("REQ-001", "平台运营策略相关需求")
        bindings = bind_requirements([req], _SPEC, llm_suggestions={"REQ-001": ["POST /nope"]})
        assert bindings[0].basis is BindingBasis.NONE
        assert bindings[0].endpoints == ()

    def test_empty_binding_produces_gap(self) -> None:
        req = _ref("REQ-002", "账号连续失败 5 次锁定 10 分钟")
        bindings = bind_requirements([req], _SPEC)
        assert bindings[0].endpoints == ()
        assert bindings[0].gap is not None
        assert bindings[0].gap.kind.value == "spec_gap"


class TestExtract:
    def test_extract_explicit_bindings(self) -> None:
        text = "绑定 @ POST /auth/login 与 @ /users"
        assert extract_explicit_bindings(text) == ["POST /auth/login", "/users"]

    def test_binding_is_frozen(self) -> None:
        import dataclasses

        import pytest

        b = Binding(requirement_id="R", endpoints=(), basis=BindingBasis.NONE)
        with pytest.raises(dataclasses.FrozenInstanceError):
            b.requirement_id = "X"  # type: ignore[misc]
