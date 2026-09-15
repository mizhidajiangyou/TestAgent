"""T11 tests: deterministic normalization + semantic validation (§4-T11)."""

from testagent.pipeline.normalization import (
    needs_reask,
    normalize_case,
    semantic_validation,
)


class TestNormalize:
    def test_placeholder_aliases_canonical(self) -> None:
        case = {
            "title": "use {{uuid}} and <uuid>",
            "preconditions": ["<token> present"],
            "steps": ["GET /x\n  with <api_key>"],
            "expected_results": ["200 ok; body valid"],
        }
        out = normalize_case(case)
        assert out["title"] == "use <RUN_ID> and <RUN_ID>"
        assert out["preconditions"] == ["<TOKEN> present"]
        assert out["steps"] == ["GET /x with <API_KEY>"]
        # semicolon split into individual assertions
        assert out["expected_results"] == ["200 ok", "body valid"]

    def test_original_not_mutated(self) -> None:
        case = {"title": "t", "steps": ["a\nb"], "expected_results": ["x;y"]}
        normalize_case(case)
        assert case["steps"] == ["a\nb"] and case["expected_results"] == ["x;y"]


class TestSemanticValidation:
    def test_missing_assertions_flagged(self) -> None:
        gaps = semantic_validation({"expected_results": ["looks fine"]})
        assert any("assertion-incomplete" in g for g in gaps)

    def test_status_code_assertion_clean(self) -> None:
        gaps = semantic_validation(
            {
                "expected_results": ["returns 201"],
                "scenario_operation": "POST /users",
                "scenario_scene": "create",
                "scenario_variant": "happy-path",
            }
        )
        assert gaps == []

    def test_identity_missing(self) -> None:
        gaps = semantic_validation(
            {
                "expected_results": ["returns 201"],
                "scenario_operation": "POST /users",
                "scenario_scene": "",
                "scenario_variant": "x",
            }
        )
        assert any("identity-missing" in g for g in gaps)

    def test_unknown_obligation_flagged(self) -> None:
        gaps = semantic_validation(
            {
                "expected_results": ["returns 201"],
                "covers_obligations": ["REQ-009-AC1"],
            },
            known_obligations={"REQ-001-AC1"},
        )
        assert any("obligation-unknown" in g for g in gaps)


class TestReaskGate:
    def test_structural_only_never_reasks(self) -> None:
        assert needs_reask([]) is False
        assert needs_reask(["structural:placeholder-style"]) is False

    def test_semantic_gap_eligible(self) -> None:
        assert needs_reask(["assertion-missing: no expected_results"]) is True
