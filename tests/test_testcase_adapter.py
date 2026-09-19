"""FH2.2 tests: dict<->TestCase adapter + H1 six-dimension contract.

Every H1 dimension is pinned against the LEGACY implementation by running
the SAME scenario through both ``TestCaseGenerator._merge_historical_cases``
(and the legacy dedup key) and the adapter — byte-equal outcomes prove the
contract fixation ("contract values come from code, never predicted").
"""

from testagent.config.models import APIEndpoint, TestCase, TestPriority, TestType
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.pipeline.testcase_adapter import (
    H1_CONTRACT,
    case_dedup_key,
    dict_to_testcase,
    merge_historical_cases,
    renumber,
)
from testagent.pipeline.testcase_adapter import testcase_to_dict as tc_to_dict_impl


def _case(
    i: int, title: str, method: str = "GET", path: str = "/users", ttype: str = "functional"
) -> TestCase:
    return TestCase(
        id=f"TC-{i:03d}",
        title=title,
        description=f"desc {i}",
        endpoint=APIEndpoint(method=method, path=path),
        test_type=TestType(ttype),
        priority=TestPriority.MEDIUM,
        steps=[f"step {i}"],
        expected_results=["ok"],
    )


def _dict_of(title: str, method: str = "GET", path: str = "/users") -> dict:
    return {
        "id": "TC-XXX",
        "title": title,
        "description": "d",
        "endpoint": f"{method} {path}",
        "test_type": "functional",
        "priority": "high",
        "preconditions": [],
        "steps": ["s"],
        "expected_results": ["ok"],
        "tags": [],
    }


class TestRoundTrip:
    def test_dict_to_testcase_round_trip(self) -> None:
        tc = _case(1, "List users")
        d = tc_to_dict_impl(tc)
        back = dict_to_testcase(d, index=1)
        assert back is not None
        assert case_dedup_key(back) == case_dedup_key(tc)
        assert back.title == tc.title
        assert back.endpoint.full_path == tc.endpoint.full_path
        assert back.test_type is tc.test_type

    def test_tolerant_enum_fallback(self) -> None:
        """Invalid enum values fall back exactly like the legacy converter
        (H1 compatibility: parity compares behavior, not strictness)."""
        d = _dict_of("x")
        d["test_type"] = "nonsense"
        d["priority"] = "nonsense"
        tc = dict_to_testcase(d)
        assert tc is not None
        assert tc.test_type is TestType.FUNCTIONAL
        assert tc.priority is TestPriority.MEDIUM

    def test_non_dict_rejected(self) -> None:
        assert dict_to_testcase("garbage") is None  # type: ignore[arg-type]


def _run_both(historical: list[TestCase], new: list[TestCase]):
    """Run the same scenario through legacy and adapter; assert equality."""
    legacy = TestCaseGenerator._merge_historical_cases(historical, new)
    adapter, new_count = merge_historical_cases(historical, new)
    assert [c.title for c in adapter] == [c.title for c in legacy]
    assert [c.id for c in adapter] == [c.id for c in legacy]
    return adapter, new_count


class TestH1Contract:
    def test_contract_documented(self) -> None:
        assert set(H1_CONTRACT) == {
            "dedup_precedence",
            "renumber_timing",
            "historical_ordering",
            "empty_historical",
            "duplicate_historical",
            "duplicate_generated",
            "dedup_key",
        }

    def test_dedup_precedence_historical_wins(self) -> None:
        """H1-1: same dedup key in baseline and net-new -> historical kept."""
        historical = [_case(1, "list users")]
        new = [_case(9, "LIST USERS")]  # same key modulo case/strip
        merged, new_count = _run_both(historical, new)
        assert new_count == 0
        assert len(merged) == 1
        assert merged[0].description == "desc 1"  # the HISTORICAL copy

    def test_renumber_timing_after_merge(self) -> None:
        """H1-2: ids assigned after merge — historical ids not preserved."""
        historical = [_case(7, "first")]
        new = [_case(8, "second")]
        merged, _ = _run_both(historical, new)
        renumber(merged)
        assert [c.id for c in merged] == ["TC-001", "TC-002"]

    def test_historical_ordering_preserved(self) -> None:
        """H1-3: baseline order verbatim."""
        historical = [_case(3, "gamma"), _case(1, "alpha"), _case(2, "beta")]
        new = [_case(9, "delta")]
        merged, _ = _run_both(historical, new)
        assert [c.title for c in merged] == ["gamma", "alpha", "beta", "delta"]

    def test_empty_historical_identity(self) -> None:
        """H1-4: no baseline -> merge is identity on new_cases."""
        new = [_case(1, "only"), _case(2, "second")]
        merged, new_count = _run_both([], new)
        assert new_count == 2
        assert [c.title for c in merged] == ["only", "second"]

    def test_duplicate_historical_dropped(self) -> None:
        """H1-5: duplicate keys WITHIN the baseline -> first kept."""
        historical = [_case(1, "dup"), _case(2, "dup")]
        merged, _ = _run_both(historical, [])
        assert len(merged) == 1
        assert merged[0].description == "desc 1"

    def test_duplicate_generated_dropped(self) -> None:
        """H1-6: duplicate keys within net-new -> first kept."""
        new = [_case(1, "dup gen"), _case(2, "dup gen")]
        merged, new_count = _run_both([], new)
        assert new_count == 1
        assert len(merged) == 1

    def test_dedup_key_byte_equal_to_legacy(self) -> None:
        tc = _case(1, "  Mixed CASE ")
        assert case_dedup_key(tc) == TestCaseGenerator._case_dedup_key(tc)
