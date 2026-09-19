"""FH2.2: dict <-> TestCase bidirectional adapter + H1 merge contract.

The NEW pipeline produces plain dicts (B6a-1); the legacy domain type is
``TestCase``. This adapter owns BOTH directions in one place so the parity
gates compare a single implementation, and fixates the H1 historical-merge
contract by READING the legacy ``_merge_historical_cases`` behavior
(plan-d: contract values come from code, never predicted).

H1 contract (fixated from ``TestCaseGenerator._merge_historical_cases`` /
``_case_dedup_key``, commit-pinned by tests):
1. **dedup precedence**: historical wins — a net-new case whose key already
   exists in the baseline is dropped, the historical copy is kept;
2. **renumber timing**: ids are assigned AFTER merge (TC-001.. sequential
   over the merged list); historical ids are NOT preserved;
3. **historical ordering**: baseline order is preserved verbatim (first
   occurrence wins inside the baseline too);
4. **empty historical**: merge is identity on new_cases;
5. **duplicate historical**: duplicate keys WITHIN the baseline are dropped
   (first occurrence kept);
6. **duplicate generated**: duplicate keys within new_cases are dropped the
   same way (first occurrence kept).

Dedup key: ``title.strip().lower() + "|" + endpoint.full_path.lower() + "|"
+ test_type.value`` (byte-equal to the legacy key).
"""

from __future__ import annotations

from typing import Any

from testagent.config.models import APIEndpoint, TestCase, TestPriority, TestType

__all__ = [
    "H1_CONTRACT",
    "case_dedup_key",
    "dict_to_testcase",
    "merge_historical_cases",
    "testcase_to_dict",
]

H1_CONTRACT = {
    "dedup_precedence": "historical wins; duplicate net-new dropped",
    "renumber_timing": "after merge, sequential TC-001.. over merged list",
    "historical_ordering": "baseline order preserved verbatim",
    "empty_historical": "merge is identity on new_cases",
    "duplicate_historical": "dropped within baseline, first kept",
    "duplicate_generated": "dropped within new_cases, first kept",
    "dedup_key": "title.strip().lower()|endpoint.full_path.lower()|test_type.value",
}


def _enum_fallback(field: str) -> Any:
    if field == "test_type":
        return TestType.FUNCTIONAL
    return TestPriority.MEDIUM


def case_dedup_key(tc: TestCase) -> str:
    """Byte-equal to ``TestCaseGenerator._case_dedup_key``."""
    return f"{tc.title.strip().lower()}|{tc.endpoint.full_path.lower()}|{tc.test_type.value}"


def dict_to_testcase(item: dict[str, Any], *, index: int = 0) -> TestCase | None:
    """Engine-dict -> TestCase. Tolerant like the legacy converter: missing
    optional fields default, invalid enum values fall back (documented H1
    compatibility — parity compares behavior, not strictness)."""
    if not isinstance(item, dict):
        return None
    endpoint_str = str(item.get("endpoint", "N/A N/A"))
    parts = endpoint_str.split(None, 1)
    method = parts[0] if parts else "N/A"
    path = parts[1] if len(parts) > 1 else "N/A"
    try:
        test_type = TestType(str(item.get("test_type", "functional")).lower())
    except ValueError:
        test_type = _enum_fallback("test_type")
    try:
        priority = TestPriority(str(item.get("priority", "medium")).lower())
    except ValueError:
        priority = _enum_fallback("priority")
    return TestCase(
        id=str(item.get("id", f"TC-{index:03d}")),
        title=str(item.get("title", "")),
        description=str(item.get("description", "")),
        endpoint=APIEndpoint(method=method, path=path),
        test_type=test_type,
        priority=priority,
        preconditions=[str(p) for p in item.get("preconditions", []) or []],
        steps=[str(s) for s in item.get("steps", []) or []],
        expected_results=[str(e) for e in item.get("expected_results", []) or []],
        tags=[str(t) for t in item.get("tags", []) or []],
    )


def testcase_to_dict(tc: TestCase) -> dict[str, Any]:
    """TestCase -> engine dict (id/narrative fields + endpoint full_path)."""
    return {
        "id": tc.id,
        "title": tc.title,
        "description": tc.description,
        "endpoint": tc.endpoint.full_path,
        "test_type": tc.test_type.value,
        "priority": tc.priority.value,
        "preconditions": list(tc.preconditions),
        "steps": list(tc.steps),
        "expected_results": list(tc.expected_results),
        "tags": list(tc.tags),
    }


def merge_historical_cases(
    historical: list[TestCase], new_cases: list[TestCase]
) -> tuple[list[TestCase], int]:
    """H1 contract implementation (byte-equal to the legacy merge).

    Returns ``(merged, new_count)`` — the net-new count is part of the
    legacy log line and the parity ledger.
    """
    merged: list[TestCase] = []
    seen: set[str] = set()
    for tc in historical:
        key = case_dedup_key(tc)
        if key not in seen:
            seen.add(key)
            merged.append(tc)
    new_count = 0
    for tc in new_cases:
        key = case_dedup_key(tc)
        if key not in seen:
            seen.add(key)
            merged.append(tc)
            new_count += 1
    return merged, new_count


def renumber(cases: list[TestCase]) -> None:
    """H1 dimension 2: sequential TC-001.. AFTER merge (in place)."""
    for idx, tc in enumerate(cases, 1):
        tc.id = f"TC-{idx:03d}"
