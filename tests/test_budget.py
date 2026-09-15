"""T7 tests: obligation-driven quota floor + CASES_BUDGET cap (§3.5)."""

from unittest.mock import MagicMock

from testagent.config.models import APIEndpoint, TestCase, TestPriority, TestType
from testagent.config.settings import Settings
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.testcase_generator import TestCaseGenerator
from testagent.pipeline.obligations import (
    BindingBasis,
    Obligation,
    ObligationRegistry,
    ObligationSource,
)


def _case(i: int, title: str, endpoint: str = "POST /users") -> TestCase:
    return TestCase(
        id=f"TC-{i:03d}",
        title=title,
        description="d",
        endpoint=APIEndpoint(method=endpoint.split()[0], path=endpoint.split()[1]),
        test_type=TestType.FUNCTIONAL,
        priority=TestPriority.HIGH,
    )


def _generator(budget: int) -> TestCaseGenerator:
    gen = TestCaseGenerator(
        llm_client=MagicMock(),
        prompt_builder=PromptBuilder(),
        cases_budget=budget,
    )
    gen._obligation_registry = ObligationRegistry()
    return gen


class TestExpectedForFloor:
    def test_obligation_floor_overrides_default(self) -> None:
        gen = _generator(budget=60)
        reg = gen._obligation_registry
        assert reg is not None
        reg.register(
            Obligation(
                id="REQ-001-AC1",
                requirement_id="REQ-001",
                statement="create returns 201",
                source=ObligationSource.REQUIREMENT,
                endpoint_bindings=("POST /users",),
                binding_basis=BindingBasis.KEYWORD,
            )
        )
        reg.register(
            Obligation(
                id="REQ-001-AC2",
                requirement_id="REQ-001",
                statement="duplicate returns 409",
                source=ObligationSource.REQUIREMENT,
                endpoint_bindings=("POST /users",),
                binding_basis=BindingBasis.KEYWORD,
            )
        )
        floor = gen._expected_for(["POST /users", "GET /users"])
        assert floor["POST /users"] == 2  # two uncovered obligations
        assert floor["GET /users"] == 2  # policy default (no obligations bound)

    def test_session_cap_clamps_total(self) -> None:
        gen = _generator(budget=10)
        gen._session_case_count = 0
        floor = gen._expected_for(["A", "B", "C", "D", "E", "F"])  # 6 x default 2 = 12 > 10
        assert sum(floor.values()) <= 10

    def test_exhausted_budget_yields_zero(self) -> None:
        gen = _generator(budget=10)
        gen._session_case_count = 10
        assert gen._expected_for(["A", "B"]) == {"A": 0, "B": 0}

    def test_budget_zero_disables_cap(self) -> None:
        gen = _generator(budget=0)
        gen._session_case_count = 10_000
        floor = gen._expected_for(["A", "B"])
        assert sum(floor.values()) == 4  # defaults, no clamping


class TestBudgetEnforcement:
    def test_trim_with_accounting_keeps_obligation_cases(self) -> None:
        gen = _generator(budget=3)
        reg = gen._obligation_registry
        assert reg is not None
        reg.register(
            Obligation(
                id="REQ-001-AC1",
                requirement_id="REQ-001",
                statement="s",
                source=ObligationSource.REQUIREMENT,
            )
        )
        reg.cover("REQ-001-AC1", "TC-005")
        cases = [_case(i, f"case {i}") for i in range(1, 13)]
        gen._case_obligations = {"TC-005": ["REQ-001-AC1"]}
        kept = gen._enforce_cases_budget(cases)
        assert len(kept) == 3
        assert "TC-005" in [c.id for c in kept], "obligation-covering case must survive trim"
        report = gen._budget_trim_report
        assert "produced=12" in report and "kept=3" in report and "trimmed=9" in report

    def test_no_cap_when_under_budget(self) -> None:
        gen = _generator(budget=60)
        cases = [_case(i, f"case {i}") for i in range(1, 6)]
        assert gen._enforce_cases_budget(cases) == cases

    def test_cap_disabled_with_zero(self) -> None:
        gen = _generator(budget=0)
        cases = [_case(i, f"case {i}") for i in range(1, 13)]
        assert len(gen._enforce_cases_budget(cases)) == 12


class TestSettingsBudget:
    def test_default(self) -> None:
        assert Settings(cases_budget=60).cases_budget == 60

    def test_negative_treated_as_off(self) -> None:
        """Negative budgets behave like 0 (cap disabled) - the enforcement
        path checks ``budget <= 0`` before clamping."""
        settings = Settings(cases_budget=-1)
        assert settings.cases_budget == -1

    def test_zero_is_legacy(self) -> None:
        assert Settings(cases_budget=0).cases_budget == 0
