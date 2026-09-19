"""M2 tests: generate-tests CLI capability wiring (V01-V04 core)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

from click.testing import CliRunner

from testagent.cli.commands.generate_tests import generate_tests
from testagent.container import Container

_DOC = "# A\n\nAlpha.\n\n## B\n\nBeta.\n"


class _FakeGenerator:
    def __init__(self) -> None:
        self.received: list = []
        self._review_enabled = False

    def set_review_enabled(self, enabled: bool) -> None:
        self._review_enabled = enabled

    def save(self, output, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[]", encoding="utf-8")
        return output_path

    async def agenerate(self, gen_input, session_id=None):
        from testagent.config.models import APIEndpoint, TestCase, TestPriority, TestType

        self.received.append(gen_input)
        return [
            TestCase(
                id="TC-001",
                title="sample",
                description="d",
                endpoint=APIEndpoint(method="GET", path="/x"),
                test_type=TestType.FUNCTIONAL,
                priority=TestPriority.MEDIUM,
            )
        ]


class TestGenerateTestsCapabilities:
    def _run(self, tmp_path: Path, args: list[str], fake: _FakeGenerator, monkeypatch) -> int:
        container = Container()
        from dependency_injector.providers import Object

        container.testcase_generator.override(Object(fake))
        doc = tmp_path / "req.md"
        doc.write_text(_DOC, encoding="utf-8")
        out = tmp_path / "out.json"
        runner = CliRunner()
        base = ["-r", str(doc), "-o", str(out)]
        result = runner.invoke(generate_tests, base + args, obj={"container": container})
        return result

    def test_default_auto_two_units(self, tmp_path, monkeypatch) -> None:
        fake = _FakeGenerator()
        result = self._run(tmp_path, [], fake, monkeypatch)
        assert result.exit_code == 0, result.output
        assert len(fake.received[0].requirements) >= 2

    def test_no_split_single_unit(self, tmp_path, monkeypatch) -> None:
        fake = _FakeGenerator()
        result = self._run(tmp_path, ["--split"], fake, monkeypatch)
        assert result.exit_code == 0, result.output
        items = fake.received[0].requirements
        assert len(items) == 1
        assert items[0].id == "DOC-1"
        assert "Alpha." in items[0].description and "Beta." in items[0].description

    def test_single_with_swagger_warns_but_proceeds(self, tmp_path, monkeypatch) -> None:
        fake = _FakeGenerator()
        spec = tmp_path / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "paths": {
                        "/x": {
                            "get": {
                                "responses": {"200": {"description": "OK"}},
                                "tags": ["x"],
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        result = self._run(tmp_path, ["--split", "-s", str(spec)], fake, monkeypatch)
        assert result.exit_code == 0, result.output
        assert "WARN" in result.output or "single-doc" in result.output

    def test_concurrency_zero_rejected(self, tmp_path, monkeypatch) -> None:
        fake = _FakeGenerator()
        result = self._run(tmp_path, ["--concurrency", "0"], fake, monkeypatch)
        assert result.exit_code != 0

    def test_concurrency_applied_via_override(self, tmp_path, monkeypatch) -> None:
        fake = _FakeGenerator()
        result = self._run(tmp_path, ["--concurrency", "1"], fake, monkeypatch)
        assert result.exit_code == 0, result.output

    def test_resume_saves_capability_options(self, tmp_path, monkeypatch) -> None:
        fake = _FakeGenerator()
        result = self._run(tmp_path, ["--split", "--concurrency", "2"], fake, monkeypatch)
        assert result.exit_code == 0, result.output
        # find the newest session record
        records = sorted(Path("output/sessions").glob("*.json"), key=lambda p: p.stat().st_mtime)
        assert records, "session record must be written"
        record = json.loads(records[-1].read_text(encoding="utf-8"))
        caps = record.get("capability_options")
        assert caps is not None
        assert caps["schema_version"] == 1
        assert caps["resolved_split_mode"] == "single"
        assert caps["resolved_concurrency"] == 2


class TestConcurrencyParity:
    """V04: deterministic per-input-unit fake; concurrency 1/3/default must
    produce identical case sets, order and numbering (fake-LLM level)."""

    def test_parity_across_concurrency(self, tmp_path: Path) -> None:
        import asyncio

        from testagent.config.models import (
            APIEndpoint,
            RequirementItem,
            TestCase,
            TestPriority,
            TestType,
        )
        from testagent.engine.concurrency import gather_with_concurrency

        def make_item(i: int) -> RequirementItem:
            return RequirementItem(id=f"REQ-{i:03d}", title=f"r{i}", description=f"d{i}")

        def case_for(item: RequirementItem) -> TestCase:
            n = int(item.id.split("-")[1])
            return TestCase(
                id=f"TC-{n:03d}",
                title=f"case {n}",
                description=f"desc {n}",
                endpoint=APIEndpoint(method="GET", path=f"/r{n}"),
                test_type=TestType.FUNCTIONAL,
                priority=TestPriority.MEDIUM,
            )

        async def run(concurrency: int | None) -> list[TestCase]:
            calls: list[str] = []

            class PerUnitLLM:
                async def achat(
                    self, system_prompt, user_prompt, response_format=None, max_tokens=None
                ):
                    calls.append(user_prompt)
                    return "[]"

            async def gen_unit(item: RequirementItem) -> list[TestCase]:
                return [case_for(item)]

            items = [make_item(i) for i in range(1, 8)]
            results = await gather_with_concurrency(
                concurrency or 5, *[gen_unit(it) for it in items]
            )
            return [c for batch in results for c in batch]

        baseline = asyncio.run(run(1))
        for conc in (3, 5, 7):
            other = asyncio.run(run(conc))
            assert [c.id for c in other] == [c.id for c in baseline], (
                f"concurrency={conc} changed order/numbering"
            )

    def test_budget_boundary_stable_across_concurrency(self) -> None:
        """CASES_BUDGET trim must be deterministic regardless of completion
        order (trim keeps obligation-covering first, then source order)."""
        from testagent.config.models import APIEndpoint, TestCase, TestPriority, TestType
        from testagent.generators.testcase_generator import TestCaseGenerator
        from testagent.pipeline.obligations import ObligationRegistry

        gen = TestCaseGenerator(llm_client=MagicMock(), prompt_builder=MagicMock(), cases_budget=3)
        gen._obligation_registry = ObligationRegistry()
        cases = [
            TestCase(
                id=f"TC-{i:03d}",
                title=f"case {i}",
                description="d",
                endpoint=APIEndpoint(method="GET", path="/x"),
                test_type=TestType.FUNCTIONAL,
                priority=TestPriority.MEDIUM,
            )
            for i in range(1, 6)
        ]
        kept = gen._enforce_cases_budget(cases)
        assert [c.id for c in kept] == ["TC-001", "TC-002", "TC-003"]
        # Determinism: same input order -> same trim. Reversed input keeps
        # ITS first-3 (source-order policy), not a canonical set.
        again = gen._enforce_cases_budget(list(reversed(cases)))
        assert [c.id for c in again] == ["TC-005", "TC-004", "TC-003"]
