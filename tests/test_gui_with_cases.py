"""M6 tests: strict GUI reference import, selection ledger, off-path parity (V15/V16)."""

import json
from pathlib import Path

import pytest

from testagent.config.models import (
    GUITestGenInput,
    RequirementItem,
)
from testagent.orchestration.gui_with_cases import (
    GuiReferenceError,
    build_gui_input_with_cases,
    load_case_references,
    select_references,
)


def _write_cases(tmp_path: Path, cases: list[dict], name: str = "cases.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(cases), encoding="utf-8")
    return p


def _case(i: int, priority: str = "medium", steps: int = 2) -> dict:
    return {
        "id": f"TC-{i:03d}",
        "title": f"case {i}",
        "priority": priority,
        "description": f"desc {i}",
        "steps": [f"step {j}" for j in range(steps)],
        "expected_results": ["ok"],
        "preconditions": ["logged in"],
        "endpoint": "GET /x",
    }


class TestStrictImport:
    def test_happy_path(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [_case(1), _case(2)])
        refs = load_case_references(src)
        assert [r.id for r in refs] == ["TC-001", "TC-002"]
        assert refs[0].steps == ("step 0", "step 1")

    def test_missing_file_fails(self, tmp_path: Path) -> None:
        with pytest.raises(GuiReferenceError, match="not found"):
            load_case_references(tmp_path / "nope.json")

    def test_empty_array_fails(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [])
        with pytest.raises(GuiReferenceError, match="no cases"):
            load_case_references(src)

    def test_bad_entry_fails(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [{"id": "", "title": "x"}])
        with pytest.raises(GuiReferenceError, match="id"):
            load_case_references(src)

    def test_duplicate_ids_fail(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [_case(1), dict(_case(1))])
        with pytest.raises(GuiReferenceError, match="duplicate"):
            load_case_references(src)

    def test_wrong_priority_fails(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [{**_case(1), "priority": "urgent"}])
        with pytest.raises(GuiReferenceError, match="priority"):
            load_case_references(src)


class TestSelection:
    def test_priority_rank_stable(self, tmp_path: Path) -> None:
        refs = [
            _case_ref(_case(1, "low")),
            _case_ref(_case(2, "high")),
            _case_ref(_case(3, "medium")),
            _case_ref(_case(4, "high")),
        ]
        selection = select_references(
            refs, max_cases=3, max_chars=16000, source_name="c.json", source_sha="abc"
        )
        assert selection.selected_ids == ["TC-002", "TC-004", "TC-003"]
        assert selection.omitted == [("TC-001", "count_limit")]

    def test_budget_skip_whole_case(self, tmp_path: Path) -> None:
        big = _case(1, steps=400)
        small = _case(2, steps=1)
        refs = [_case_ref(big), _case_ref(small)]
        selection = select_references(
            refs, max_cases=10, max_chars=900, source_name="c.json", source_sha="abc"
        )
        assert selection.selected_ids == ["TC-002"]
        assert ("TC-001", "oversized_case") in selection.omitted

    def test_none_fit_fails(self, tmp_path: Path) -> None:
        refs = [_case_ref(_case(1, steps=1000))]
        with pytest.raises(GuiReferenceError, match="no reference case"):
            select_references(
                refs, max_cases=5, max_chars=200, source_name="c.json", source_sha="abc"
            )


def _case_ref(case: dict):
    import json as _json
    import tempfile
    from pathlib import Path

    from testagent.orchestration.gui_with_cases import load_case_references

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        _json.dump([case], fh)
        path = fh.name
    return load_case_references(Path(path))[0]


class TestBuildGuiInput:
    def test_gen_cases_appended_and_renamed_on_conflict(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [_case(1), _case(2)])
        existing = [RequirementItem(id="GEN-CASES", title="existing", description="d")]
        gui_input, selection = build_gui_input_with_cases(
            requirements=existing,
            reference_path=src,
            url="https://x.test",
            endpoints=[],
            output_language="english",
            generator=None,
        )
        assert gui_input.requirements[0].id == "GEN-CASES"
        assert gui_input.requirements[-1].id.startswith("GEN-CASES-")
        assert selection.original_count == 2
        # reference content traceable
        assert "TC-001" in gui_input.requirements[-1].description
        assert selection.rendered_chars > 0

    def test_off_path_untouched_input_shape(self) -> None:
        """Off path (no --with-testcases) constructs the plain input; the
        on-path builder never removes original requirements (v4 §8.3)."""
        reqs = [RequirementItem(id="REQ-001", title="t", description="d")]
        gui_input = GUITestGenInput(
            requirements=reqs, url="https://x", endpoints=[], output_language="english"
        )
        assert len(gui_input.requirements) == 1


class _SpyGenerator:
    def __init__(self) -> None:
        self.received: list[GUITestGenInput] = []

    def generate(self, gen_input: GUITestGenInput) -> str:
        self.received.append(gen_input)
        return "# playwright script"


class TestEndToEndOrchestration:
    def test_reference_reaches_generator(self, tmp_path: Path) -> None:
        src = _write_cases(tmp_path, [_case(1)])
        spy = _SpyGenerator()
        gui_input, _ = build_gui_input_with_cases(
            requirements=[RequirementItem(id="R1", title="t", description="d")],
            reference_path=src,
            url="https://x.test",
            endpoints=[],
            output_language="english",
            generator=spy,
        )
        assert spy.received == []
        # The CLI layer invokes the generator with this input:
        out = spy.generate(gui_input)
        assert out == "# playwright script"
        assert spy.received[0].requirements[-1].id.startswith("GEN-CASES")
