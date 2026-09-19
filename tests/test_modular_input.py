"""M1 tests: split-mode resolution, single-doc adapter, settings override
and resume capability_options (v4 §3.3/§4.1, V01/V03 partial)."""

from pathlib import Path

import pytest

from testagent.container import Container
from testagent.orchestration.input_adapter import (
    resolve_requirements,
    resolve_split_mode,
)
from testagent.orchestration.settings_override import (
    ResolvedCapabilities,
    apply_capabilities_override,
    parse_capability_options,
    resolve_capabilities,
)

_DOC = "# Report\n\n## Section A\n\nAlpha requirement.\n\n## Section B\n\nBeta requirement.\n"


class TestSplitMode:
    def test_explicit_overrides_settings(self) -> None:
        assert resolve_split_mode("single", "auto") == "single"
        assert resolve_split_mode(None, "single") == "single"
        assert resolve_split_mode(None, "auto") == "auto"

    def test_invalid_rejected(self) -> None:
        with pytest.raises(ValueError, match="split mode"):
            resolve_split_mode("both", "auto")
        with pytest.raises(ValueError, match="split mode"):
            resolve_split_mode(None, "chapter")


class TestSingleDocAdapter:
    def test_single_yields_one_full_text_unit(self, tmp_path: Path) -> None:
        doc = tmp_path / "spec.md"
        doc.write_text(_DOC, encoding="utf-8")
        items = resolve_requirements(str(doc), "single")
        assert len(items) == 1
        item = items[0]
        assert item.id == "DOC-1"
        assert item.title == "spec"
        assert item.module == ""
        assert item.acceptance_criteria == []
        # Full text preserved verbatim (no trim, no chapter split).
        assert "Alpha requirement." in item.description
        assert "Beta requirement." in item.description

    def test_auto_keeps_chapters(self, tmp_path: Path) -> None:
        doc = tmp_path / "spec.md"
        doc.write_text(_DOC, encoding="utf-8")
        items = resolve_requirements(str(doc), "auto")
        assert len(items) >= 2

    def test_missing_file_fails_before_llm(self) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_requirements("/nonexistent/doc.md", "single")

    def test_whitespace_body_rejected(self, tmp_path: Path) -> None:
        doc = tmp_path / "empty.md"
        doc.write_text("   \n\n  ", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            resolve_requirements(str(doc), "single")


class TestResolveCapabilities:
    def _settings(self, split="auto", conc=5):
        class LLM:
            max_concurrency = conc

        class S:
            llm = LLM()
            split_mode = split

        return S()

    def test_priority_chain(self) -> None:
        r = resolve_capabilities(
            cli_split="single",
            cli_concurrency=3,
            saved={"schema_version": 1, "resolved_split_mode": "auto", "resolved_concurrency": 2},
            settings=self._settings(split="single", conc=4),
        )
        assert r.split_mode == "single" and r.split_mode_source == "cli"
        assert r.concurrency == 3 and r.concurrency_source == "cli"

    def test_resume_beats_settings(self) -> None:
        r = resolve_capabilities(
            cli_split=None,
            cli_concurrency=None,
            saved={"schema_version": 1, "resolved_split_mode": "single", "resolved_concurrency": 1},
            settings=self._settings(),
        )
        assert r.split_mode == "single" and r.split_mode_source == "resume"
        assert r.concurrency == 1 and r.concurrency_source == "resume"

    def test_corrupt_saved_rejected(self) -> None:
        with pytest.raises(ValueError, match="schema_version"):
            resolve_capabilities(
                cli_split=None,
                cli_concurrency=None,
                saved={"schema_version": 99},
                settings=self._settings(),
            )
        with pytest.raises(ValueError, match="split mode"):
            resolve_capabilities(
                cli_split=None,
                cli_concurrency=None,
                saved={"schema_version": 1, "resolved_split_mode": "chapter"},
                settings=self._settings(),
            )

    def test_bool_and_nonpositive_concurrency_rejected(self) -> None:
        for bad in (True, 0, -1):
            with pytest.raises(ValueError, match="concurrency"):
                resolve_capabilities(
                    cli_split=None,
                    cli_concurrency=bad,
                    saved=None,
                    settings=self._settings(),
                )


class TestSettingsOverride:
    def test_override_preserves_container_injection(self) -> None:
        container = Container()
        original_settings = container.settings()
        resolved = ResolvedCapabilities(
            split_mode="single",
            concurrency=1,
            split_mode_source="cli",
            concurrency_source="cli",
        )
        apply_capabilities_override(container, resolved)
        overridden = container.settings()
        assert overridden.split_mode == "single"
        assert overridden.llm.max_concurrency == 1
        # untouched fields survive (output_dir, endpoint config object identity of values)
        assert overridden.output_dir == original_settings.output_dir
        assert overridden.llm.base_url == original_settings.llm.base_url

    def test_parse_capability_options(self) -> None:
        assert parse_capability_options(None) is None
        assert parse_capability_options({}) is None
        assert parse_capability_options({"capability_options": None}) is None
        good = {"schema_version": 1, "resolved_split_mode": "auto", "resolved_concurrency": 3}
        assert parse_capability_options({"capability_options": good}) == good
        with pytest.raises(ValueError, match="capability_options"):
            parse_capability_options({"capability_options": "corrupt"})


class TestCapabilityLayerDirection:
    """v4 s1.2: capability layers never import Container/generators/engine internals."""

    def test_no_forbidden_imports(self) -> None:
        import ast
        import pathlib as pl

        forbidden = ("testagent.container", "testagent.generators")
        layer_dirs = ["testagent/review", "testagent/artifact", "testagent/orchestration"]
        violations: list[str] = []
        repo = pl.Path(__file__).parents[1]
        for d in layer_dirs:
            for py in sorted((repo / d).glob("*.py")):
                tree = ast.parse(py.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.ImportFrom)
                        and node.module
                        and node.module.startswith(forbidden)
                    ):
                        violations.append(f"{py.name}: {node.module}")
        assert not violations, f"capability layer imports forbidden modules: {violations}"
