"""Pipeline-layer unit tests (plan-c Step 1-3): manifest strictness, parity
modes, UnitStatus taxonomy, synthetic contexts, registry conflicts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from testagent.pipeline.manifest import Manifest, load_manifest
from testagent.pipeline.parity import (
    ParityMode,
    check_renumber,
    diff_artifacts,
    diff_text,
)
from testagent.pipeline.status import (
    RECOVERY_POLICY,
    UnitStatus,
    unit_result_from_engine,
    unit_status_from_exception,
    unit_status_from_outcome,
)
from testagent.pipeline.synthetic import build_synthetic_context

REPO = Path(__file__).parents[1]

# ----------------------------------------------------------------------
# Manifest strictness (B1.4/B4.2)
# ----------------------------------------------------------------------


def _minimal_manifest(**overrides: object) -> dict:
    base = {
        "name": "demo",
        "pipeline": {"stages": [{"name": "s1", "template": "t.j2", "system_prompt": "inline:x"}]},
    }
    base.update(overrides)
    return base


class TestManifestStrictness:
    def test_minimal_manifest_parses(self) -> None:
        m = Manifest.model_validate(_minimal_manifest())
        assert m.name == "demo"
        assert m.manifest_version == 1

    def test_typo_field_rejected(self) -> None:
        # The v2 killer: "requird" would silently mean required=False.
        with pytest.raises(ValidationError, match="requird"):
            Manifest.model_validate(
                _minimal_manifest(inputs=[{"name": "a", "kind": "text", "requird": True}])
            )

    def test_unknown_version_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unsupported manifest_version 99"):
            Manifest.model_validate(_minimal_manifest(manifest_version=99))

    def test_uppercase_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must match"):
            Manifest.model_validate(_minimal_manifest(name="Demo"))

    def test_reserved_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reserved CLI command"):
            Manifest.model_validate(_minimal_manifest(name="checkpoint"))

    def test_duplicate_aliases_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate aliases"):
            Manifest.model_validate(_minimal_manifest(aliases=["foo", "foo"]))

    def test_alias_equal_to_name_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicates the manifest name"):
            Manifest.model_validate(_minimal_manifest(aliases=["demo"]))

    def test_empty_stages_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least one stage"):
            Manifest.model_validate(_minimal_manifest(pipeline={"stages": []}))

    def test_split_requires_input(self) -> None:
        with pytest.raises(ValidationError, match=r"requires split\.input"):
            Manifest.model_validate(
                _minimal_manifest(
                    pipeline={
                        "stages": [
                            {
                                "name": "s1",
                                "template": "t.j2",
                                "system_prompt": "inline:x",
                                "split": {"by": "per_input"},
                            }
                        ]
                    }
                )
            )

    def test_choice_without_choices_rejected(self) -> None:
        with pytest.raises(ValidationError, match="requires choices"):
            Manifest.model_validate(_minimal_manifest(inputs=[{"name": "fmt", "kind": "choice"}]))

    def test_load_manifest_reads_file(self, tmp_path: Path) -> None:
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(_minimal_manifest()), encoding="utf-8")
        assert load_manifest(p).name == "demo"

    def test_template_refs_includes_review(self) -> None:
        m = Manifest.model_validate(
            _minimal_manifest(
                pipeline={
                    "stages": [{"name": "s1", "template": "a.j2", "system_prompt": "file:sys.j2"}]
                },
                review={"template": "rev.j2"},
            )
        )
        assert m.template_refs() == ["a.j2", "sys.j2", "rev.j2"]


# ----------------------------------------------------------------------
# Parity comparator (B1.3)
# ----------------------------------------------------------------------

_OLD = [
    {
        "id": "TC-001",
        "title": "Login",
        "description": "d",
        "endpoint": "/login",
        "tags": ["auth"],
        "foo": "IMPORTANT",
    },
    {
        "id": "TC-002",
        "title": "Logout",
        "description": "d",
        "endpoint": "/logout",
        "tags": ["auth"],
    },
]


class TestParity:
    def test_strict_catches_extra_field_both_directions(self) -> None:
        """Review P0-3's false-equivalence example: the v2 comparator
        dropped unknown fields and called it equivalent — STRICT must not."""
        new = [dict(_OLD[0]), dict(_OLD[1])]
        del new[0]["foo"]
        assert diff_artifacts(_OLD, new, ParityMode.STRICT)
        new2 = [dict(_OLD[0]), dict(_OLD[1])]
        new2[0]["bar"] = "NEW"
        assert diff_artifacts(_OLD, new2, ParityMode.STRICT)

    def test_strict_is_order_sensitive(self) -> None:
        new = [dict(_OLD[1]), dict(_OLD[0])]
        assert diff_artifacts(_OLD, new, ParityMode.STRICT)

    def test_strict_equal_passes(self) -> None:
        new = [dict(x) for x in _OLD]
        assert diff_artifacts(_OLD, new, ParityMode.STRICT) == []

    def test_semantic_ignores_order_and_non_whitelist(self) -> None:
        new = [dict(_OLD[1]), dict(_OLD[0])]
        for item in new:
            item.pop("foo", None)
        # Sorted equivalence + whitelist fields only.
        diffs = diff_artifacts([dict(_OLD[0]), dict(_OLD[1])], new, ParityMode.SEMANTIC)
        assert diffs == []

    def test_semantic_still_catches_whitelist_diff(self) -> None:
        new = [dict(_OLD[0]), dict(_OLD[1])]
        new[0]["title"] = "Login V2"
        assert any("title" in d for d in diff_artifacts(_OLD, new, ParityMode.SEMANTIC))

    def test_semantic_catches_count_mismatch(self) -> None:
        assert diff_artifacts(_OLD, _OLD[:1], ParityMode.SEMANTIC)

    def test_check_renumber(self) -> None:
        items = [{"id": "TC-001"}, {"id": "TC-003"}]
        problems = check_renumber(items)
        assert len(problems) == 1 and "TC-003" in problems[0]

    def test_diff_text(self) -> None:
        assert diff_text("a\n", "a") == []
        assert diff_text("a", "b")


# ----------------------------------------------------------------------
# UnitStatus taxonomy (B3.1)
# ----------------------------------------------------------------------


class TestUnitStatus:
    def test_seven_states_closed(self) -> None:
        assert len(UnitStatus) == 7
        assert {s.name for s in UnitStatus} == {
            "SUCCESS",
            "EMPTY",
            "INVALID",
            "TIMEOUT",
            "PROVIDER_ERROR",
            "VALIDATION_ERROR",
            "CANCELLED",
        }

    def test_recovery_policy_covers_all_seven(self) -> None:
        assert set(RECOVERY_POLICY) == set(UnitStatus)

    def test_undefined_member_fails_at_import_time(self) -> None:
        """The closed-set discipline: referencing UnitStatus.FAILED raises
        AttributeError (never a silent new state)."""
        missing = "FAILED"
        with pytest.raises(AttributeError):
            getattr(UnitStatus, missing)

    def test_outcome_mapping(self) -> None:
        from testagent.engine.model_profiles import Outcome

        assert unit_status_from_outcome(Outcome.OK) is UnitStatus.SUCCESS
        assert unit_status_from_outcome(Outcome.TRUNCATED_PARTIAL) is UnitStatus.SUCCESS
        assert unit_status_from_outcome(Outcome.BUDGET_EXHAUSTED) is UnitStatus.EMPTY
        assert unit_status_from_outcome(Outcome.TRANSIENT_EMPTY) is UnitStatus.PROVIDER_ERROR

    def test_unit_result_metadata(self) -> None:
        from testagent.engine.model_profiles import Outcome

        r = unit_result_from_engine(
            [{"a": 1}],
            outcome=Outcome.TRUNCATED_PARTIAL,
            finish_reason="length",
            engine_recovered=False,
        )
        assert r.status is UnitStatus.SUCCESS and r.partial and not r.engine_recovered

        r2 = unit_result_from_engine(
            [],
            outcome=Outcome.BUDGET_EXHAUSTED,
            finish_reason="length",
            engine_recovered=True,
        )
        assert r2.status is UnitStatus.EMPTY and r2.engine_recovered  # D2: no re-ask

        r3 = unit_result_from_engine(
            [{"a": 1}],
            outcome=Outcome.BUDGET_EXHAUSTED,
            finish_reason="length",
            engine_recovered=True,
        )
        assert r3.status is UnitStatus.SUCCESS and r3.engine_recovered

    def test_exception_mapping(self) -> None:
        from testagent.engine.llm_client import (
            LLMCallTimeoutError,
            LLMOutputTooLongError,
            ReasoningBudgetExhaustedError,
        )

        assert unit_status_from_exception(LLMCallTimeoutError("t")) is UnitStatus.TIMEOUT
        assert unit_status_from_exception(ReasoningBudgetExhaustedError("b")) is UnitStatus.EMPTY
        assert unit_status_from_exception(LLMOutputTooLongError("e")) is UnitStatus.PROVIDER_ERROR


# ----------------------------------------------------------------------
# Synthetic context (B3.4)
# ----------------------------------------------------------------------


class TestSyntheticContext:
    def test_declared_context_wins(self) -> None:
        m = Manifest.model_validate(
            _minimal_manifest(
                inputs=[{"name": "text", "kind": "text"}],
                template_context={"text": "EXPLICIT"},
            )
        )
        assert build_synthetic_context(m)["text"] == "EXPLICIT"

    def test_kind_samples_fill_undeclared(self) -> None:
        m = Manifest.model_validate(
            _minimal_manifest(inputs=[{"name": "swagger", "kind": "swagger"}])
        )
        ctx = build_synthetic_context(m)
        assert "GET" in ctx["swagger"]
        assert "GET" in ctx["endpoints_text"]

    def test_pipeline_vars_always_present(self) -> None:
        m = Manifest.model_validate(_minimal_manifest())
        ctx = build_synthetic_context(m)
        for key in ("output_language", "json_mode", "historical_cases"):
            assert key in ctx


class TestSessionIdGuard:
    """B-group fix: session ids arrive from the command line and are interpolated
    into file paths (``--resume``, ``checkpoint recover``), so they must not be
    able to escape the directory they are joined into."""

    def test_minted_and_handwritten_ids_pass(self) -> None:
        from testagent.pipeline.executor import validate_session_id

        assert validate_session_id("41e4ae12ab34") == "41e4ae12ab34"
        assert validate_session_id("rec1") == "rec1"  # fixtures use short names

    @pytest.mark.parametrize(
        "bad",
        ["../../etc/passwd", "a/b", "a\\b", "", "x" * 65, "id with space", "../x"],
    )
    def test_path_escaping_ids_rejected(self, bad: str) -> None:
        from testagent.pipeline.executor import validate_session_id

        with pytest.raises(ValueError, match="invalid session id"):
            validate_session_id(bad)

    def test_recover_snapshot_rejects_before_touching_the_filesystem(self, tmp_path: Path) -> None:
        from testagent.pipeline.executor import recover_snapshot

        with pytest.raises(ValueError, match="invalid session id"):
            recover_snapshot("../../outside", str(tmp_path))


class TestValidators:
    """Direct coverage for the text validators.

    The legacy chain tested these through its own generators; those files go
    away with the generators (B5.4), and a validator nobody calls directly is a
    validator whose error strings can change without anyone noticing — they feed
    the targeted re-ask prompt.
    """

    @staticmethod
    def _ctx(**parsed: object) -> object:
        from testagent.pipeline.inputs import TaskContext

        return TaskContext(parsed=dict(parsed))

    def test_strip_fences_handles_language_tag_and_prose_edges(self) -> None:
        from testagent.pipeline.validators import strip_fences

        assert strip_fences("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1"
        assert strip_fences("  import os\n") == "import os"
        assert strip_fences("```\nx=1\n```trailing prose```").startswith("x=1")

    def test_python_compile_error_string_names_the_file(self) -> None:
        from testagent.pipeline.manifest import ValidatorSpec
        from testagent.pipeline.validators import validate_text

        problems = validate_text(
            "def broken(:\n    pass\n", [ValidatorSpec(kind="python_compile")], self._ctx()
        )
        assert len(problems) == 1
        assert problems[0].startswith("python_compile: invalid syntax")

    def test_xml_validator_checks_declaration_closing_and_root_in_order(self) -> None:
        from testagent.pipeline.manifest import ValidatorSpec
        from testagent.pipeline.validators import validate_text

        spec = ValidatorSpec(
            kind="xml",
            root="jmeterTestPlan",
            declaration="<?xml",
            closing="</jmeterTestPlan>",
        )
        whole = '<?xml version="1.0"?><jmeterTestPlan><foo/></jmeterTestPlan>'
        assert validate_text(whole, [spec], self._ctx()) == []
        problems = validate_text("<jmeterTestPlan/>", [spec], self._ctx())
        assert any("declaration" in p for p in problems)
        assert any("closing" in p for p in problems)
        wrong_root = validate_text('<?xml version="1.0"?><other/>{', [spec], self._ctx())
        assert any("root tag" in p or "xml:" in p for p in wrong_root)

    def test_when_guard_skips_the_validator(self) -> None:
        from testagent.pipeline.manifest import ValidatorSpec
        from testagent.pipeline.validators import validate_text

        spec = ValidatorSpec(kind="xml", root="x", when={"script_format": "jmeter"})
        assert validate_text("<not xml", [spec], self._ctx(script_format="k6")) == []
        assert validate_text("<not xml", [spec], self._ctx(script_format="jmeter")) != []


class TestNoSilentTemplateFallback:
    """The legacy builders caught every template error and dropped to an inline
    prompt; the task-package chain must NOT, because ``tasks/<pkg>/prompts`` is
    the declared source of truth and a silently substituted prompt is a silent
    behavior change."""

    def test_missing_template_raises_instead_of_rendering_a_fallback(self) -> None:
        from jinja2 import TemplateNotFound

        from testagent.pipeline.registry import get_registry

        task = get_registry(REPO / "tasks").get("gui")
        with pytest.raises(TemplateNotFound):
            task.render("prompts/definitely_not_here.j2", {})
        with pytest.raises(TemplateNotFound):
            task.system_prompt("file:prompts/definitely_not_here.j2", {})
