"""E2E golden path (plan-c B4.9): the FULL chain through the real CLI
registration, registry, inputs, executor (runtime generate_unit), validators,
writers, snapshot and recover — against a fake LLM. Any pipeline refactor
must keep this green first.

Architecture gate (plan-c B4.11): pipeline modules must not import the
legacy generators / prompt_builder / conversation — enforced as a test so
the gate is CI-constant, not a human grep.
"""

import ast
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import CliRunner

# Imported at COLLECTION time (cwd = repo root) so the module-level
# register_tasks() resolves ./tasks correctly and the hidden `_example`
# command exists for every test below regardless of later chdir.
from testagent.cli import main
from testagent.engine.llm_client import LLMResponse

REPO = Path(__file__).parents[1]


# ----------------------------------------------------------------------
# B4.11 — import architecture gate
# ----------------------------------------------------------------------

FORBIDDEN_PREFIXES = (
    "testagent.generators",
    "testagent.engine.prompt_builder",
    "testagent.engine.conversation",
)


def _pipeline_modules() -> list[Path]:
    return sorted((REPO / "testagent" / "pipeline").glob("*.py"))


class TestArchitectureGate:
    def test_pipeline_never_imports_legacy_layers(self) -> None:
        """AST-level import scan: no pipeline module may import the legacy
        generators / prompt_builder / conversation (plan-c B4.11). The wiring
        of legacy + pipeline happens ONLY in the composition root
        (container.py / cli), which is not part of this gate."""
        violations: list[str] = []
        for path in _pipeline_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                targets: list[str] = []
                if isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    targets = [node.module]
                for t in targets:
                    if t.startswith(FORBIDDEN_PREFIXES):
                        violations.append(f"{path.name}: imports {t}")
        assert not violations, f"architecture gate violated: {violations}"

    def test_no_legacy_build_calls_in_new_layers(self) -> None:
        """The P4b grep gate, scripted (review #22): NEW layers (pipeline /
        web / cli commands) must not call the legacy build_*_prompt seams —
        they stay sealed until the pointer switch in B7. The legacy
        generators themselves are exempt (they ARE the legacy layer until
        B6b/B7 delete them)."""
        new_layers = [
            REPO / "testagent" / "pipeline",
            REPO / "testagent" / "web",
            REPO / "testagent" / "cli",
        ]
        seams = ("build_testcase_prompt", "build_api_prompt", "build_review_prompt")
        for layer in new_layers:
            for py in sorted(layer.rglob("*.py")):
                src = py.read_text(encoding="utf-8")
                for seam in seams:
                    assert seam not in src, f"{py.relative_to(REPO)} references legacy seam {seam}"

    def test_pipeline_never_reads_the_settings_singleton(self) -> None:
        """DI rule: the pipeline runs on the INJECTED Settings. Calling
        ``get_settings()`` inside a pipeline module ignores every override and
        makes behavior depend on which caller warmed the lru_cache first — the
        defect that turned the 2026-09-19 parity suite red under a full run."""
        for path in _pipeline_modules():
            src = path.read_text(encoding="utf-8")
            assert "get_settings(" not in src, f"{path.name} reads the settings singleton"

    # ------------------------------------------------------------------
    # plan-l L-1 — shared-model uniqueness machine checks (skeleton by T1,
    # the first settings.py writer). Owners extend the tracked sets as
    # their fields land via [shared-model] commits; LINK-S1a delivers the
    # full version. These methods are the ONLY home for the uniqueness
    # assertions (v3 ruling: no separate test class outside this gate).
    # ------------------------------------------------------------------

    #: Planned shared dataclass fields per owner (plan-k §4.1). T3 landed
    #: response_schemas; T8 (five identity fields) / T10 (binds,
    #: executability) / S1a (path_id, source_stage) extend as they land.
    _SHARED_MODEL_FIELDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "APIEndpoint": ("response_schemas",),
        "TestCase": (
            "binds",
            "executability",
            "scenario_operation",
            "scenario_scene",
            "scenario_variant",
            "equivalence_class",
            "covers_obligations",
            "path_id",
            "source_stage",
        ),
    }
    #: Planned shared Settings keys; T1 lands the first (AUDIT_DUMP_ENABLED).
    # Every non-LLM settings key added since the snapshot baseline (defect
    # ⑩: only 1 of 15 keys was tracked). LLMSettings sub-model keys are
    # excluded — the uniqueness test reads the Settings class body.
    _SHARED_SETTINGS_KEYS: tuple[str, ...] = (
        "audit_dump_enabled",
        "conflict_policy",
        "cases_budget",
        "links_enabled",
        "links_prose_enabled",
        "links_r6_min_score",
        "links_max_neighbors",
        "links_neighbor_chars",
        "links_cluster_size",
        "links_max_planned_paths",
        "links_max_hops",
        "links_l3b_max_calls",
        "links_seeding",
        "links_l0_max_chars",
        "split_mode",
        "single_doc_warn_tokens",
        "review_chunk_size",
        "review_max_chunk_failure_ratio",
        "review_max_prompt_chars",
        "gui_reference_max_cases",
        "gui_reference_max_chars",
    )
    #: Baseline field order snapshots — shared-model field slips must append,
    #: never reorder (plan-l L-1 hard constraint).
    _TESTCASE_BASELINE_FIELDS = (
        "id",
        "title",
        "description",
        "endpoint",
        "test_type",
        "priority",
        "preconditions",
        "steps",
        "expected_results",
        "tags",
    )
    _APIENDPOINT_BASELINE_FIELDS = (
        "method",
        "path",
        "summary",
        "description",
        "parameters",
        "request_body",
        "responses",
        "tags",
    )

    @staticmethod
    def _class_field_order(path: Path, class_name: str) -> list[str]:
        """Ordered top-level annotated fields of one class (AST-level)."""
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return [
                    t.target.id
                    for t in node.body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
                ]
        return []

    def test_shared_fields_declared_once(self) -> None:
        """Every planned shared field is declared exactly once in models.py."""
        models = REPO / "testagent" / "config" / "models.py"
        declared: dict[str, list[str]] = {}
        for class_name in self._SHARED_MODEL_FIELDS:
            declared[class_name] = self._class_field_order(models, class_name)
        for class_name, fields in self._SHARED_MODEL_FIELDS.items():
            in_class = declared[class_name]
            for field in fields:
                assert in_class.count(field) == 1, (
                    f"{class_name}.{field} declared {in_class.count(field)} times"
                )

    def test_settings_keys_declared_once(self) -> None:
        """Every planned shared settings key is declared exactly once."""
        settings = REPO / "testagent" / "config" / "settings.py"
        keys = self._class_field_order(settings, "Settings")
        for key in self._SHARED_SETTINGS_KEYS:
            assert keys.count(key) == 1, f"Settings.{key} declared {keys.count(key)} times"

    #: Shared model fields the links domain must never read via raw
    #: getattr-with-default (plan-links-v15 §3.1 / plan-k §4.2). Delivered
    #: by LINK-S1b; S2 modules and later links code are covered.
    _LINKS_SHARED_FIELDS = ("response_schemas", "binds", "executability", "path_id", "source_stage")
    # Actual links-domain modules on disk (defect ⑩: the original list
    # named a nonexistent links.py and missed four delivered modules while
    # the scan skipped missing files — a gate that silently scans nothing).
    _LINKS_MODULES = (
        "links_fields.py",
        "links_graph.py",
        "links_prompt_contract.py",
        "links_r6.py",
        "pathplanner.py",
        "linkcheck.py",
        "context_builder.py",
    )

    def test_links_domain_no_raw_getattr_shared_fields(self) -> None:
        """AST scan of the links domain: no three-argument getattr whose
        attribute name is a shared field (silent-fallback ban)."""
        pipeline_dir = REPO / "testagent" / "pipeline"
        violations: list[str] = []
        for name in self._LINKS_MODULES:
            path = pipeline_dir / name
            if not path.exists():
                continue  # not yet delivered stages stay skipped
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 3
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                    and node.args[1].value in self._LINKS_SHARED_FIELDS
                ):
                    violations.append(f"{name}:{node.lineno}")
        assert not violations, f"raw getattr on shared fields: {violations}"

    def test_baseline_field_order_unchanged(self) -> None:
        """TestCase/APIEndpoint baseline fields keep their original order
        and stay at the front of the dataclass (append-only field slips)."""
        models = REPO / "testagent" / "config" / "models.py"
        testcase = self._class_field_order(models, "TestCase")
        apiendpoint = self._class_field_order(models, "APIEndpoint")
        assert tuple(testcase[: len(self._TESTCASE_BASELINE_FIELDS)]) == (
            self._TESTCASE_BASELINE_FIELDS
        )
        assert tuple(apiendpoint[: len(self._APIENDPOINT_BASELINE_FIELDS)]) == (
            self._APIENDPOINT_BASELINE_FIELDS
        )


# ----------------------------------------------------------------------
# B4.9 — E2E golden path
# ----------------------------------------------------------------------


class _FakeLLMModule:
    """Patch target for the container's llm_client: a metadata-complete fake."""

    intent_capable = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def set_session_id(self, sid: str) -> None:
        self.session_id = sid  # type: ignore[attr-defined]

    async def averify(self) -> None:
        return None

    async def achat_with_meta(self, system: str, user: str, **kw: Any) -> LLMResponse:
        self.calls.append((system, user))
        return LLMResponse(text='[{"text": "golden"}]', finish_reason="stop")

    @property
    def primary_model(self) -> str:
        return "fake-model"


@pytest.fixture()
def fake_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _FakeLLMModule:
    """Wire a fake LLM into every Container the CLI constructs.

    dependency-injector providers capture function references at class
    definition, so patching module attributes is not enough — the CLASS-LEVEL
    provider is overridden instead, which every ``Container()`` instance
    inherits. Snapshots land under the tmp cwd's ``./output``.
    """
    from dependency_injector import providers

    from testagent.container import Container
    from testagent.pipeline.executor import PipelineExecutor
    from testagent.pipeline.runtime import build_generate_unit

    fake = _FakeLLMModule()
    monkeypatch.chdir(tmp_path)

    class _Settings:
        class LLM:
            max_concurrency = 2
            json_mode = False

        llm = LLM
        output_language = "english"
        output_dir = "./output"

    fake_executor = PipelineExecutor(fake, _Settings, generate_unit=build_generate_unit(fake))
    Container.pipeline_executor.override(providers.Object(fake_executor))
    yield fake
    Container.pipeline_executor.reset_override()


class TestGoldenPath:
    def test_example_task_end_to_end(self, fake_llm: _FakeLLMModule) -> None:
        """manifest load -> registry -> input parse -> render -> fake LLM ->
        jsonschema validation -> merge/renumber -> snapshot -> writer."""
        from testagent.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["_example", "--text", "hello", "--session", "gold123", "-o", "out/echo.json"],
            standalone_mode=False,
        )
        assert result.exit_code == 0, result.output

        payload = json.loads((Path("out/echo.json")).read_text(encoding="utf-8"))
        assert payload == [{"text": "golden", "id": "TC-001"}]
        # The fake LLM saw the rendered prompt (template + input).
        assert fake_llm.calls and "hello" in fake_llm.calls[0][1]
        # Snapshot exists and matches the artifact.
        snap = json.loads(
            (Path("output") / "gold123.pre_review_snapshot.json").read_text(encoding="utf-8")
        )
        assert snap["artifact"] == payload and snap["stage"] == "pre_review"

    def test_checkpoint_list_and_recover(self, fake_llm: _FakeLLMModule) -> None:
        from testagent.cli import main

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["_example", "--text", "hello", "--session", "rec1", "-o", "out/echo.json"],
            standalone_mode=False,
        )
        assert result.exit_code == 0, result.output

        listed = runner.invoke(main, ["checkpoint", "list"], standalone_mode=False)
        assert listed.exit_code == 0 and "rec1" in listed.output

        saved = runner.invoke(
            main,
            ["checkpoint", "recover", "rec1", "--save-as", "out/recovered.json"],
            standalone_mode=False,
        )
        assert saved.exit_code == 0, saved.output
        recovered = json.loads(Path("out/recovered.json").read_text(encoding="utf-8"))
        assert recovered == [{"text": "golden", "id": "TC-001"}]

    def test_tasks_validate_strict(self, fake_llm: _FakeLLMModule) -> None:
        """The repo's real _example package validates clean under --strict
        (registry overridden to the repo path — the tmp cwd has no tasks)."""
        from dependency_injector import providers

        from testagent.container import Container
        from testagent.pipeline.registry import get_registry

        Container.task_registry.override(providers.Object(get_registry(REPO / "tasks")))
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["tasks", "validate", "--strict"], standalone_mode=False)
            assert result.exit_code == 0, result.output
            assert "_example" in result.output
        finally:
            Container.task_registry.reset_override()

    def test_validate_strict_catches_broken_template(
        self, fake_llm: _FakeLLMModule, tmp_path: Path
    ) -> None:
        """A package whose template reference is missing fails --strict
        with exit 1 and names the broken reference."""
        from dependency_injector import providers

        from testagent.container import Container
        from testagent.pipeline.registry import Registry, TaskPackage

        tasks_dir = tmp_path / "brokenpkg"
        tasks_dir.mkdir(parents=True)
        (tasks_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": "broken",
                    "pipeline": {
                        "stages": [
                            {
                                "name": "s",
                                "template": "missing.j2",
                                "system_prompt": "inline:x",
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        manifest = json.loads((tasks_dir / "manifest.json").read_text(encoding="utf-8"))
        from testagent.pipeline.manifest import Manifest

        pkg = TaskPackage("broken", Manifest.model_validate(manifest), tasks_dir)
        Container.task_registry.override(providers.Object(Registry([pkg])))
        try:
            runner = CliRunner()
            result = runner.invoke(main, ["tasks", "validate", "--strict"], standalone_mode=False)
            assert result.exit_code == 1
            assert "missing.j2" in result.output
        finally:
            Container.task_registry.reset_override()


class TestFingerprintParity:
    def test_diff_fingerprints_detects_request_changes(self) -> None:
        """B4.10: parity fails when the REQUEST side differs even if the
        artifact would match (fake-LLM coincidence guard)."""
        from testagent.pipeline.fingerprint import FingerprintLog, diff_fingerprints

        old = FingerprintLog()
        old.record("m", "sys", "user-a", {"max_tokens": 100})
        new = FingerprintLog()
        new.record("m", "sys", "user-a", {"max_tokens": 200})
        assert diff_fingerprints(old, new)

        same = FingerprintLog()
        same.record("m", "sys", "user-a", {"max_tokens": 100})
        assert diff_fingerprints(old, same) == []
