"""FH2.3 tests: tasks/testcase migration equivalence + pipeline E2E.

- Template migration: byte-identical (sha256) with the frozen legacy
  prompts — the strongest reverse-rename equivalence (no rename at all).
- Manifest: strict-validated (registry), phase1=per_input, phase2 batch=2,
  merge dedup keys == H1 key set, truncation wired.
- Pipeline E2E (fake LLM): per_input + batch stages produce items, merge
  dedups by (title, endpoint, test_type), renumber assigns TC-001.. —
  matching the H1 contract.
"""

import hashlib
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from testagent.cli import main
from testagent.pipeline.manifest import load_manifest

REPO = Path(__file__).parents[1]

_TEMPLATE_EQUIVALENCE = [
    ("testcase_prompt.j2", "templates/testcase_prompt.j2"),
    ("api_prompt.j2", "templates/api_prompt.j2"),
    ("review_prompt.j2", "templates/review_prompt.j2"),
]


class TestTemplateMigration:
    def test_templates_byte_identical_to_frozen(self) -> None:
        for migrated, frozen in _TEMPLATE_EQUIVALENCE:
            migrated_bytes = (REPO / "tasks" / "testcase" / "prompts" / migrated).read_bytes()
            frozen_bytes = (REPO / frozen).read_bytes()
            assert (
                hashlib.sha256(migrated_bytes).hexdigest()
                == hashlib.sha256(frozen_bytes).hexdigest()
            ), f"{migrated} diverged from frozen {frozen}"

    def test_manifest_is_strict_valid(self) -> None:
        manifest = load_manifest(REPO / "tasks" / "testcase" / "manifest.json")
        assert manifest.name == "testcase"
        stage_names = [s.name for s in manifest.pipeline.stages]
        assert stage_names == ["phase1", "phase2_api"]

    def test_phase2_batch_size_is_two(self) -> None:
        manifest = load_manifest(REPO / "tasks" / "testcase" / "manifest.json")
        phase2 = manifest.pipeline.stages[1]
        assert phase2.split.by == "batch"
        assert phase2.split.batch_size == 2
        assert phase2.split.input == "endpoints"

    def test_merge_dedup_keys_match_h1(self) -> None:
        manifest = load_manifest(REPO / "tasks" / "testcase" / "manifest.json")
        keys = manifest.pipeline.merge.dedup.keys
        assert keys == ["title", "endpoint", "test_type"]
        assert manifest.pipeline.merge.renumber.format == "TC-{i:03d}"

    def test_truncation_enabled_with_scope_field(self) -> None:
        manifest = load_manifest(REPO / "tasks" / "testcase" / "manifest.json")
        spec = manifest.pipeline.truncation
        assert spec.enabled is True
        assert spec.scope_key_field == "endpoint"

    def test_schema_covers_quality_fields(self) -> None:
        schema = json.loads(
            (REPO / "tasks" / "testcase" / "schema" / "testcase.schema.json").read_text(
                encoding="utf-8"
            )
        )
        props = set(schema["properties"])
        assert {"binds", "executability", "path_id", "source_stage"} <= props
        assert {
            "scenario_operation",
            "scenario_scene",
            "scenario_variant",
            "equivalence_class",
            "covers_obligations",
        } <= props


class TestPipelineE2E:
    def _requirements(self, tmp_path: Path) -> str:
        doc = tmp_path / "req.md"
        doc.write_text(
            "# Module\n\n## REQ-001\n\nUser management.\n\n## REQ-002\n\nOrders.\n",
            encoding="utf-8",
        )
        return str(doc)

    def _spec(self, tmp_path: Path) -> str:
        spec = tmp_path / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "paths": {
                        "/users": {
                            "get": {"tags": ["users"], "responses": {"200": {"description": "OK"}}},
                            "post": {
                                "tags": ["users"],
                                "responses": {"201": {"description": "OK"}},
                            },
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return str(spec)

    @pytest.fixture()
    def engine_fake_llm(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """fake_llm variant returning SCHEMA-VALID testcase items, wired
        through build_engine_generate_unit (FH2.1 exercised via the CLI)."""
        from dependency_injector import providers

        from tests.test_pipeline_e2e import _FakeLLMModule

        monkeypatch.chdir(tmp_path)
        fake = _FakeLLMModule()

        orig = fake.achat_with_meta

        async def valid_response(system: str, user: str, **kw):
            resp = await orig(system, user, **kw)
            resp.text = json.dumps(
                [
                    {
                        "id": "TC-XXX",
                        "title": "golden case",
                        "endpoint": "GET /users",
                        "test_type": "functional",
                        "priority": "high",
                        "steps": ["call GET /users"],
                        "expected_results": ["200 with the user list"],
                    },
                    {
                        "id": "TC-XXX",
                        "title": "another case",
                        "endpoint": "POST /users",
                        "test_type": "negative",
                        "priority": "medium",
                        "steps": ["call POST /users with a bad body"],
                        "expected_results": ["400 with a validation error"],
                    },
                ]
            )
            return resp

        fake.achat_with_meta = valid_response

        from testagent.container import Container
        from testagent.pipeline.executor import PipelineExecutor
        from testagent.pipeline.runtime import build_engine_generate_unit

        class _Settings:
            class LLM:
                max_concurrency = 2
                json_mode = False
                max_output_tokens = 16000

            llm = LLM
            output_language = "english"
            output_dir = "./output"

        executor = PipelineExecutor(
            fake,
            _Settings,
            generate_unit=build_engine_generate_unit(
                fake, output_token_cap=_Settings.llm.max_output_tokens
            ),
        )
        Container.pipeline_executor.override(providers.Object(executor))
        yield fake
        Container.pipeline_executor.reset_override()

    def test_cli_run_produces_renumbered_cases(self, engine_fake_llm, tmp_path: Path) -> None:
        out = tmp_path / "cases.json"
        # Task packages control review via REVIEW_ENABLED (no --review flag
        # on the dynamic command); default is false.
        result = CliRunner().invoke(
            main,
            [
                "testcase",
                "-r",
                self._requirements(tmp_path),
                "-s",
                self._spec(tmp_path),
                "-o",
                str(out),
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        cases = json.loads(out.read_text(encoding="utf-8"))
        assert cases, "artifact must not be empty (clicommand exit-1 gate)"
        ids = [c["id"] for c in cases]
        assert ids == [f"TC-{i:03d}" for i in range(1, len(ids) + 1)], (
            "H1 renumber timing: ids assigned after merge, sequential"
        )
        assert engine_fake_llm.calls, "engine-backed units must call the LLM"
