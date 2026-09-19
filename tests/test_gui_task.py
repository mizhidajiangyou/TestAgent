"""FH1.1 / plan-k B5.2 tests: tasks/gui task package migration.

- Template: reverse-rename diff proves equivalence (rename endpoints_text
  back to endpoints -> byte-identical with the frozen gui_test_prompt.j2).
- Manifest: strict-valid, url default resolves from
  settings.gui.target_url (config/constants.DEFAULT_TARGET_URL).
- CLI E2E with a fake LLM: python_compile validator accepts a valid script,
  text contract writes the .py file; fingerprint (system,user) recorded.
"""

from pathlib import Path

import pytest
from click.testing import CliRunner

from testagent.cli import main
from testagent.pipeline.manifest import load_manifest

REPO = Path(__file__).parents[1]


class TestGuiTaskManifest:
    def test_manifest_is_strict_valid(self) -> None:
        manifest = load_manifest(REPO / "tasks" / "gui" / "manifest.json")
        assert manifest.name == "gui"
        assert manifest.pipeline.stages[0].output.contract == "text"

    def test_reverse_rename_diff_is_zero(self) -> None:
        """Migrated template with endpoints_text renamed back to endpoints
        must be byte-identical with the frozen gui_test_prompt.j2."""
        migrated = (REPO / "tasks" / "gui" / "prompts" / "main.j2").read_text(encoding="utf-8")
        frozen = (REPO / "templates" / "gui_test_prompt.j2").read_text(encoding="utf-8")
        reversed_rename = migrated.replace("{{ endpoints_text }}", "{{ endpoints }}").replace(
            "{% if endpoints_text %}", "{% if endpoints %}"
        )
        assert reversed_rename == frozen

    def test_url_default_from_settings(self) -> None:
        from testagent.config.constants import DEFAULT_TARGET_URL
        from testagent.config.settings import Settings

        settings = Settings()
        assert settings.gui.target_url == DEFAULT_TARGET_URL


class TestGuiPipelineE2E:
    @pytest.fixture()
    def gui_fake_llm(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        from dependency_injector import providers

        from tests.test_pipeline_e2e import _FakeLLMModule

        monkeypatch.chdir(tmp_path)
        fake = _FakeLLMModule()

        orig = fake.achat_with_meta

        async def script_response(system: str, user: str, **kw):
            resp = await orig(system, user, **kw)
            resp.text = (
                "import pytest\nfrom playwright.sync_api import Page, expect\n\n\n"
                "def test_login(page: Page) -> None:\n"
                "    page.goto('https://example.com')\n"
                "    expect(page).to_have_url('https://example.com')\n"
            )
            return resp

        fake.achat_with_meta = script_response

        from testagent.container import Container
        from testagent.pipeline.executor import PipelineExecutor
        from testagent.pipeline.runtime import build_generate_unit

        class _Settings:
            class LLM:
                max_concurrency = 2
                json_mode = False

            llm = LLM
            output_language = "english"
            output_dir = "./output"

        executor = PipelineExecutor(fake, _Settings, generate_unit=build_generate_unit(fake))
        Container.pipeline_executor.override(providers.Object(executor))
        yield fake
        Container.pipeline_executor.reset_override()

    def test_cli_run_writes_compilable_script(self, gui_fake_llm, tmp_path: Path) -> None:
        doc = tmp_path / "req.md"
        doc.write_text("# GUI\n\nLogin flow.\n", encoding="utf-8")
        out = tmp_path / "gui_test.py"
        result = CliRunner().invoke(
            main,
            ["gui", "-r", str(doc), "--url", "https://example.test", "-o", str(out)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        script = out.read_text(encoding="utf-8")
        compile(script, "gui_test.py", "exec")  # python_compile validator parity
        # fingerprint: the fake saw the rendered prompt
        assert gui_fake_llm.calls, "LLM must have been called once (text contract)"
        assert "https://example.test" in gui_fake_llm.calls[0][1]
