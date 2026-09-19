"""B5.1 tasks/perf — fingerprint parity + run-through tests (plan-d B5.1 gate).

Gate definition (plan-d v3 B5.1): ``validate --strict`` + **fingerprint
parity zero-diff (I2 whitelist)** + fake-LLM run-through.

What "fingerprint parity" means here concretely: driving the LEGACY
``PerformanceGenerator`` and the NEW pipeline (executor + runtime +
tasks/perf package) with equivalent inputs and scripted responses, the
observable LLM-request boundary — the ordered sequence of
``(system_prompt, user_prompt)`` pairs, per client role — must be
IDENTICAL, and so must the final artifact. That is exactly the I2
whitelist (model + prompt pair; params are bare at this boundary and
logical labels never leak into requests — see the label-mapping test).

Template migration discipline is pinned by reverse-rename diffs: the
task-package templates must be byte-identical to the frozen legacy
templates after undoing the documented variable renames
(``endpoints``→``endpoints_text``, ``config.X``→``X``,
``script_kind``→``script_format``).

Registered divergences (B5.3 failure-suite scope, see task.md):
- invalid jmeter generation: legacy raises ValueError; pipeline records
  an INVALID unit and yields an empty artifact;
- review of an invalid generation: legacy reviews the invalid script;
  the pipeline drops it before review (no repair path yet);
- meta.json path/shape (``perf_test.js.meta.json`` vs legacy
  ``perf_test.meta.json``; status dict vs legacy reviewed flag);
- timeout/provider-error: legacy propagates the exception; the pipeline
  records the failed unit (``fan_out_recover=false`` matches legacy
  empty-response semantics: no re-ask).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from testagent.config.models import PerfGenInput, PerformanceConfig
from testagent.engine.llm_client import LLMResponse
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.performance_generator import PerformanceGenerator
from testagent.parsers.swagger_parser import SwaggerParser
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.fingerprint import FingerprintLog
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.manifest import load_manifest
from testagent.pipeline.registry import TaskPackage
from testagent.pipeline.runtime import (
    REVIEW_FAILED,
    REVIEWED,
    build_generate_unit,
    build_review_runner,
)
from testagent.pipeline.writers import write_artifact


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


REPO = Path(__file__).parents[1]
TASK_ROOT = REPO / "tasks" / "perf"
SWAGGER = REPO / "examples" / "ecommerce_swagger.json"

#: One source of truth for the perf parameters on both sides.
PERF_KW: dict[str, Any] = {
    "base_url": "https://api.example.com",
    "virtual_users": 100,
    "duration_seconds": 300,
    "ramp_up_seconds": 60,
    "think_time_ms": 500,
    "auth_type": "none",
}

#: Script constants double as EXPECTED artifacts — both the legacy
#: ``_extract_script`` and the pipeline ``strip_fences`` strip surrounding
#: whitespace, so the constants are the post-strip forms.
K6_SCRIPT = "import http from 'k6/http';\nexport default function () {}"
K6_SCRIPT_FENCED = f"```javascript\n{K6_SCRIPT}\n```"
K6_REVISED = "import http from 'k6/http';\nexport default function () { sleep(1); }"
JMX_SCRIPT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<jmeterTestPlan version="1.2">\n</jmeterTestPlan>'
)
JMX_SCRIPT_FENCED = f"```xml\n{JMX_SCRIPT}\n```"
JMX_REVISED = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<jmeterTestPlan version="1.2">\n'
    "  <hashTree/>\n"
    "</jmeterTestPlan>"
)


# ----------------------------------------------------------------------
# Fake LLM: records the (role, system, user) boundary on every path the
# legacy generator (sync chat/chat_with_meta) and the pipeline (async
# achat_with_meta) use.
# ----------------------------------------------------------------------


class _SubClient:
    def __init__(self, script: list[Any], role: str, sink: list[tuple[str, str, str]]) -> None:
        self._script = list(script)
        self._role = role
        self._sink = sink
        self.model_name = f"{role}-m"

    def _respond(self, system: str, user: str) -> Any:
        self._sink.append((self._role, system, user))
        idx = sum(1 for r, *_ in self._sink if r == self._role) - 1
        return self._script[min(idx, len(self._script) - 1)]

    @staticmethod
    def _as_response(item: Any) -> LLMResponse:
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=str(item), finish_reason="stop")

    def chat(self, system: str, user: str) -> str:
        return self._as_response(self._respond(system, user)).text

    def chat_with_meta(self, system: str, user: str) -> LLMResponse:
        return self._as_response(self._respond(system, user))

    async def achat_with_meta(self, system: str, user: str) -> LLMResponse:
        return self.chat_with_meta(system, user)


class ParityFakeLLM:
    """Multi-model-shaped double: primary + secondary sub-clients."""

    intent_capable = True
    primary_model = "primary-m"

    def __init__(self, primary_script: list[Any], secondary_script: list[Any] | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self.primary = _SubClient(primary_script, "primary", self.calls)
        self._secondary = _SubClient(
            secondary_script if secondary_script is not None else primary_script,
            "secondary",
            self.calls,
        )
        self.session_id = ""

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [(system, user) for _, system, user in self.calls]

    @property
    def roles(self) -> list[str]:
        return [role for role, *_ in self.calls]

    def secondary_client(self) -> _SubClient:
        return self._secondary

    def set_session_id(self, sid: str) -> None:
        self.session_id = sid

    async def averify(self) -> None:
        return None

    def chat(self, system: str, user: str) -> str:
        return self.primary.chat(system, user)

    def chat_with_meta(self, system: str, user: str) -> LLMResponse:
        return self.primary.chat_with_meta(system, user)

    async def achat_with_meta(self, system: str, user: str) -> LLMResponse:
        return await self.primary.achat_with_meta(system, user)


def _settings(
    tmp_path: Path,
    *,
    review_enabled: bool = False,
    output_language: str = "english",
    script_format: str = "k6",
) -> type:
    class Perf:
        base_url = PERF_KW["base_url"]
        virtual_users = PERF_KW["virtual_users"]
        duration_seconds = PERF_KW["duration_seconds"]
        ramp_up_seconds = PERF_KW["ramp_up_seconds"]
        think_time_ms = PERF_KW["think_time_ms"]
        auth_type = PERF_KW["auth_type"]

    class LLM:
        max_concurrency = 2
        json_mode = False

    fmt = script_format  # class bodies cannot see the enclosing name directly
    lang = output_language
    review = review_enabled
    out_dir = str(tmp_path)

    class Settings:
        llm = LLM
        perf = Perf
        script_format = fmt
        output_dir = out_dir
        output_language = lang
        review_enabled = review
        review_max_rounds = 2

    return Settings


def _task() -> TaskPackage:
    manifest = load_manifest(TASK_ROOT / "manifest.json")
    return TaskPackage(manifest.name, manifest, TASK_ROOT)


_ENDPOINTS = SwaggerParser().parse(str(SWAGGER))


def _run_legacy(
    fake: ParityFakeLLM,
    *,
    script_format: str,
    output_language: str,
    review_enabled: bool,
) -> str:
    generator = PerformanceGenerator(
        llm_client=fake,
        prompt_builder=PromptBuilder(),
        script_format=script_format,
        output_language=output_language,
        review_enabled=review_enabled,
        review_llm_client=fake.secondary_client(),
        review_max_rounds=2,
    )
    return generator.generate(
        PerfGenInput(endpoints=_ENDPOINTS, config=PerformanceConfig(**PERF_KW))
    )


async def _run_pipeline(
    fake: ParityFakeLLM,
    settings: type,
    *,
    script_format: str,
    tmp_path: Path,
) -> Any:
    task = _task()
    ctx = parse_inputs(
        task.manifest, {"swagger": str(SWAGGER), "script_format": script_format}, settings
    )
    executor = PipelineExecutor(
        fake,
        settings,
        generate_unit=build_generate_unit(fake),
        review_runner=build_review_runner(fake),
    )
    return await executor.arun(task, ctx, session_id="parity1")


# ----------------------------------------------------------------------
# Template migration discipline (reverse-rename diffs)
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Frozen legacy prompt bytes (B5.4 / plan-k 删除门前置固化)
#
# Recorded from ``templates/*.j2`` at commit 4f73474, i.e. from the legacy
# generator that B5.4 deletes. The reverse-rename diff below is what proves the
# task-package copy still equals those historical bytes — the comparison
# survives the deletion instead of disappearing with it.
# ----------------------------------------------------------------------
def _sha_full(text: str) -> str:
    """Full hex digest (the module's ``_sha`` above truncates to 12 for the
    fingerprint comparisons; the frozen template bytes need the whole digest)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


LEGACY_PERFORMANCE_PROMPT_SHA256 = (
    "b13767dc92ac2ab2a40f2c7cd2c97d5fd38190ec77e4255ed79cdcabf6783df5"
)
LEGACY_SCRIPT_REVIEW_PROMPT_SHA256 = (
    "7b4a1fccd2a5f2ec35253fa3b66150905c1e072909660f7ca6527853e9da6a5c"
)


class TestTemplateDiscipline:
    def test_generation_template_is_renamed_copy_of_legacy(self) -> None:
        adapted = (TASK_ROOT / "prompts" / "performance.j2").read_text(encoding="utf-8")
        reversed_ = (
            adapted.replace("{{ endpoints_text }}", "{{ endpoints }}")
            .replace("{{ virtual_users", "{{ config.virtual_users")
            .replace("{{ duration_seconds", "{{ config.duration_seconds")
            .replace("{{ ramp_up_seconds", "{{ config.ramp_up_seconds")
            .replace("{{ think_time_ms", "{{ config.think_time_ms")
            .replace("{{ base_url", "{{ config.base_url")
            .replace("{{ auth_type", "{{ config.auth_type")
            .replace("{{ (think_time_ms", "{{ (config.think_time_ms")
        )
        assert _sha_full(reversed_) == LEGACY_PERFORMANCE_PROMPT_SHA256

    def test_review_template_is_renamed_copy_of_legacy(self) -> None:
        adapted = (TASK_ROOT / "prompts" / "script_review.j2").read_text(encoding="utf-8")
        reversed_ = adapted.replace("script_format", "script_kind")
        assert _sha_full(reversed_) == LEGACY_SCRIPT_REVIEW_PROMPT_SHA256

    def test_package_validates_strict(self) -> None:
        task = _task()
        assert task.validate_renderable() == []
        assert task.validate_references() == []


# ----------------------------------------------------------------------
# Fingerprint parity matrix (the B5.1 hard gate)
# ----------------------------------------------------------------------


class TestFingerprintParity:
    @pytest.mark.parametrize("script_format", ["k6", "jmeter"])
    @pytest.mark.parametrize("review_enabled", [False, True])
    @pytest.mark.parametrize("output_language", ["english", "chinese"])
    @pytest.mark.parametrize("fenced", [False, True], ids=["plain", "fenced"])
    async def test_request_sequences_and_artifacts_match(
        self,
        tmp_path: Path,
        script_format: str,
        review_enabled: bool,
        output_language: str,
        fenced: bool,
    ) -> None:
        if script_format == "k6":
            gen = K6_SCRIPT_FENCED if fenced else K6_SCRIPT
            r1, r2 = K6_REVISED, K6_SCRIPT
        else:
            gen = JMX_SCRIPT_FENCED if fenced else JMX_SCRIPT
            r1, r2 = JMX_REVISED, JMX_SCRIPT

        if review_enabled:
            primary_script: list[Any] = [gen, r2]
            secondary_script: list[Any] = [r1]
        else:
            primary_script, secondary_script = [gen], None

        legacy_fake = ParityFakeLLM(primary_script, secondary_script)
        legacy_artifact = _run_legacy(
            legacy_fake,
            script_format=script_format,
            output_language=output_language,
            review_enabled=review_enabled,
        )

        pipe_fake = ParityFakeLLM(primary_script, secondary_script)
        settings = _settings(
            tmp_path, review_enabled=review_enabled, output_language=output_language
        )
        result = await _run_pipeline(
            pipe_fake, settings, script_format=script_format, tmp_path=tmp_path
        )

        # The hard gate: identical ordered (system, user) request pairs.
        assert pipe_fake.pairs == legacy_fake.pairs, (
            f"request mismatch for fmt={script_format} review={review_enabled} "
            f"lang={output_language} fenced={fenced}"
        )
        # Client-role assignment matches too (generation=primary, odd review
        # rounds=secondary, even=primary — the alternation contract).
        assert pipe_fake.roles == legacy_fake.roles
        # Request counts: 1 generation (+2 review rounds when enabled).
        assert len(pipe_fake.calls) == (3 if review_enabled else 1)
        # Artifact parity.
        assert result.artifact == legacy_artifact

    async def test_from_settings_defaults_parity(self, tmp_path: Path) -> None:
        """No CLI overrides: the pipeline resolves every default from
        settings (script_format, perf.*) exactly as the legacy CLI did
        (``value or settings.perf.x`` / ``settings.script_format``)."""
        legacy_fake = ParityFakeLLM([JMX_SCRIPT])
        legacy_artifact = _run_legacy(
            legacy_fake, script_format="jmeter", output_language="english", review_enabled=False
        )
        pipe_fake = ParityFakeLLM([JMX_SCRIPT])
        settings = _settings(tmp_path, script_format="jmeter")  # raw omits script_format
        task = _task()
        ctx = parse_inputs(task.manifest, {"swagger": str(SWAGGER)}, settings)
        executor = PipelineExecutor(
            pipe_fake,
            settings,
            generate_unit=build_generate_unit(pipe_fake),
            review_runner=build_review_runner(pipe_fake),
        )
        result = await executor.arun(task, ctx, session_id="parity3")
        assert pipe_fake.pairs == legacy_fake.pairs
        assert result.artifact == legacy_artifact == JMX_SCRIPT
        # The jmeter branch actually rendered (not the k6 default).
        assert "JMeter JMX test plan" in pipe_fake.pairs[0][1]

    async def test_non_default_params_parity(self, tmp_path: Path) -> None:
        """CLI-level parameter overrides must reach the prompts identically
        (pipeline raw inputs vs legacy PerformanceConfig)."""
        overrides = {
            "base_url": "https://staging.example.dev",
            "virtual_users": 42,
            "duration_seconds": 120,
            "ramp_up_seconds": 15,
            "think_time_ms": 1500,
            "auth_type": "bearer",
        }
        legacy_fake = ParityFakeLLM([K6_SCRIPT])
        generator = PerformanceGenerator(
            llm_client=legacy_fake,
            prompt_builder=PromptBuilder(),
            script_format="k6",
            output_language="english",
            review_enabled=False,
        )
        legacy_artifact = generator.generate(
            PerfGenInput(endpoints=_ENDPOINTS, config=PerformanceConfig(**overrides))
        )

        pipe_fake = ParityFakeLLM([K6_SCRIPT])
        settings = _settings(tmp_path)
        task = _task()
        ctx = parse_inputs(
            task.manifest,
            {"swagger": str(SWAGGER), "script_format": "k6", **overrides},
            settings,
        )
        executor = PipelineExecutor(
            pipe_fake,
            settings,
            generate_unit=build_generate_unit(pipe_fake),
            review_runner=build_review_runner(pipe_fake),
        )
        result = await executor.arun(task, ctx, session_id="parity2")
        assert pipe_fake.pairs == legacy_fake.pairs
        assert result.artifact == legacy_artifact == K6_SCRIPT
        # The overrides actually reached the prompt (not just equal-equal).
        assert "- Virtual Users: 42" in pipe_fake.pairs[0][1]
        assert "- Base URL: https://staging.example.dev" in pipe_fake.pairs[0][1]
        assert "sleep(1 )" in pipe_fake.pairs[0][1]  # think_time_ms 1500 // 1000

    async def test_empty_response_parity_k6(self, tmp_path: Path) -> None:
        """Empty first response (k6): both sides make exactly ONE request,
        return an empty artifact and burn no review round (registered
        divergence is jmeter-only: legacy raises, pipeline INVALID)."""
        legacy_fake = ParityFakeLLM([""], [])
        legacy_artifact = _run_legacy(
            legacy_fake, script_format="k6", output_language="english", review_enabled=True
        )
        pipe_fake = ParityFakeLLM([""], [])
        settings = _settings(tmp_path, review_enabled=True)
        result = await _run_pipeline(pipe_fake, settings, script_format="k6", tmp_path=tmp_path)
        assert legacy_artifact == ""
        assert result.artifact == ""
        assert len(legacy_fake.calls) == len(pipe_fake.calls) == 1
        assert pipe_fake.pairs == legacy_fake.pairs
        assert result.review_meta is None

    async def test_fingerprint_log_matches_captured_pairs(self, tmp_path: Path) -> None:
        """The B4.10 FingerprintLog machinery stays honest on the new side:
        recorded entries equal the fake-observed boundary (sha of the pair),
        and the unit label is the documented pipeline label (I2: labels
        never leak into the request itself — the pair comparison above is
        the semantic gate; this pins the recording machinery)."""
        fake = ParityFakeLLM([K6_SCRIPT])
        settings = _settings(tmp_path)
        task = _task()
        ctx = parse_inputs(
            task.manifest, {"swagger": str(SWAGGER), "script_format": "k6"}, settings
        )
        log = FingerprintLog()
        executor = PipelineExecutor(
            fake,
            settings,
            generate_unit=build_generate_unit(fake),
            review_runner=build_review_runner(fake),
        )
        await executor.arun(task, ctx, session_id="fp1", fingerprint_log=log)
        assert len(log.entries) == 1
        entry = log.entries[0]
        assert entry.label == "perf:script"  # CALL_LABEL: <task>:<stage>
        assert entry.model == "primary-m"
        system, user = fake.pairs[0]
        assert entry.system_prompt_sha == _sha(system)
        assert entry.user_prompt_sha == _sha(user)


# ----------------------------------------------------------------------
# Run-through: pipeline-side behaviours (fake-LLM driven)
# ----------------------------------------------------------------------


class TestPerfRunThrough:
    async def test_k6_happy_path_no_review(self, tmp_path: Path) -> None:
        fake = ParityFakeLLM([K6_SCRIPT])
        settings = _settings(tmp_path, review_enabled=False)
        result = await _run_pipeline(fake, settings, script_format="k6", tmp_path=tmp_path)
        assert result.artifact == K6_SCRIPT
        assert result.units_failed == 0
        assert result.review_meta is None  # no review -> no meta (legacy parity)

    async def test_k6_reviewed_meta_and_writer(self, tmp_path: Path) -> None:
        fake = ParityFakeLLM([K6_SCRIPT, K6_SCRIPT], [K6_REVISED])
        settings = _settings(tmp_path, review_enabled=True)
        result = await _run_pipeline(fake, settings, script_format="k6", tmp_path=tmp_path)
        assert result.artifact == K6_SCRIPT  # round 2 (primary) wins
        assert result.review_meta is not None
        assert result.review_meta["status"] == REVIEWED
        assert result.review_meta["rounds_succeeded"] == 2
        # Writer: .js extension keyed by script_format + meta sidecar.
        path = write_artifact(
            _task().manifest,
            result.artifact,
            tmp_path / "perf_test.js",
            "text",
            ctx={"script_format": "k6"},
            review_meta=result.review_meta,
        )
        assert path == tmp_path / "perf_test.js"
        assert path.read_text(encoding="utf-8") == K6_SCRIPT
        assert (tmp_path / "perf_test.js.meta.json").exists()

    async def test_jmeter_extension_and_review_fallback(self, tmp_path: Path) -> None:
        # Review round 1 returns garbage, round 2 returns a VALID jmx — the
        # legacy parse guard (validate closure) must reject the garbage and
        # the loop's built-in retry re-asks within the round.
        fake = ParityFakeLLM([JMX_SCRIPT, JMX_SCRIPT], ["not xml at all", JMX_REVISED])
        settings = _settings(tmp_path, review_enabled=True)
        result = await _run_pipeline(fake, settings, script_format="jmeter", tmp_path=tmp_path)
        assert result.artifact == JMX_SCRIPT
        assert result.review_meta is not None
        assert result.review_meta["status"] == REVIEWED
        path = write_artifact(
            _task().manifest,
            result.artifact,
            tmp_path / "perf_test.js",
            "text",
            ctx={"script_format": "jmeter"},
        )
        assert path == tmp_path / "perf_test.jmx"

    async def test_jmeter_invalid_review_candidate_rejected(self, tmp_path: Path) -> None:
        # Every review attempt returns structurally invalid JMX (wrong
        # root): each round's parse fails (validate closure) and the
        # built-in retry fails too -> all rounds failed -> REVIEW_FAILED
        # -> snapshot (original) kept.
        bad = '<?xml version="1.0" encoding="UTF-8"?>\n<wrongRoot/>'
        fake = ParityFakeLLM([JMX_SCRIPT, bad], [bad])
        settings = _settings(tmp_path, review_enabled=True)
        result = await _run_pipeline(fake, settings, script_format="jmeter", tmp_path=tmp_path)
        assert result.artifact == JMX_SCRIPT  # snapshot fallback
        assert result.review_meta is not None
        assert result.review_meta["status"] == REVIEW_FAILED

    async def test_jmeter_invalid_generation_is_invalid_unit(self, tmp_path: Path) -> None:
        # REGISTERED DIVERGENCE (B5.3 failure suite): legacy raises
        # ValueError here; the pipeline records an INVALID unit and yields
        # an empty artifact.
        fake = ParityFakeLLM(["<not-jmx/>\n"])
        settings = _settings(tmp_path, review_enabled=True)
        result = await _run_pipeline(fake, settings, script_format="jmeter", tmp_path=tmp_path)
        assert result.artifact == ""
        assert result.units_failed == 1
        assert result.review_meta is None  # empty artifact -> review skipped

    async def test_empty_generation_skips_review(self, tmp_path: Path) -> None:
        # Legacy parity: ``if review_enabled and script`` — an empty first
        # response yields an empty artifact with NO review round burned
        # (fan_out_recover=false: no re-ask, matching legacy single-shot).
        fake = ParityFakeLLM([""])
        settings = _settings(tmp_path, review_enabled=True)
        result = await _run_pipeline(fake, settings, script_format="k6", tmp_path=tmp_path)
        assert result.artifact == ""
        assert len(fake.calls) == 1
        assert result.review_meta is None

    async def test_review_uses_declared_system_prompt(self, tmp_path: Path) -> None:
        fake = ParityFakeLLM([K6_SCRIPT, K6_SCRIPT], [K6_REVISED])
        settings = _settings(tmp_path, review_enabled=True, output_language="chinese")
        await _run_pipeline(fake, settings, script_format="k6", tmp_path=tmp_path)
        # Round 1 (secondary) carries the migrated review system prompt,
        # including the chinese language hint (legacy assembly).
        review_system = fake.calls[1][1]
        assert review_system.startswith("You are a meticulous senior performance engineer")
        assert review_system.endswith(
            "Write all script comments, user-facing labels and summary text "
            "in Simplified Chinese (keep code keywords and identifiers in English)."
        )


def test_cli_writes_raw_script_not_json_string(tmp_path: Path) -> None:
    """Regression found by the 2026-09-20 real-model acceptance run.

    ``testagent perf -o x.js`` wrote a JSON-encoded string instead of a script:
    the dynamic CLI reads its format from a key the generated option never sets,
    so the writer fell back to ``json`` and quoted the whole script (literal
    ``\\n`` in the file). A .js file is not compilable Python, so nothing in the
    suite noticed — the gui twin of this test asserts the same thing for .py.
    """
    from click.testing import CliRunner
    from dependency_injector import providers

    from testagent.cli import main
    from testagent.container import Container
    from testagent.pipeline.executor import PipelineExecutor
    from testagent.pipeline.runtime import build_generate_unit
    from tests.web_fakes import ScriptedLLM

    class _ScriptLLM(ScriptedLLM):
        async def achat_with_meta(self, system, user, response_format=None, max_tokens=None):  # type: ignore[override]
            resp = await super().achat_with_meta(system, user, response_format, max_tokens)
            resp.text = K6_SCRIPT
            return resp

    llm = _ScriptLLM([K6_SCRIPT])
    settings = _settings(tmp_path, review_enabled=False)
    executor = PipelineExecutor(llm, settings, generate_unit=build_generate_unit(llm))
    Container.pipeline_executor.override(providers.Object(executor))
    try:
        spec = tmp_path / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "openapi": "3.0.0",
                    "paths": {
                        "/users": {
                            "get": {"tags": ["u"], "responses": {"200": {"description": "ok"}}}
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        out = tmp_path / "perf.js"
        result = CliRunner().invoke(
            main,
            ["perf", "-s", str(spec), "--base-url", "https://x.test", "-o", str(out)],
            catch_exceptions=False,
        )
    finally:
        Container.pipeline_executor.reset_override()

    assert result.exit_code == 0, result.output
    written = out.read_text(encoding="utf-8")
    assert not written.startswith('"'), f"writer JSON-encoded the script: {written[:60]}"
    assert chr(92) + "n" not in written, "JSON-escaped newlines in the file"
    assert "import http" in written
    assert written.splitlines() == K6_SCRIPT.splitlines(), written[:80]
