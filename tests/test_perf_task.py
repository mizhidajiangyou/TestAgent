"""B5.1 tasks/perf — package discipline + run-through tests (plan-d B5.1 gate).

The fingerprint-parity half moved out, because it needs the legacy
``PerformanceGenerator`` as its oracle: ``test_perf_parity_legacy.py`` holds the
live comparison (and is the only place allowed to record cells), and
``test_perf_parity_oracle.py`` replays the recorded cells with no legacy import
so the guard survives plan-k B5.4. Shared doubles and the cells live in
``tests/perf_parity_capture.py``.

What stays here only ever talks to the task package: strict validation, the
reverse-rename template discipline (pinned against the FROZEN legacy template
bytes, recorded at commit 4f73474, so it outlives the templates too), the
fingerprint recording machinery, and the fake-LLM run-through.

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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

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
from tests.perf_parity_capture import (
    JMX_REVISED,
    JMX_SCRIPT,
    K6_REVISED,
    K6_SCRIPT,
    SWAGGER,
    TASK_ROOT,
    ParityFakeLLM,
    Settings,
)
from tests.perf_parity_capture import (
    sha12 as _sha,
)


def _settings(
    tmp_path: Path,
    *,
    review_enabled: bool = False,
    output_language: str = "english",
    script_format: str = "k6",
) -> Any:
    return Settings(
        str(tmp_path),
        review_enabled=review_enabled,
        output_language=output_language,
        script_format=script_format,
    )


def _task() -> TaskPackage:
    manifest = load_manifest(TASK_ROOT / "manifest.json")
    return TaskPackage(manifest.name, manifest, TASK_ROOT)


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


class TestFingerprintRecording:
    """The recording machinery itself (label, model, pair hashes).

    The request-boundary parity it used to sit next to now lives in
    ``test_perf_parity_legacy.py`` / ``test_perf_parity_oracle.py``.
    """

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
