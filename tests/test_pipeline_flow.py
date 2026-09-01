"""Registry conflict detection, input parsing, splitting, merging, writers,
executor behaviour + snapshots (plan-c B3.2/B3.5/B4.3/B4.5/B4.6)."""

import json
from pathlib import Path
from typing import Any

import pytest

from testagent.pipeline.executor import (
    SNAPSHOT_SUFFIX,
    PipelineExecutor,
    list_snapshots,
    recover_snapshot,
)
from testagent.pipeline.inputs import INPUT_PARSERS, TaskContext, parse_inputs
from testagent.pipeline.manifest import (
    Manifest,
    MergeSpec,
    SplitSpec,
    StageSpec,
    WhenSpec,
)
from testagent.pipeline.merge import StageResult, apply_merge
from testagent.pipeline.registry import Registry, TaskPackage
from testagent.pipeline.split import evaluate_when, make_units
from testagent.pipeline.status import UnitResult, UnitStatus
from testagent.pipeline.writers import resolve_extension, write_artifact

EXAMPLE_MANIFEST = json.loads(
    (Path(__file__).parents[1] / "tasks" / "_example" / "manifest.json").read_text(encoding="utf-8")
)


def _manifest(**overrides: Any) -> Manifest:
    base = dict(EXAMPLE_MANIFEST)
    base.update(overrides)
    return Manifest.model_validate(base)


def _pkg(tmp_path: Path, manifest: Manifest | None = None) -> TaskPackage:
    m = manifest or _manifest()
    root = tmp_path / m.name
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "echo.j2").write_text('Return [{"text": "{{ text }}"}]', encoding="utf-8")
    (root / "manifest.json").write_text(m.model_dump_json(), encoding="utf-8")
    return TaskPackage(m.name, m, root)


# ----------------------------------------------------------------------
# Registry conflicts (B3.5)
# ----------------------------------------------------------------------


class TestRegistryConflicts:
    def test_all_collision_classes_reported(self, tmp_path: Path) -> None:
        a = _pkg(tmp_path, _manifest(name="alpha", aliases=["shared"]))
        b = _pkg(tmp_path, _manifest(name="beta", aliases=["shared"]))
        c = _pkg(tmp_path, _manifest(name="gamma", aliases=["alpha"]))
        reg = Registry([a, b, c])
        kinds = " ".join(reg.conflicts)
        assert "alias/alias" in kinds  # 'shared' declared by alpha and beta
        assert "alias/name" in kinds  # gamma's alias 'alpha' == a task name

    def test_name_name_conflict(self, tmp_path: Path) -> None:
        a = _pkg(tmp_path, _manifest(name="dup"))
        b = _pkg(tmp_path, _manifest(name="dup"))
        reg = Registry([a, b])
        assert any("name/name" in c for c in reg.conflicts)

    def test_clean_registry_has_no_conflicts(self, tmp_path: Path) -> None:
        reg = Registry([_pkg(tmp_path, _manifest(name="alpha", aliases=["al"]))])
        assert reg.conflicts == []
        assert "al" in reg
        assert reg.get("al").name == "alpha"

    def test_hidden_tasks_excluded_by_default(self, tmp_path: Path) -> None:
        reg = Registry([_pkg(tmp_path, _manifest(name="_hidden"))])
        assert reg.tasks() == []
        assert len(reg.tasks(include_hidden=True)) == 1


# ----------------------------------------------------------------------
# Inputs (B3.2/B4.5a)
# ----------------------------------------------------------------------

_INPUT_MANIFEST = _manifest(require_any=[], inputs=[])


class TestInputs:
    def test_parser_registry_covers_all_kinds(self) -> None:
        for kind in ("swagger", "requirements", "file", "text", "choice", "int", "bool"):
            assert kind in INPUT_PARSERS, kind

    def test_text_input_parsed(self) -> None:
        ctx = parse_inputs(_manifest(), {"text": "hello"})
        assert ctx.parsed["text"] == "hello"

    def test_int_and_bool_coerced(self) -> None:
        m = _manifest(
            require_any=[],
            inputs=[
                {"name": "n", "kind": "int"},
                {"name": "b", "kind": "bool"},
            ],
        )
        ctx = parse_inputs(m, {"n": "42", "b": "true"})
        assert ctx.parsed["n"] == 42 and ctx.parsed["b"] is True

    def test_require_any_enforced(self) -> None:
        m = _manifest(require_any=["text"])
        with pytest.raises(ValueError, match="requires at least one"):
            parse_inputs(m, {})

    def test_from_settings_default(self) -> None:
        m = _manifest(
            require_any=[],
            inputs=[{"name": "lang", "kind": "text", "default": "from_settings:output_language"}],
        )

        class _S:
            class LLM:
                max_concurrency = 7
                json_mode = False

            llm = LLM
            output_language = "english"

        ctx = parse_inputs(m, {}, _S)
        assert ctx.parsed["lang"] == "english"
        assert ctx.settings_views["max_concurrency"] == 7

    def test_plain_default_used_without_settings(self) -> None:
        m = _manifest(require_any=[], inputs=[{"name": "t", "kind": "text", "default": "fallback"}])
        ctx = parse_inputs(m, {})
        assert ctx.parsed["t"] == "fallback"


# ----------------------------------------------------------------------
# Split (B4.5b)
# ----------------------------------------------------------------------

_STAGE = StageSpec(name="s", template="t.j2", system_prompt="inline:x")


class TestSplit:
    def test_single(self) -> None:
        ctx = TaskContext(parsed={"a": 1})
        units = make_units(_STAGE, ctx)
        assert len(units) == 1 and units[0][1]["a"] == 1

    def test_per_input(self) -> None:
        stage = StageSpec(
            name="s",
            template="t.j2",
            system_prompt="inline:x",
            split=SplitSpec(by="per_input", input="requirements"),
        )
        ctx = TaskContext(parsed={"requirements": ["r1", "r2"]})
        units = make_units(stage, ctx)
        assert [u[0] for u in units] == ["s 1/2", "s 2/2"]
        assert units[0][1]["_unit_item"] == "r1"

    def test_batch(self) -> None:
        stage = StageSpec(
            name="s",
            template="t.j2",
            system_prompt="inline:x",
            split=SplitSpec(by="batch", input="endpoints", batch_size=2),
        )
        ctx = TaskContext(parsed={"endpoints": ["e1", "e2", "e3"]})
        units = make_units(stage, ctx)
        assert len(units) == 2 and units[1][1]["_unit_batch"] == ["e3"]

    def test_evaluate_when(self) -> None:
        ctx = TaskContext(raw={"swagger": "x"}, parsed={})
        assert evaluate_when(WhenSpec(input_present=["swagger"]), ctx)
        assert not evaluate_when(WhenSpec(input_absent=["swagger"]), ctx)


# ----------------------------------------------------------------------
# Merge (B4.5d)
# ----------------------------------------------------------------------


class TestMerge:
    def test_baseline_first_then_dedup_then_renumber(self) -> None:
        spec = MergeSpec(
            baseline_input="history",
            dedup={"keys": ["title"], "normalize": "lower"},  # type: ignore[arg-type]
        )
        ctx = TaskContext(parsed={"history": [{"id": "X", "title": "Login"}]})
        stages = [
            StageResult(
                name="s1",
                items=[
                    {"id": "?", "title": "login"},  # duplicate (normalized)
                    {"id": "?", "title": "Logout"},
                ],
            )
        ]
        merged = apply_merge(spec, stages, ctx)
        assert [m["title"] for m in merged] == ["Login", "Logout"]
        assert [m["id"] for m in merged] == ["TC-001", "TC-002"]

    def test_empty_results_merge_to_empty(self) -> None:
        ctx = TaskContext(parsed={})
        assert apply_merge(MergeSpec(), [], ctx) == []


# ----------------------------------------------------------------------
# Writers (B4.5e)
# ----------------------------------------------------------------------


class TestWriters:
    def test_resolve_extension_fixed(self) -> None:
        assert resolve_extension({"extension": ".py"}, {}) == ".py"

    def test_resolve_extension_by_format(self) -> None:
        spec = {"extension": {"format": {"k6": ".js", "jmeter": ".jmx"}}}
        assert resolve_extension(spec, {"format": "jmeter"}) == ".jmx"
        assert resolve_extension(spec, {"format": "k6"}) == ".js"

    def test_resolve_extension_by_input(self) -> None:
        spec = {"extension": {"by_input": "kind", "map": {"gui": ".py"}}}
        assert resolve_extension(spec, {"kind": "gui"}) == ".py"

    def test_write_json_roundtrip(self, tmp_path: Path) -> None:
        m = _manifest()
        path = write_artifact(m, [{"a": 1}], tmp_path / "out.json", "json")
        assert json.loads(path.read_text(encoding="utf-8")) == [{"a": 1}]

    def test_write_text_with_extension_and_meta(self, tmp_path: Path) -> None:
        m = _manifest(output={"default_path": "", "formats": {"text": {"extension": ".py"}}})
        path = write_artifact(
            m,
            "print('hi')",
            tmp_path / "script",
            "text",
            review_meta={"rounds": 1},
        )
        assert path.suffix == ".py"
        assert path.read_text(encoding="utf-8") == "print('hi')"
        meta = json.loads(path.with_suffix(".py.meta.json").read_text(encoding="utf-8"))
        assert meta == {"rounds": 1}

    def test_write_csv(self, tmp_path: Path) -> None:
        m = _manifest()
        items = [{"id": "TC-001", "title": "t", "steps": ["a", "b"]}]
        path = write_artifact(m, items, tmp_path / "out.csv", "csv")
        content = path.read_text(encoding="utf-8-sig")
        assert "TC-001" in content and "a; b" in content


# ----------------------------------------------------------------------
# Executor + snapshots (B4.6)
# ----------------------------------------------------------------------


class _FakeLLM:
    """LLM double with the metadata the executor touches."""

    intent_capable = True

    def __init__(self, script: list[str]) -> None:
        self._script = list(script)
        self.calls = 0
        self.session_id = ""

    def set_session_id(self, sid: str) -> None:
        self.session_id = sid

    async def averify(self) -> None:
        return None

    async def achat_with_meta(self, system: str, user: str, **kw: Any) -> Any:
        from testagent.engine.llm_client import LLMResponse

        self.calls += 1
        text = self._script[min(self.calls - 1, len(self._script) - 1)]
        return LLMResponse(text=text, finish_reason="stop")


class _Settings:
    class LLM:
        max_concurrency = 2

    llm = LLM
    output_dir = ""


@pytest.fixture()
def settings_tmp(tmp_path: Path) -> Any:
    _Settings.output_dir = str(tmp_path)
    return _Settings


def _make_gen(llm: _FakeLLM, outcomes: list[UnitResult] | None = None) -> Any:
    """generate_unit closure: script outcomes OR LLM-backed parse."""
    from testagent.pipeline.runtime import _extract_json_list

    async def gen(task, stage, label, unit_ctx, ctx, sid, fp):
        if outcomes is not None:
            idx = min(gen_calls["n"], len(outcomes) - 1)  # noqa: F821
            return outcomes[idx]
        system = task.system_prompt(stage.system_prompt)
        user = task.render(stage.template, {**ctx.parsed, **unit_ctx})
        resp = await llm.achat_with_meta(system, user)
        items = _extract_json_list(resp.text)
        return UnitResult(
            status=UnitStatus.SUCCESS if items else UnitStatus.EMPTY, items=items or []
        )

    return gen


class TestExecutor:
    async def test_golden_path_snapshot_and_result(self, tmp_path: Path, settings_tmp: Any) -> None:
        """E2E golden path (executor slice): render -> generate -> merge ->
        snapshot; the snapshot equals the artifact and carries the session
        id. The default renumber spec stamps TC-001 on the single item."""
        pkg = _pkg(tmp_path)
        llm = _FakeLLM(['[{"text": "hello"}]'])
        executor = PipelineExecutor(llm, settings_tmp, generate_unit=_make_gen(llm))
        ctx = parse_inputs(pkg.manifest, {"text": "hello"})
        result = await executor.arun(pkg, ctx, session_id="sess123")

        assert result.session_id == "sess123"
        assert result.artifact == [{"text": "hello", "id": "TC-001"}]
        snap_path = tmp_path / f"sess123{SNAPSHOT_SUFFIX}"
        assert snap_path.exists()
        payload = json.loads(snap_path.read_text(encoding="utf-8"))
        assert payload["artifact"] == result.artifact
        assert payload["stage"] == "pre_review"
        assert recover_snapshot("sess123", str(tmp_path))["count"] == 1
        assert list_snapshots(str(tmp_path))[0]["session_id"] == "sess123"

    async def test_engine_recovered_empty_not_reasked(
        self, tmp_path: Path, settings_tmp: Any
    ) -> None:
        """Decision D2: EMPTY with engine_recovered is NOT re-asked."""
        pkg = _pkg(tmp_path)
        llm = _FakeLLM([])
        calls = {"n": 0}

        async def gen(task, stage, label, unit_ctx, ctx, sid, fp):
            calls["n"] += 1
            return UnitResult(status=UnitStatus.EMPTY, engine_recovered=True)

        executor = PipelineExecutor(llm, settings_tmp, generate_unit=gen)
        ctx = parse_inputs(pkg.manifest, {"text": "hello"})
        result = await executor.arun(pkg, ctx)
        assert calls["n"] == 1  # no recovery re-ask
        assert result.units_failed == 1
        assert result.artifact == []

    async def test_mixed_fanout_recovers_only_empty(
        self, tmp_path: Path, settings_tmp: Any
    ) -> None:
        """Fan-out semantics (legacy _fan_out_recover + D2): with at least
        one SUCCESS sibling, the plain-EMPTY unit is re-asked serially; the
        engine-recovered one is not."""
        pkg = _pkg(
            tmp_path,
            _manifest(
                pipeline={
                    "stages": [
                        {
                            "name": "s",
                            "template": "prompts/echo.j2",
                            "system_prompt": "inline:x",
                            "split": {"by": "per_input", "input": "requirements"},
                        }
                    ]
                },
                inputs=[{"name": "text", "kind": "text"}, {"name": "requirements", "kind": "text"}],
            ),
        )
        llm = _FakeLLM([])
        calls = {"n": 0}
        per_unit: dict[int, UnitResult] = {}

        async def gen(task, stage, label, unit_ctx, ctx, sid, fp):
            calls["n"] += 1
            # First pass: unit 1 OK, unit 2 plain-EMPTY, unit 3 recovered-EMPTY.
            # Recovery pass: unit 2 succeeds.
            if calls["n"] <= 3:
                per_unit[calls["n"]] = UnitResult(
                    status=UnitStatus.EMPTY,
                    engine_recovered=(calls["n"] == 3),
                )
                if calls["n"] == 1:
                    per_unit[1] = UnitResult(
                        status=UnitStatus.SUCCESS, items=[{"text": f"u{calls['n']}"}]
                    )
                return per_unit[calls["n"]]
            return UnitResult(status=UnitStatus.SUCCESS, items=[{"text": "recovered"}])

        executor = PipelineExecutor(llm, settings_tmp, generate_unit=gen)
        ctx = parse_inputs(pkg.manifest, {"text": "t", "requirements": "r1\nr2\nr3"})
        # per_input split needs a list in parsed; the text parser yields a str.
        ctx.parsed["requirements"] = ["r1", "r2", "r3"]
        result = await executor.arun(pkg, ctx)
        assert calls["n"] == 4  # 3 first-pass + 1 recovery (unit 2 only)
        assert result.units_failed == 1  # the engine-recovered unit
        texts = [it["text"] for it in result.artifact]
        assert "recovered" in texts

    async def test_all_empty_no_reask(self, tmp_path: Path, settings_tmp: Any) -> None:
        """All units empty (no sibling success) -> model is down -> no retry."""
        pkg = _pkg(tmp_path)
        llm = _FakeLLM([])
        calls = {"n": 0}

        async def gen(task, stage, label, unit_ctx, ctx, sid, fp):
            calls["n"] += 1
            return UnitResult(status=UnitStatus.EMPTY)

        executor = PipelineExecutor(llm, settings_tmp, generate_unit=gen)
        ctx = parse_inputs(pkg.manifest, {"text": "hello"})
        await executor.arun(pkg, ctx)
        assert calls["n"] == 1

    async def test_snapshot_pruning(
        self, tmp_path: Path, settings_tmp: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CHECKPOINT_KEEP_LAST", "3")
        for i in range(5):
            (tmp_path / f"old{i}{SNAPSHOT_SUFFIX}").write_text("{}", encoding="utf-8")
        pkg = _pkg(tmp_path)
        llm = _FakeLLM(['[{"text": "x"}]'])
        executor = PipelineExecutor(llm, settings_tmp, generate_unit=_make_gen(llm))
        ctx = parse_inputs(pkg.manifest, {"text": "hello"})
        await executor.arun(pkg, ctx)
        remaining = list(tmp_path.glob(f"*{SNAPSHOT_SUFFIX}"))
        assert len(remaining) <= 4  # 3 kept + the fresh one from this run
