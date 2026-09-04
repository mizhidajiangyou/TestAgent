"""Review runtime tests (plan-d B5.0 / v3 R1+R2+R6 / plan-e E5).

Covers: the ReviewHooks protocol's four responsibilities, the three review
terminal states (REVIEWED / REVIEW_REJECTED / REVIEW_FAILED — FAILED ≠
REJECTED), retention trigger/exemption, the multi-round lifecycle
(max_rounds=1 vs >1, round-1-changed → round-2), model alternation, and
the executor-level e2e: a review explosion falls back to the pre-review
snapshot content.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from testagent.engine.llm_client import LLMResponse
from testagent.pipeline.executor import SNAPSHOT_SUFFIX, PipelineExecutor
from testagent.pipeline.inputs import TaskContext, parse_inputs
from testagent.pipeline.review_hooks import (
    make_list_hooks,
    make_text_hooks,
)
from testagent.pipeline.runtime import (
    REVIEW_DISABLED,
    REVIEW_FAILED,
    REVIEW_REJECTED,
    REVIEWED,
    build_review_runner,
)
from testagent.pipeline.status import UnitResult, UnitStatus

REPO = Path(__file__).parents[1]
EXAMPLE_MANIFEST = json.loads(
    (REPO / "tasks" / "_example" / "manifest.json").read_text(encoding="utf-8")
)


class ReviewFakeLLM:
    """LLM double for the review loop (primary + secondary models).

    ``script`` items are returned in order per achat call; exceptions raise.
    ``review_script`` drives the SECONDARY client (odd rounds) so tests can
    assert alternation.
    """

    intent_capable = True

    def __init__(
        self,
        script: list[Any],
        *,
        review_script: list[Any] | None = None,
        model: str = "primary-m",
    ) -> None:
        self._script = list(script)
        self._review_script = list(review_script) if review_script else list(script)
        self.model_name = model
        self.calls: list[str] = []  # "primary"|"review" per request
        self.all_prompts: list[str] = []  # user prompts in request order
        self.primary = _SubClient(self._script, "primary", self.calls, self.all_prompts)
        self._secondary = _SubClient(self._review_script, "review", self.calls, self.all_prompts)
        self.session_id = ""

    def secondary_client(self) -> _SubClient:
        return self._secondary

    def set_session_id(self, sid: str) -> None:
        self.session_id = sid

    async def averify(self) -> None:
        return None

    async def achat_with_meta(self, system: str, user: str, **kw: Any) -> LLMResponse:
        # Delegates to the primary sub-client (which does the call counting).
        return await self.primary.achat_with_meta(system, user, **kw)


class _SubClient:
    def __init__(
        self, script: list[Any], role: str, calls: list[str], all_prompts: list[str]
    ) -> None:
        self._script = script
        self._role = role
        self._calls = calls
        self._all_prompts = all_prompts
        self.model_name = f"{role}-m"
        self.seen_prompts: list[tuple[str, str]] = []

    async def achat_with_meta(self, system: str, user: str, **kw: Any) -> LLMResponse:
        self._calls.append(self._role)
        self._all_prompts.append(user)
        self.seen_prompts.append((system, user))
        idx = sum(1 for c in self._calls if c == self._role) - 1
        item = self._script[min(idx, len(self._script) - 1)]
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(text=str(item), finish_reason="stop")


class _Settings:
    class LLM:
        max_concurrency = 2
        json_mode = False

    llm = LLM
    output_dir = ""
    output_language = "english"
    review_enabled = True
    review_max_rounds = 2


def _task(
    tmp_path: Path, *, review_enabled: Any = "from_settings", max_rounds: Any = "from_settings"
) -> Any:
    """A minimal task package with a review template (real TaskPackage)."""
    from testagent.pipeline.manifest import Manifest
    from testagent.pipeline.registry import TaskPackage

    base = dict(EXAMPLE_MANIFEST)
    base["review"] = {
        "enabled": review_enabled,
        "max_rounds": max_rounds,
        "template": "prompts/review.j2",
        "system_prompt": "inline:review-sys",
    }
    manifest = Manifest.model_validate(base)
    root = tmp_path / manifest.name
    (root / "prompts").mkdir(parents=True, exist_ok=True)
    (root / "prompts" / "echo.j2").write_text('Return [{"text": "x"}]', encoding="utf-8")
    (root / "prompts" / "review.j2").write_text(
        "round {{ round }} artifact {{ artifact }}", encoding="utf-8"
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return TaskPackage(manifest.name, manifest, root)


def _ctx(task: Any) -> TaskContext:
    return parse_inputs(task.manifest, {"text": "hello"})


ARTIFACT = [
    {"id": "TC-001", "title": "a"},
    {"id": "TC-002", "title": "b"},
    {"id": "TC-003", "title": "c"},
    {"id": "TC-004", "title": "d"},
]


# ----------------------------------------------------------------------
# R6 protocol: four responsibilities
# ----------------------------------------------------------------------


class TestReviewHooksProtocol:
    def test_build_prompt_receives_current_and_round(self) -> None:
        seen: list[tuple[str, int]] = []

        def render(serialized: str, round_idx: int) -> str:
            seen.append((serialized, round_idx))
            return f"prompt-{round_idx}"

        hooks = make_list_hooks(render)
        assert hooks.build_prompt(ARTIFACT, 2) == "prompt-2"
        assert seen[0][1] == 2
        assert json.loads(seen[0][0]) == ARTIFACT  # list serialized as JSON

    def test_parse_none_marks_unparseable(self) -> None:
        hooks = make_list_hooks(lambda s, r: s)
        assert hooks.parse("garbage not json") is None
        assert hooks.parse(json.dumps([{"a": 1}])) == [{"a": 1}]

    def test_apply_is_identity_and_the_only_merge_point(self) -> None:
        # E5: review output IS the full new artifact for both factories.
        for hooks in (make_list_hooks(lambda s, r: s), make_text_hooks(lambda s, r: s)):
            reviewed: Any = [{"new": 1}] if isinstance(hooks.parse("[]"), list) else "script"
            assert hooks.apply(ARTIFACT, reviewed) is reviewed

    def test_retention_check_list_vs_text(self) -> None:
        list_hooks = make_list_hooks(lambda s, r: s)
        # 1 of 4 kept = 25% < 50% → rejected.
        assert list_hooks.retention_check(ARTIFACT, ARTIFACT[:1]) is False
        # 3 of 4 kept = 75% → accepted.
        assert list_hooks.retention_check(ARTIFACT, ARTIFACT[:3]) is True
        # Empty original never rejects (first-round semantics).
        assert list_hooks.retention_check([], ARTIFACT[:1]) is True
        text_hooks = make_text_hooks(lambda s, r: s)
        # Scripts are exempt: any shrink accepted.
        assert text_hooks.retention_check("a" * 100, "x") is True


# ----------------------------------------------------------------------
# R2 three terminal states
# ----------------------------------------------------------------------


class TestReviewStates:
    async def test_reviewed(self, tmp_path: Path) -> None:
        refined = [{"id": f"TC-{i:03d}", "title": "r"} for i in range(1, 5)]
        llm = ReviewFakeLLM([json.dumps(refined)])
        task = _task(tmp_path, max_rounds=1)
        outcome = await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        assert outcome.status == REVIEWED
        assert outcome.artifact == refined
        assert outcome.meta["rounds_succeeded"] == 1

    async def test_review_rejected_keeps_original(self, tmp_path: Path) -> None:
        # Parseable but shrunk below the 50% retention guard.
        llm = ReviewFakeLLM([json.dumps([{"id": "TC-001", "title": "only-one"}])])
        task = _task(tmp_path)
        outcome = await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        assert outcome.status == REVIEW_REJECTED
        assert outcome.artifact == ARTIFACT  # original, NOT the snapshot
        assert outcome.meta["rejected_reason"] == "retention_guard"

    async def test_review_failed_unparseable_falls_back_to_snapshot(self, tmp_path: Path) -> None:
        llm = ReviewFakeLLM(["not json at all", "still not json"])
        task = _task(tmp_path)
        outcome = await build_review_runner(llm)(
            task, ARTIFACT, "SNAPSHOT", _ctx(task), _Settings()
        )
        assert outcome.status == REVIEW_FAILED
        assert outcome.artifact == "SNAPSHOT"  # R2: snapshot content
        assert outcome.meta["failure_reason"] == "unparseable_review_output"

    async def test_review_failed_exception_falls_back_to_snapshot(self, tmp_path: Path) -> None:
        llm = ReviewFakeLLM([RuntimeError("review LLM exploded")])
        task = _task(tmp_path)
        outcome = await build_review_runner(llm)(
            task, ARTIFACT, "SNAPSHOT", _ctx(task), _Settings()
        )
        assert outcome.status == REVIEW_FAILED
        assert outcome.artifact == "SNAPSHOT"
        assert "exploded" in outcome.meta["failure_reason"]

    async def test_review_disabled(self, tmp_path: Path) -> None:
        llm = ReviewFakeLLM([])
        task = _task(tmp_path, review_enabled=False)
        outcome = await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        assert outcome.status == REVIEW_DISABLED
        assert outcome.artifact == ARTIFACT
        assert llm.calls == []

    async def test_text_artifact_shrink_is_exempt(self, tmp_path: Path) -> None:
        # A much shorter reviewed script is still REVIEWED (text exempt).
        llm = ReviewFakeLLM(["print('tight')"])
        task = _task(tmp_path)
        outcome = await build_review_runner(llm)(
            task, "print('a')\n" * 50, "SNAP", _ctx(task), _Settings()
        )
        assert outcome.status == REVIEWED
        assert outcome.artifact == "print('tight')"


# ----------------------------------------------------------------------
# Multi-round lifecycle + alternation (P1-3)
# ----------------------------------------------------------------------


class TestReviewLifecycle:
    async def test_round1_changed_round2_runs_on_refined(self, tmp_path: Path) -> None:
        r1 = [{"id": f"TC-{i:03d}", "title": "v1"} for i in range(1, 5)]
        r2 = [{"id": f"TC-{i:03d}", "title": "v2"} for i in range(1, 5)]
        # Round 1 → secondary (review_script), round 2 → primary (script);
        # each sub-client has its own script index.
        llm = ReviewFakeLLM([json.dumps(r2)], review_script=[json.dumps(r1)])
        task = _task(tmp_path)
        outcome = await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        assert outcome.status == REVIEWED
        assert outcome.artifact == r2
        assert outcome.meta["rounds_executed"] == 2
        assert outcome.meta["rounds_succeeded"] == 2
        # Round 2's prompt carried the round-1 artifact (current), not the
        # original — the loop feeds the refined artifact forward. Round 1
        # goes to the secondary client (alternation), round 2 to primary.
        prompts = llm.all_prompts
        assert len(prompts) == 2
        assert "round 1" in prompts[0] and "round 2" in prompts[1]
        assert "v1" in prompts[1] and "v1" not in prompts[0]

    async def test_max_rounds_one(self, tmp_path: Path) -> None:
        refined = [{"id": f"TC-{i:03d}", "title": "r"} for i in range(1, 5)]
        llm = ReviewFakeLLM([json.dumps(refined), json.dumps(refined)])
        task = _task(tmp_path, max_rounds=1)
        outcome = await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        assert outcome.meta["rounds_executed"] == 1
        assert llm.calls.count("primary") + llm.calls.count("review") == 1

    async def test_model_alternation_odd_review_even_primary(self, tmp_path: Path) -> None:
        r1 = [{"id": f"TC-{i:03d}", "title": "v1"} for i in range(1, 5)]
        r2 = [{"id": f"TC-{i:03d}", "title": "v2"} for i in range(1, 5)]
        llm = ReviewFakeLLM(
            [json.dumps(r1), json.dumps(r2)],
            review_script=[json.dumps(r1), json.dumps(r2)],
        )
        task = _task(tmp_path)
        await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        # Legacy semantics: odd rounds → review (secondary), even → primary.
        assert llm.calls == ["review", "primary"]

    async def test_declared_system_prompt_used(self, tmp_path: Path) -> None:
        refined = [{"id": "TC-001", "title": "r"}]
        llm = ReviewFakeLLM([json.dumps(refined)])
        task = _task(tmp_path)
        await build_review_runner(llm)(task, ARTIFACT, "SNAP", _ctx(task), _Settings())
        system, _ = llm.primary.seen_prompts[0]
        assert system == "review-sys"  # from review.system_prompt, not default


# ----------------------------------------------------------------------
# Executor e2e: review explosion falls back to the snapshot (B5.0 DoD)
# ----------------------------------------------------------------------


async def _make_gen(items: list[dict[str, Any]]) -> Any:
    async def gen(task, stage, label, unit_ctx, ctx, sid, fp):
        return UnitResult(status=UnitStatus.SUCCESS, items=items)

    return gen


class TestExecutorReviewE2E:
    async def test_review_exception_artifact_equals_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _Settings.output_dir = str(tmp_path)
        task = _task(tmp_path)
        gen_llm = ReviewFakeLLM([])
        review_llm = ReviewFakeLLM([RuntimeError("boom")])
        executor = PipelineExecutor(
            gen_llm,
            _Settings,
            generate_unit=await _make_gen([{"text": "golden"}]),
            review_runner=build_review_runner(review_llm),
        )
        result = await executor.arun(task, _ctx(task), session_id="revfail")
        # R2: REVIEW_FAILED → artifact = snapshot content (with renumber id).
        assert result.review_meta is not None
        assert result.review_meta["status"] == REVIEW_FAILED
        snap = json.loads((tmp_path / f"revfail{SNAPSHOT_SUFFIX}").read_text(encoding="utf-8"))
        assert result.artifact == snap["artifact"]

    async def test_review_rejected_artifact_equals_original(self, tmp_path: Path) -> None:
        _Settings.output_dir = str(tmp_path)
        task = _task(tmp_path)
        gen_llm = ReviewFakeLLM([])
        review_llm = ReviewFakeLLM(['[{"id": "TC-001", "title": "one"}]'])
        executor = PipelineExecutor(
            gen_llm,
            _Settings,
            generate_unit=await _make_gen(ARTIFACT),
            review_runner=build_review_runner(review_llm),
        )
        result = await executor.arun(task, _ctx(task), session_id="revrej")
        assert result.review_meta is not None
        assert result.review_meta["status"] == REVIEW_REJECTED
        # Original (pre-review merged artifact with renumber) kept.
        assert [c["id"] for c in result.artifact] == [
            f"TC-{i:03d}" for i in range(1, len(ARTIFACT) + 1)
        ]

    async def test_no_runner_review_meta_none(self, tmp_path: Path) -> None:
        _Settings.output_dir = str(tmp_path)
        task = _task(tmp_path)
        executor = PipelineExecutor(
            ReviewFakeLLM([]), _Settings, generate_unit=await _make_gen([{"text": "x"}])
        )
        result = await executor.arun(task, _ctx(task), session_id="norev")
        assert result.review_meta is None
