"""Prompt oracle fixtures: what the task-package chain MUST keep sending.

Why a recorded oracle (and not just the live legacy comparison): the
legacy-vs-new comparison in ``test_migration_parity_pipeline_prompts.py`` is the
tool that PROVED these bytes, but it imports the legacy generators — so the
moment plan-k B6b.5/B7.3 deletes them, that file goes with them and the prompts
lose their only guard. These fixtures carry the guard forward: the recorded
pairs were measured off the legacy chain while it existed, and this test replays
the new chain against them with no legacy import at all.

Recording follows the standing discipline: a default run is read-only
(replay + diff), and rewriting a baseline needs ``TESTAGENT_RECORD_PARITY=1``.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import pytest

from tests.parity_harness import Fixture, ensure_fixture
from tests.prompt_capture import (
    CASE_SCENARIOS,
    REPO,
    TEXT_SCENARIOS,
    new_case_prompts,
    new_text_prompts,
)

TASK = "prompt_oracle"
#: Where each recorded set came from. The legacy chain is gone in the tree that
#: deletes it; this string is then the only statement of the oracle's origin.
ORIGIN = {
    "case": "legacy TestCaseGenerator.generate (verified byte-equal by "
    "tests/test_migration_parity_pipeline_prompts.py before recording)",
    "text": "legacy GUITestGenerator/PerformanceGenerator.generate (same proof, "
    "including the script-review pass)",
}


def _capture(tmp: Path, scenario: dict[str, Any]) -> list[tuple[str, str]]:
    kwargs = {k: v for k, v in scenario.items() if k != "name"}
    if "language" in kwargs and "module" in kwargs:
        return new_text_prompts(tmp, **kwargs)
    return new_case_prompts(tmp, **kwargs)


def _fixture(tmp: Path, scenario: dict[str, Any]) -> Fixture:
    prompts = _capture(tmp, scenario)
    kind = "text" if "module" in scenario else "case"
    return Fixture(
        name=scenario["name"],
        task=TASK,
        input={k: v for k, v in scenario.items() if k != "name"},
        request_trace=[{"index": i, "system": s, "user": u} for i, (s, u) in enumerate(prompts)],
        meta={
            "recorded_from": ORIGIN[kind],
            "chain": "tasks/" + str(scenario.get("module", "testcase")),
            "tasks_dir": "tasks",
        },
    )


@pytest.fixture(scope="module")
def baselines(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Fixture]:
    out: dict[str, Fixture] = {}
    for scenario in CASE_SCENARIOS + TEXT_SCENARIOS:
        tmp = tmp_path_factory.mktemp(f"oracle-{scenario['name']}")
        out[scenario["name"]] = ensure_fixture(
            TASK, scenario["name"], lambda tmp=tmp, scenario=scenario: _fixture(tmp, scenario)
        )
    return out


@pytest.mark.parametrize(
    "scenario",
    CASE_SCENARIOS + TEXT_SCENARIOS,
    ids=[s["name"] for s in CASE_SCENARIOS + TEXT_SCENARIOS],
)
def test_prompt_oracle_replays_without_drift(
    scenario: dict[str, Any], baselines: dict[str, Fixture], tmp_path: Path
) -> None:
    """Replay in a FRESH directory; any changed byte is a behavior change."""
    recorded = baselines[scenario["name"]]
    assert recorded.request_trace, f"{scenario['name']}: empty oracle"
    replayed = _capture(tmp_path / scenario["name"], scenario)
    assert len(replayed) == len(recorded.request_trace), (
        f"{scenario['name']}: call count {len(recorded.request_trace)} -> {len(replayed)}"
    )
    problems: list[str] = []
    for index, (system, user) in enumerate(replayed):
        want = recorded.request_trace[index]
        for field, got in (("system", system), ("user", user)):
            if want[field] != got:
                diff = [
                    line
                    for line in difflib.unified_diff(
                        want[field].splitlines(), got.splitlines(), lineterm="", n=0
                    )
                    if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
                ]
                problems.append(f"call {index} {field}: " + " || ".join(diff[:4]))
    assert not problems, f"{scenario['name']} drifted from the recorded oracle:\n" + "\n".join(
        problems
    )


def test_oracle_fixtures_exist_for_every_scenario() -> None:
    """A silently missing baseline would make the gate above vacuous."""
    for scenario in CASE_SCENARIOS + TEXT_SCENARIOS:
        assert (
            REPO / "tests" / "fixtures" / "migration" / TASK / f"{scenario['name']}.json"
        ).is_file(), f"missing prompt oracle fixture: {scenario['name']}"


def test_legacy_imports_are_absent_from_this_gate() -> None:
    """The whole point of the oracle: it must outlive the legacy chain.

    Checked as import statements (AST), not as text — a substring scan flags
    the very list of forbidden modules this test declares.
    """
    import ast

    forbidden = ("testagent.generators", "testagent.engine.prompt_builder", "engine.conversation")
    offenders: list[str] = []
    for name in ("test_prompt_oracle.py", "prompt_capture.py"):
        tree = ast.parse((REPO / "tests" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [str(alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            offenders += [f"{name}: {t}" for t in targets if t.startswith(forbidden)]
    assert not offenders, f"the oracle gate imports the legacy layer: {offenders}"
