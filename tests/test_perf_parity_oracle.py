"""Recorded perf parity: the task-package chain must keep sending these bytes.

``tests/test_perf_parity_legacy.py`` proved these cells against the legacy
``PerformanceGenerator`` and is the ONLY place allowed to record them; this file
replays them with no legacy import, so it still guards the prompts after plan-k
B5.4 deletes that generator. Recording stays opt-in (``TESTAGENT_RECORD_PARITY=1``)
in the recorder file — a default run must not rewrite the baseline it checks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from tests.parity_harness import Fixture, ensure_fixture
from tests.perf_parity_capture import CELLS, run_pipeline

TASK = "perf_parity"


def _never_build(name: str) -> Any:
    def _raise() -> Fixture:
        raise RuntimeError(
            f"perf parity cell {name!r} has no baseline, and this file cannot record one: "
            "run TESTAGENT_RECORD_PARITY=1 pytest tests/test_perf_parity_legacy.py while the "
            "legacy chain still exists (it is the oracle)."
        )

    return _raise


@pytest.mark.parametrize("cell", CELLS, ids=[cell["name"] for cell in CELLS])
def test_perf_parity_cell_matches_the_recorded_oracle(cell: dict[str, Any], tmp_path: Path) -> None:
    recorded = ensure_fixture(TASK, cell["name"], _never_build(cell["name"]))
    observed = asyncio.run(run_pipeline(tmp_path, cell))

    want_pairs = [[entry["system"], entry["user"]] for entry in recorded.request_trace]
    got_pairs = observed["pairs"]
    assert len(got_pairs) == len(want_pairs), (
        f"{cell['name']}: call count {len(want_pairs)} -> {len(got_pairs)}"
    )
    for index, (want, got) in enumerate(zip(want_pairs, got_pairs, strict=True)):
        for position, field in enumerate(("system", "user")):
            assert got[position] == want[position], (
                f"{cell['name']} call {index} {field} drifted:\n"
                f"  oracle  : {want[position][:260]!r}\n"
                f"  pipeline: {got[position][:260]!r}"
            )
    want_roles = [entry["role"] for entry in recorded.request_trace]
    assert observed["roles"] == want_roles, (
        f"{cell['name']}: client-role alternation changed ({want_roles} -> {observed['roles']})"
    )
    artifact = recorded.artifact if isinstance(recorded.artifact, dict) else {}
    assert observed["artifact"] == artifact.get("artifact"), (
        f"{cell['name']}: artifact drifted:\n  oracle  : {artifact.get('artifact')!r}\n"
        f"  pipeline: {observed['artifact']!r}"
    )
    assert observed["review_status"] == artifact.get("review_status"), (
        f"{cell['name']}: review status {artifact.get('review_status')!r} -> "
        f"{observed['review_status']!r}"
    )


def test_every_cell_has_a_baseline() -> None:
    """A silently missing cell turns the gate above into a no-op."""
    for cell in CELLS:
        path = Path(__file__).parent / "fixtures" / "migration" / TASK / f"{cell['name']}.json"
        assert path.is_file(), f"missing perf parity baseline: {cell['name']}"
