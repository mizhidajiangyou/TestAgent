"""Live legacy-vs-pipeline perf parity, and the ONLY recorder of the cells.

Two jobs that have to live together: the comparison is what proves the recorded
baseline says "what the legacy chain sent", and the recorder must run while that
chain still exists. Delete this file together with ``PerformanceGenerator``
(plan-k B5.4); the guard that survives is ``test_perf_parity_oracle.py``.

Cell semantics come from plan-d B5.1 (I2 whitelist): the ordered
``(system, user)`` request pairs and the final artifact must be identical, and
so must which client role made each call (generation = primary, odd review
rounds = secondary — the alternation contract).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from testagent.config.models import PerfGenInput, PerformanceConfig
from testagent.engine.prompt_builder import PromptBuilder
from testagent.generators.performance_generator import PerformanceGenerator
from testagent.parsers.swagger_parser import SwaggerParser
from tests.parity_harness import Fixture, ensure_fixture
from tests.perf_parity_capture import CELLS, PERF_KW, SWAGGER, ParityFakeLLM, run_pipeline

TASK = "perf_parity"
_ENDPOINTS = SwaggerParser().parse(str(SWAGGER))


def _run_legacy(cell: dict[str, Any]) -> tuple[list[list[str]], list[str], Any, Any]:
    """Drive the legacy generator for one cell; returns the same observables."""
    fake = ParityFakeLLM(cell["primary_script"], cell["secondary_script"])
    overrides = cell.get("overrides")
    generator = PerformanceGenerator(
        llm_client=fake,  # type: ignore[arg-type]
        prompt_builder=PromptBuilder(),
        review_enabled=cell["review_enabled"],
        review_llm_client=fake.secondary_client() if overrides is None else None,
        review_max_rounds=2,
        output_language=cell["output_language"],
    )
    generator.set_script_format(cell["script_format"])
    config = PerformanceConfig(**(overrides or _default_config()))
    artifact = generator.generate(PerfGenInput(endpoints=_ENDPOINTS, config=config))
    # ReviewResult carries used_review/rounds_*, not a status string; the
    # pipeline's meta status is the same fact spelled differently (REVIEWED ==
    # at least one round succeeded). Record the equivalent, not a guess.
    last = generator._last_review
    status = None if last is None else ("REVIEWED" if last.used_review else "REVIEW_FAILED")
    return ([[system, user] for system, user in fake.pairs], list(fake.roles), artifact, status)


def _default_config() -> dict[str, Any]:
    """Same values the settings double feeds the pipeline: a parity cell that
    lets the two sides resolve defaults differently proves nothing."""
    return dict(PERF_KW)


def _fixture(tmp_path: Path, cell: dict[str, Any]) -> Fixture:
    pairs, roles, artifact, review_status = _run_legacy(cell)
    return Fixture(
        name=cell["name"],
        task=TASK,
        input={
            "script_format": cell["script_format"],
            "review_enabled": cell["review_enabled"],
            "output_language": cell["output_language"],
            "overrides": bool(cell.get("overrides")),
            "omit_format": bool(cell.get("omit_format")),
        },
        request_trace=[
            {"index": index, "role": role, "system": system, "user": user}
            for index, ((system, user), role) in enumerate(zip(pairs, roles, strict=True))
        ],
        artifact={"artifact": artifact, "review_status": review_status},
        meta={"recorded_from": "legacy PerformanceGenerator.generate (live comparison below)"},
    )


@pytest.mark.parametrize("cell", CELLS, ids=[cell["name"] for cell in CELLS])
async def test_pipeline_matches_legacy_cell(cell: dict[str, Any], tmp_path: Path) -> None:
    legacy_pairs, legacy_roles, legacy_artifact, _ = _run_legacy(cell)
    observed = await run_pipeline(tmp_path, cell)
    assert observed["pairs"] == legacy_pairs, (
        f"{cell['name']}: request boundary differs\n"
        f"  legacy  : {[p[1][:120] for p in legacy_pairs]}\n"
        f"  pipeline: {[p[1][:120] for p in observed['pairs']]}"
    )
    assert observed["roles"] == legacy_roles
    assert observed["artifact"] == legacy_artifact


@pytest.mark.parametrize("cell", CELLS, ids=[cell["name"] for cell in CELLS])
def test_record_perf_parity_cells(cell: dict[str, Any], tmp_path_factory: Any) -> None:
    """Recording entry point: ``TESTAGENT_RECORD_PARITY=1`` writes the baseline
    from the LEGACY run; a default run only re-reads what is committed."""
    tmp = tmp_path_factory.mktemp(f"record-{cell['name']}")
    fixture = ensure_fixture(TASK, cell["name"], lambda tmp=tmp, cell=cell: _fixture(tmp, cell))
    assert fixture.task == TASK
