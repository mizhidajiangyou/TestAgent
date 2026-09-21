"""LINK-S7 rollback proof: a links-enabled CONFIG must be a links-absent RUN.

v15 §8.2 states the comparison this script cannot fake: the eight dimensions
are replayed after the S5b/S6b wiring against the baselines frozen BEFORE the
wiring existed, with ``LINKS_ENABLED=true`` set anyway. Diff zero means the
package opt-in really is the switch — no prompt material, no identity columns,
no gate report, and no sidecar file written next to the snapshot.

Each fixture replays in a FRESH subprocess (v15 §8.3): one run's registry,
settings cache and cwd must not be able to color another's readings.

Usage:
    .venv/bin/python scripts/parity_links_off.py
Exit: 0 all cells identical, 1 any difference (a difference is never waived).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "fixtures" / "migration" / "testcase" / "MANIFEST.json"


def subset_ids() -> list[str]:
    """The mechanically frozen ids (no add/remove rights here, plan-l L-4)."""
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    subset = document.get("post_wiring_replay_subset") or {}
    ids = sorted({fixture_id for group in subset.values() for fixture_id in group})
    if not ids:
        raise SystemExit("post_wiring_replay_subset is empty — FH2.4 cannot be passed")
    return ids


def run_one(fixture_id: str) -> int:
    """Replay one fixture in-process with links configured ON, package OFF."""
    sys.path.insert(0, str(REPO))
    from tests.parity_harness import load_fixture, minimal_diff
    from tests.test_migration_parity_testcase import (
        SCENARIOS,
        _fixture_name,
        _run_taskcase,
        _write_inputs,
    )

    scenario = fixture_id.removeprefix("tc-")
    config = SCENARIOS[scenario]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        raw = _write_inputs(tmp, historical=config.get("historical"))
        replayed = _run_taskcase(
            tmp,
            raw,
            list(config["responses"]),
            finish=config.get("finish", "stop"),
            review_enabled=config.get("review", False),
            settings_overrides={"links_enabled": True, "links_prose_enabled": True},
        )
        sidecars = list((tmp / "output").glob("*.links.json"))
    recorded = load_fixture("testcase", _fixture_name(scenario))
    diff = minimal_diff(recorded, replayed)
    if sidecars:
        print(f"  links sidecar written with the package opt-in off: {sidecars}")
    if diff:
        print(f"  {fixture_id}: {json.dumps(diff, ensure_ascii=False)[:400]}")
    ok = not diff and not sidecars
    print(f"{'PASS' if ok else 'FAIL'} {fixture_id} (links config ON, package OFF)")
    return 0 if ok else 1


def main() -> int:
    if "--_one" in sys.argv:
        return run_one(sys.argv[sys.argv.index("--_one") + 1])
    failures = 0
    for fixture_id in subset_ids():
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "parity_links_off.py"), "--_one", fixture_id],
            capture_output=True,
            text=True,
            cwd=str(REPO),
            check=False,
        )
        print(proc.stdout, end="")
        if proc.stderr.strip():
            print(proc.stderr.strip(), file=sys.stderr)
        if proc.returncode != 0:
            failures += 1
    print(f"parity_links_off: {failures} failed cell(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
