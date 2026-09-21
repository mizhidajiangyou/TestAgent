"""Milestone gate archiver (plan-k §9.3): run the full gate battery, write
``output/gates/<date>-<node>-<sha>/manifest.json`` + one log per gate.

Every gate runs as its own subprocess and its REAL exit code is recorded —
never behind a shell pipe (a ``| tail`` eats the code and turns a red gate
into a green one). The script itself exits non-zero when any gate fails, so
"no record" and "record says FAIL" cannot be confused with a pass.

Usage:
    .venv/bin/python scripts/gate_archive.py --node FH2.6-b7.1-web-switch
    .venv/bin/python scripts/gate_archive.py --node X --only pytest,mypy
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GATES_ROOT = REPO / "output" / "gates"


def _gate_list(py: str) -> list[tuple[str, list[str]]]:
    """(id, argv) — plan-k §9.1 全量门 + task.md §五 追加的 links 两道门."""
    testagent = [str(Path(".venv") / "bin" / "testagent")]
    return [
        ("ruff", [py, "-m", "ruff", "check", "testagent/", "tests/", "scripts/"]),
        ("format", [py, "-m", "ruff", "format", "--check", "testagent/", "tests/", "scripts/"]),
        ("mypy", [py, "-m", "mypy", "testagent/"]),
        ("pytest", [py, "-m", "pytest", "tests/", "-q"]),
        ("arch", [py, "-m", "pytest", "tests/test_pipeline_e2e.py::TestArchitectureGate", "-q"]),
        ("engine-golden", [py, "-m", "pytest", "tests/test_engine_event_baseline.py", "-q"]),
        ("tasks-validate", [*testagent, "tasks", "validate", "--strict"]),
        ("links-golden", [py, "scripts/golden_links.py", "--stage", "all"]),
        ("r6-graph", [py, "scripts/benchmark_r6.py", "--stage", "graph"]),
        ("r6-selected", [py, "scripts/benchmark_r6.py", "--stage", "selected"]),
        # LINK-S7 stage-exit commands (v15 §9/§10): the 13-endpoint baseline
        # measurement and the links-off rollback proof. The second one is the
        # only gate that can show "links wired" and "links-off bytes unchanged"
        # in the same archive.
        ("links-baseline", [py, "scripts/measure_baseline.py"]),
        ("links-parity-off", [py, "scripts/parity_links_off.py"]),
    ]


def _rel_py() -> str:
    """Interpreter path relative to the repo: evidence files are tracked, so an
    absolute ``/Users/<name>/...`` in them leaks the operator's machine and
    reads differently on every other clone (rule: 入库正文只引用仓库内路径)."""
    # Unresolved: sys.executable is the venv launcher (a symlink), and that is
    # the identity worth recording. Resolving it lands on a homebrew interpreter
    # outside the repo and erases the venv from the evidence.
    exe = Path(sys.executable)
    try:
        return str(exe.relative_to(REPO))
    except ValueError:
        return "<python>"


def _sanitize(text: str) -> str:
    """Strip the operator's machine out of tracked evidence.

    Logs embed pytest's own paths (warning headers, docs URLs); an absolute
    ``/Users/<name>/...`` inside a committed artifact reads differently on every
    other clone and names a person who is not part of the measurement.
    """
    home = str(Path.home())
    for needle, replacement in ((str(REPO), "<repo>"), (home, "~")):
        if needle and needle != str(Path(__file__).root):
            text = text.replace(needle, replacement)
    return text


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="milestone label, e.g. FH2.6-b7.1")
    parser.add_argument("--only", default="", help="comma-separated gate ids to run (default: all)")
    parser.add_argument("--note", default="", help="one-line verdict note")
    args = parser.parse_args()

    only = {part.strip() for part in args.only.split(",") if part.strip()}
    py = _rel_py()
    gates = [gate for gate in _gate_list(py) if not only or gate[0] in only]
    if not gates:
        print(f"[gate][FAIL] --only matched nothing (known: {_gate_list(py)})")
        return 2

    sha = _git("rev-parse", "--short", "HEAD") or "nohead"
    # Measured BEFORE the archive dir exists: creating it makes the tree
    # dirty, which would make every archive self-report as not-HEAD-exact.
    dirty = bool(_git("status", "--porcelain"))
    started = datetime.now(UTC)
    name = f"{started:%Y%m%d}-{args.node}-{sha[:7]}"
    target = GATES_ROOT / name
    if target.exists():
        print(f"[gate][FAIL] {target} already exists — archives are immutable evidence")
        return 2
    # Stage OUTSIDE output/gates and move into place only once the manifest is
    # written. Writing the logs in place made the half-finished archive visible
    # to the pytest gate inside the very same run (tests/test_repo_hygiene
    # rejects a gate dir without a manifest), so every archive self-reported
    # FAIL — the observer contaminated the observation.
    staged = REPO / "output" / ".gates-staging" / name
    staged.mkdir(parents=True, exist_ok=True)
    target = staged  # every write below goes to the staging copy

    results: list[dict[str, Any]] = []
    for gate_id, argv in gates:
        proc = subprocess.run(
            argv,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        log = target / f"{gate_id}.log"
        log.write_text(
            _sanitize(
                f"$ {' '.join(argv)}\n[cwd] <repo>\n[exit] {proc.returncode}\n\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n"
            ),
            encoding="utf-8",
        )
        tail = _sanitize(proc.stdout or proc.stderr).strip().splitlines()
        results.append(
            {
                "id": gate_id,
                "cmd": " ".join(argv),
                "exit": proc.returncode,
                "result": "PASS" if proc.returncode == 0 else "FAIL",
                "log": log.name,
                "summary_tail": " | ".join(tail[-3:]) if tail else "",
            }
        )
        print(f"[gate] {gate_id}: {'PASS' if proc.returncode == 0 else 'FAIL'}")

    verdict = all(item["exit"] == 0 for item in results)
    manifest = {
        "milestone": args.node,
        "date_utc": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "commit_sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        # No operator name: this file is tracked, and "who ran it" is not part
        # of what makes the evidence replayable.
        "executor": "agent session (Qoder)",
        "gates": results,
        "note": args.note,
        "verdict": "PASS" if verdict else "FAIL",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    final = GATES_ROOT / name
    shutil.move(str(target), str(final))
    staging_parent = target.parent
    if not any(staging_parent.iterdir()):
        staging_parent.rmdir()
    print(f"[gate] archive: {final.relative_to(REPO)} verdict={manifest['verdict']}")
    if dirty:
        print("[gate][WARN] working tree dirty — the archive does not describe HEAD exactly")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
