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
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GATES_ROOT = REPO / "output" / "gates"


def _gate_list(py: str) -> list[tuple[str, list[str]]]:
    """(id, argv) — plan-k §9.1 全量门 + task.md §五 追加的 links 两道门."""
    testagent = [str(REPO / ".venv" / "bin" / "testagent")]
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
    ]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", required=True, help="milestone label, e.g. FH2.6-b7.1")
    parser.add_argument(
        "--only", default="", help="comma-separated gate ids to run (default: all)"
    )
    parser.add_argument("--note", default="", help="one-line verdict note")
    args = parser.parse_args()

    only = {part.strip() for part in args.only.split(",") if part.strip()}
    py = sys.executable
    gates = [gate for gate in _gate_list(py) if not only or gate[0] in only]
    if not gates:
        print(f"[gate][FAIL] --only matched nothing (known: {_gate_list(py)})")
        return 2

    sha = _git("rev-parse", "--short", "HEAD") or "nohead"
    started = datetime.now(UTC)
    target = GATES_ROOT / f"{started:%Y%m%d}-{args.node}-{sha[:7]}"
    target.mkdir(parents=True, exist_ok=True)

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
            f"$ {' '.join(argv)}\n[cwd] {REPO}\n[exit] {proc.returncode}\n\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}\n",
            encoding="utf-8",
        )
        tail = (proc.stdout or proc.stderr).strip().splitlines()
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
    dirty = bool(_git("status", "--porcelain"))
    manifest = {
        "milestone": args.node,
        "date_utc": started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "commit_sha": sha,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "executor": "agent (Qoder session) + " + os.environ.get("USER", "unknown"),
        "gates": results,
        "note": args.note,
        "verdict": "PASS" if verdict else "FAIL",
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[gate] archive: {target.relative_to(REPO)} verdict={manifest['verdict']}")
    if dirty:
        print("[gate][WARN] working tree dirty — the archive does not describe HEAD exactly")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
