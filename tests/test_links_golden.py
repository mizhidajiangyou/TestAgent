"""Links golden replay + R6 benchmark as pytest gates (defect ⑩).

`scripts/golden_links.py` replay and `scripts/benchmark_r6.py` are hard
gates; wrapping them as tests puts them on the default pytest path (and
therefore CI) instead of relying on someone remembering to run the
scripts. The subprocess isolates the script's sys.path bootstrap.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )


class TestLinksGates:
    def test_golden_replay_all_stages(self) -> None:
        result = _run("golden_links.py", "--stage", "all")
        assert result.returncode == 0, f"golden replay failed:\n{result.stdout}\n{result.stderr}"

    def test_r6_benchmark_graph_stage(self) -> None:
        result = _run("benchmark_r6.py", "--stage", "graph")
        assert result.returncode == 0, f"R6 graph gate failed:\n{result.stdout}\n{result.stderr}"

    def test_r6_benchmark_selected_stage(self) -> None:
        result = _run("benchmark_r6.py", "--stage", "selected")
        assert result.returncode == 0, f"R6 selected gate failed:\n{result.stdout}\n{result.stderr}"

    def test_measure_baseline(self) -> None:
        result = _run("measure_baseline.py")
        assert result.returncode == 0, f"baseline measure failed:\n{result.stdout}\n{result.stderr}"
