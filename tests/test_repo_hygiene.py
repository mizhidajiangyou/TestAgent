"""Repo hygiene gates (written because these rules previously lived only in
prose, and prose does not fail a build).

Three things get enforced:

1. **Milestone gate archives are real**: every ``output/gates/<dir>`` has a
   ``manifest.json`` with a resolvable ``commit_sha``, the required provenance
   keys, a ``PASS`` verdict, and a non-empty gate list whose referenced log
   files exist next to it. "无归档视为门未过" only means something if a check
   validates the archive's shape.
2. **Tracked files carry no machine-specific paths**: an absolute
   ``/Users/<name>/...`` inside committed evidence leaks the operator's machine
   and reads differently on every clone.
3. **Tracked text cites only tracked paths**: a doc saying "see
   ``output/e2e_bookstore_review_swagger.json``" is unreadable after a clone,
   because that path is gitignored. Citations must resolve inside the repo
   (``examples/`` for artifacts, ``output/gates/`` for evidence); a documented
   runtime output is written ``./output/...`` and is not a citation.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
GATES = REPO / "output" / "gates"

#: A repo-path citation ends in one of these extensions.
PATH_TOKEN_RE = re.compile(
    r"(?P<lead>[\w./-]?)"
    r"(?P<path>(?:output|examples|tests|scripts|testagent|tasks|monitoring|docker)/"
    r"[A-Za-z0-9_.\-/]*?[A-Za-z0-9_\-.]+\.(?:json|md|js|py|jmx|log|csv|txt|yaml|yml|j2))"
)
MACHINE_PATH_RES = (
    re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    re.compile(r"/home/[A-Za-z0-9._-]+/"),
    re.compile(r"C:\\Users\\", re.I),
)
SCAN_SUFFIXES = {".md", ".json", ".py", ".yml", ".yaml", ".j2", ".toml", ".cfg", ".log"}
#: "正文"= markdown 散文。围栏代码块里的是命令参数、配置文件里的是挂载路径,
#: 都不算引用 (否则 README 的 ``-o output/x.json`` 示例会被误判)。
DOC_SUFFIXES = {".md"}
FENCE_RE = re.compile(r"^(```|~~~)", re.M)
MAX_SCAN_BYTES = 2_000_000


def _reachable_set() -> frozenset[str]:
    # --cached --others --exclude-standard = 已入库 + 未入库但不会被忽略的文件。
    # 用"会不会被 ignore"而不是"是否已 git add"作判据: 证据必须对克隆可达,
    # 与暂存状态无关。
    out = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return frozenset(p for p in out.split("\0") if p)


@pytest.fixture(scope="module")
def reachable() -> frozenset[str]:
    return _reachable_set()


def _gate_dirs() -> list[Path]:
    assert GATES.is_dir(), "output/gates/ 不存在"
    return sorted(d for d in GATES.iterdir() if d.is_dir())


def _scannable(rel: str, reachable: frozenset[str]) -> str | None:
    """Return a repo-reachable file's text, or None when it should be skipped."""
    if rel not in reachable or Path(rel).suffix not in SCAN_SUFFIXES:
        return None
    path = REPO / rel
    try:
        if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
            return None
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:  # pragma: no cover - unreadable blob, not a doc problem
        return None


def _fenced_spans(text: str) -> list[tuple[int, int]]:
    """Byte ranges covered by ``` / ~~~ fenced code blocks."""
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    for match in FENCE_RE.finditer(text):
        if open_at is None:
            open_at = match.start()
        else:
            spans.append((open_at, match.end()))
            open_at = None
    if open_at is not None:  # unterminated fence: treat the rest as code
        spans.append((open_at, len(text)))
    return spans


class TestGateArchives:
    """plan-k §9.3: the archive IS the milestone's evidence."""

    def test_naming_manifest_and_replayable_body(self) -> None:
        dirs = _gate_dirs()
        assert dirs, "output/gates/ 里一个归档都没有——没有任何东西能证明里程碑过过门"
        for d in dirs:
            assert re.fullmatch(r"\d{8}-.+-[0-9a-f]{7}", d.name), (
                f"{d.name}: 归档目录必须命名成 <UTC日期>-<节点>-<sha7>"
            )
            assert (d / "manifest.json").is_file(), f"{d.name} 缺 manifest.json"
            assert (d / "summary.md").is_file() or list(d.glob("*.log")), (
                f"{d.name}: 既无 summary.md 也无日志, 正文无法独立复核"
            )

    def test_manifest_provenance_and_verdict(self) -> None:
        for d in _gate_dirs():
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            for key in ("milestone", "date_utc", "commit_sha", "verdict"):
                assert key in man, f"{d.name}: manifest 缺字段 {key}"
            assert str(man["verdict"]).startswith("PASS"), (
                f"{d.name}: verdict={man['verdict']!r} —— 未过的门不能作为证据入库"
            )
            sha = str(man["commit_sha"])
            probe = subprocess.run(
                ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
                cwd=REPO,
                capture_output=True,
                check=False,
            )
            assert probe.returncode == 0, (
                f"{d.name}: commit_sha={sha} 解析不到 commit —— 证据没钉住它验证的那个 commit"
            )
            assert sha.startswith(d.name.rsplit("-", 1)[-1]), (
                f"{d.name}: 目录名短 sha 与 manifest.commit_sha 不一致"
            )

    def test_gate_entries_have_cmd_and_outcome(self) -> None:
        for d in _gate_dirs():
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            entries = man.get("gates") or man.get("commands") or man.get("checks") or []
            assert entries, f"{d.name}: 门清单为空"
            for entry in entries:
                assert entry.get("cmd"), f"{d.name}: 门条目缺 cmd, 无法复跑"
                has_outcome = (
                    entry.get("result")
                    or entry.get("exit") is not None
                    or entry.get("exit_code") is not None
                )
                assert has_outcome, f"{d.name}: 门条目 {entry.get('cmd')!r} 既无结果也无退出码"
                log = entry.get("log")
                if log:
                    assert (d / log).is_file(), f"{d.name}: 引用的日志 {log} 不在归档里"


class TestNoMachinePaths:
    def test_repo_files_are_machine_neutral(self, reachable: frozenset[str]) -> None:
        offenders = [
            rel
            for rel in sorted(reachable)
            if (text := _scannable(rel, reachable)) is not None
            and any(rx.search(text) for rx in MACHINE_PATH_RES)
        ]
        assert not offenders, f"入库文件含机器绝对路径: {offenders[:8]}"


class TestCitationsResolve:
    def test_repo_text_cites_only_reachable_paths(self, reachable: frozenset[str]) -> None:
        offenders: list[str] = []
        for rel in sorted(reachable):
            if Path(rel).suffix not in DOC_SUFFIXES:
                continue
            text = _scannable(rel, reachable)
            if text is None:
                continue
            in_fence = _fenced_spans(text)
            for match in PATH_TOKEN_RE.finditer(text):
                if match.group("lead") in {".", "/"}:
                    continue  # ./output/x.json = 运行期输出, 不是仓库内容引用
                if any(a <= match.start() < b for a, b in in_fence):
                    continue  # 代码块内: 命令参数 / 示例输出
                token = match.group("path")
                if token.startswith("output/") and not token.startswith("output/gates/"):
                    offenders.append(f"{rel}: 未入库的运行产物被引用 → {token}")
                elif token not in reachable:
                    offenders.append(f"{rel}: 引用路径不在仓库里 → {token}")
        assert not offenders, (
            "入库正文的引用必须可达且已入库 (把数字抄进正文, 或放 examples/):\n"
            + "\n".join(offenders[:15])
        )
