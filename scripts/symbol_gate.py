"""Symbol gate (plan-k §8.3 / §12.7): who still references the symbols a
deletion step wants to remove — imports AND textual mentions, AST-verified.

A deletion gate may only close when this reports zero consumers outside the
files being deleted. Plain ``grep`` is not enough: a module can be referenced
through an import alias, a provider wiring line, or a docstring that lies, and
each of those needs a different verdict. So the output separates:

- ``IMPORT``  : a real ``import``/``from ... import`` edge (AST) — must be zero
- ``NAME``    : the identifier appears in code — must be zero (or explained)
- ``TEXT``    : only in comments/docstrings/markdown — harmless, but the doc
                must stop promising the symbol exists

Usage:
    .venv/bin/python scripts/symbol_gate.py TestCaseGenerator PerformanceGenerator
    .venv/bin/python scripts/symbol_gate.py --allow testagent/generators build_generate_unit
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("testagent", "tests", "scripts", "tasks")
Kind = Literal["IMPORT", "NAME", "TEXT"]


def _classify(node: ast.AST, names: set[str]) -> tuple[Kind, str] | None:
    """Import edges and code-level name uses; returns (kind, evidence)."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            for name in names:
                if name.lower() in alias.name.lower() or alias.name.endswith(f".{name.lower()}"):
                    return "IMPORT", alias.name
    if isinstance(node, ast.ImportFrom) and node.module:
        for alias in node.names:
            for name in names:
                if name in (alias.name, alias.asname or "") or name.lower() in node.module.lower():
                    return "IMPORT", f"{node.module}.{alias.name}"
    if isinstance(node, ast.Name):
        for name in names:
            if node.id == name:
                return "NAME", node.id
    if isinstance(node, ast.Attribute):
        for name in names:
            if node.attr == name:
                return "NAME", node.attr
    return None


def scan(symbols: list[str], allow: list[str]) -> dict[str, list[tuple[str, int, str, str]]]:
    """symbol -> [(file, line, kind, evidence)]"""
    hits: dict[str, list[tuple[str, int, str, str]]] = {s: [] for s in symbols}
    names = set(symbols)
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if any(rel.startswith(pref) for pref in allow):
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            try:
                tree = ast.parse("\n".join(lines))
            except SyntaxError:  # pragma: no cover - repo files parse
                tree = None

            ast_hits: dict[int, tuple[str, str]] = {}
            if tree is not None:
                for node in ast.walk(tree):
                    found = _classify(node, names)
                    if found is not None:
                        ast_hits[getattr(node, "lineno", 0)] = found

            for idx, line in enumerate(lines, 1):
                for symbol in symbols:
                    if symbol not in line:
                        continue
                    if idx in ast_hits and ast_hits[idx][0] == "IMPORT":
                        hits[symbol].append((rel, idx, "IMPORT", ast_hits[idx][1]))
                    elif idx in ast_hits:
                        hits[symbol].append((rel, idx, "NAME", ast_hits[idx][1]))
                    elif line.lstrip().startswith(("#", '"""', "'''", "*")):
                        hits[symbol].append((rel, idx, "TEXT", line.strip()[:70]))
                    else:
                        # mentioned on a code line without an import/attribute
                        # edge: string literals, provider names, help text
                        hits[symbol].append((rel, idx, "NAME", line.strip()[:70]))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="+")
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="path prefixes to skip (the files being deleted)",
    )
    parser.add_argument(
        "--fail-on",
        default="IMPORT,NAME",
        help="comma-separated kinds that make the gate fail (default: IMPORT,NAME)",
    )
    args = parser.parse_args()

    failing = {kind.strip() for kind in args.fail_on.split(",") if kind.strip()}
    hits = scan(args.symbols, args.allow)
    violations = 0
    for symbol, rows in hits.items():
        print(f"== {symbol}: {len(rows)} 处引用")
        for rel, line, kind, evidence in rows:
            flag = "FAIL" if kind in failing else "ok  "
            if kind in failing:
                violations += 1
            print(f"   [{flag}] {kind:6s} {rel}:{line}  {evidence}")
    print(
        f"\n[symbol-gate] symbols={args.symbols} allow={args.allow or '-'} violations={violations}"
    )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
