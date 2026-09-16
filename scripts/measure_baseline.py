"""Baseline measurement for the links pipeline (LINK-S0, plan-links-v15
§2 / §10).

Recomputes the module/endpoint/relation counts and L0 character size over
the ecommerce fixtures. Hard-fails (non-zero exit) on missing data or
assertion failures — an observation script that cannot fail is no gate.

Usage: .venv/bin/python scripts/measure_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from testagent.config.models import APIEndpoint  # noqa: E402
from testagent.parsers.swagger_parser import SwaggerParser  # noqa: E402


def _module_of(ep: APIEndpoint) -> str:
    tags = ep.tags or []
    if tags and str(tags[0]).strip():
        return str(tags[0]).strip().casefold()
    segments = [s for s in ep.path.strip("/").split("/") if s and not s.startswith("{")]
    return segments[0].casefold() if segments else "root"


def main() -> int:
    spec_path = REPO / "examples" / "ecommerce_swagger.json"
    if not spec_path.exists():
        print(f"[measure][FAIL] spec missing: {spec_path}")
        return 2
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    endpoints = SwaggerParser().parse(spec)

    modules: dict[str, int] = {}
    for ep in endpoints:
        modules[_module_of(ep)] = modules.get(_module_of(ep), 0) + 1

    from testagent.pipeline.context_builder import ContextBuilder

    l0 = ContextBuilder(endpoints).build_l0()

    print("[measure] endpoints:", len(endpoints))
    print("[measure] modules:", json.dumps(modules, ensure_ascii=False, sort_keys=True))
    print("[measure] l0_chars:", len(l0))

    failures: list[str] = []
    if len(endpoints) == 0:
        failures.append("no endpoints parsed")
    if len(modules) < 2:
        failures.append("expected multiple modules for cross-module measurement")
    if len(l0) == 0:
        failures.append("L0 rendering is empty")
    # v15 s2 verified baseline: 13 endpoints.
    if len(endpoints) != 13:
        failures.append(f"baseline asserts 13 endpoints, got {len(endpoints)}")
    if failures:
        for failure in failures:
            print(f"[measure][FAIL] {failure}")
        return 1
    print("[measure][PASS]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
