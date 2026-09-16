"""LinkGraph golden fixture tool (LINK-S0, plan-links-v15 §4 golden).

Records (``--record``) or replays (default) the deterministic graph and
planner outputs for the ecommerce fixture. Replaying after a code change
must be byte-identical; any diff is a hard failure. Recording writes a
provenance header (record time, rule version) and is only legitimate
BEFORE the changes under test (v15 §2 golden discipline).

Usage:
    .venv/bin/python scripts/golden_links.py --stage fixture --record
    .venv/bin/python scripts/golden_links.py --stage fixture
    .venv/bin/python scripts/golden_links.py --stage graph
    .venv/bin/python scripts/golden_links.py --stage planner
    .venv/bin/python scripts/golden_links.py --stage all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from testagent.config.models import APIEndpoint  # noqa: E402
from testagent.parsers.swagger_parser import SwaggerParser  # noqa: E402

GOLDEN_DIR = REPO / "tests" / "fixtures" / "links"


def _endpoints() -> list[APIEndpoint]:
    spec = json.loads((REPO / "examples" / "ecommerce_swagger.json").read_text(encoding="utf-8"))
    return SwaggerParser().parse(spec)


def _requirement_text() -> str:
    return (REPO / "examples" / "ecommerce_requirements.md").read_text(encoding="utf-8")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, check=True
        ).stdout.strip()[:12]
    except Exception:
        return "unknown"


def build_graph_fixture() -> dict:
    """Deterministic R1-R4 graph snapshot (no R6: prose off in fixture)."""
    from testagent.pipeline.links_graph import build_graph

    endpoints = _endpoints()
    graph = build_graph(endpoints)
    return {
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "kind": e.kind,
                "candidates": [
                    {
                        "producer": c.producer,
                        "consumer": c.consumer,
                        "mode": c.mode,
                        "rule": c.rule,
                    }
                    for c in e.binding_candidates
                ],
            }
            for e in sorted(graph.edges, key=lambda e: (e.source, e.target, e.kind))
        ],
    }


def build_planner_fixture() -> dict:
    """Planner output over R6-on graph (deterministic full pipeline)."""
    from testagent.pipeline.links_graph import build_graph
    from testagent.pipeline.links_r6 import extract_r6_edges
    from testagent.pipeline.module_aliases import BUILTIN_ALIASES
    from testagent.pipeline.pathplanner import plan_paths

    endpoints = _endpoints()
    r6, _ = extract_r6_edges(
        _requirement_text(), doc="ecom", aliases=BUILTIN_ALIASES, min_score=1.0
    )
    graph = build_graph(endpoints, r6_edges=r6)
    planning = plan_paths(graph, endpoints, max_planned=8, l3b_budget=3)
    return {
        "counts": planning["counts"],
        "invalid_attempts_by_reason": planning["invalid_attempts_by_reason"],
        "paths": [
            {
                "path_id": c.path_id,
                "endpoints": list(c.endpoints),
                "static_class": c.static_class,
                "priority": c.priority,
                "bindings": [
                    {
                        "producer": h.candidate.producer if h.candidate else "",
                        "consumer": h.candidate.consumer if h.candidate else "",
                        "mode": h.candidate.mode if h.candidate else "",
                    }
                    for h in c.business_hops
                    if h.kind == "DATA_FLOW"
                ],
            }
            for c in planning["planned"]
        ],
        "selected": [c.path_id for c in planning["selected"]],
        "deferred": planning["deferred"],
        "not_materialized": planning["not_materialized"],
    }


def build_cluster_fixture() -> dict:
    """L3a lifecycle clusters on the 13-endpoint real fixture (v15 §5.3)."""
    from testagent.pipeline.context_builder import ContextBuilder

    clusters = ContextBuilder(_endpoints()).split_clusters()
    return {"clusters": clusters, "cluster_count": len(clusters)}


STAGES = {
    "graph": ("links_graph.json", build_graph_fixture),
    "planner": ("links_planner.json", build_planner_fixture),
    "fixture": ("links_clusters.json", build_cluster_fixture),
}


def _payload_sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description="links golden fixtures")
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    parser.add_argument("--record", action="store_true", help="record golden (pre-change only)")
    args = parser.parse_args()
    stage_names = list(STAGES) if args.stage == "all" else [args.stage]

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for stage in stage_names:
        filename, builder = STAGES[stage]
        path = GOLDEN_DIR / filename
        payload = builder()
        if args.record:
            payload = {
                "provenance": {
                    "recorded_utc": datetime.now(UTC).isoformat(),
                    "git_sha": _git_sha(),
                    "rule_version": "v15",
                },
                "payload_sha": _payload_sha(payload),
                "payload": payload,
            }
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[golden][RECORDED] {stage} -> {path.relative_to(REPO)}")
            continue
        if not path.exists():
            failures.append(f"{stage}: golden missing ({filename}); record BEFORE the change")
            continue
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("payload_sha") != _payload_sha(stored["payload"]):
            failures.append(f"{stage}: golden file corrupted (sha mismatch)")
        elif stored["payload_sha"] != _payload_sha(payload):
            failures.append(f"{stage}: replay diff vs golden ({filename})")
        else:
            print(f"[golden][PASS] {stage}")
    if failures:
        for failure in failures:
            print(f"[golden][FAIL] {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
