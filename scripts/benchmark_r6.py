"""R6 benchmark (LINK-S2a/S3, plan-links-v15 §11.2).

Three-layer evaluation over the r6_benchmark fixtures:
- layer 1: undirected module-pair precision / recall (>= 0.75);
- layer 2: directed-relation precision / recall (N>=20: precision >= 0.95,
  recall >= 0.75; small N: precision = 1.0);
- layer 3 (S3+): selected R6 hops must be grounded in the human oracle.

Hard failure on: truth non-empty but prediction empty, abnormal
denominators, threshold miss. Exit code 0 = pass; 1 = fail.

Usage:
    .venv/bin/python scripts/benchmark_r6.py --stage graph
    .venv/bin/python scripts/benchmark_r6.py --stage selected
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from testagent.config.models import APIEndpoint  # noqa: E402
from testagent.parsers.swagger_parser import SwaggerParser  # noqa: E402
from testagent.pipeline.links_graph import build_graph  # noqa: E402
from testagent.pipeline.links_r6 import extract_r6_edges  # noqa: E402
from testagent.pipeline.module_aliases import BUILTIN_ALIASES  # noqa: E402
from testagent.pipeline.pathplanner import plan_paths  # noqa: E402

MIN_PAIR = 0.75
MIN_DIR_PRECISION_LARGE = 0.95
MIN_DIR_RECALL = 0.75


def _load_fixtures() -> tuple[list[APIEndpoint], str, dict]:
    spec = json.loads((REPO / "examples" / "r6_benchmark_swagger.json").read_text(encoding="utf-8"))
    endpoints = SwaggerParser().parse(spec)
    text = (REPO / "examples" / "r6_benchmark_requirements.md").read_text(encoding="utf-8")
    truth = json.loads((REPO / "examples" / "r6_truth.json").read_text(encoding="utf-8"))
    return endpoints, text, truth


def _predict_edges(text: str):
    edges, _abstained = extract_r6_edges(
        text,
        doc="R",
        aliases=BUILTIN_ALIASES,
        min_score=1.0,
    )
    return edges


def _layers(edges, truth: dict) -> tuple[dict, dict]:
    predicted_pairs: set[tuple[str, str]] = set()
    predicted_directed: set[tuple[str, str, str]] = set()
    for edge in edges:
        predicted_pairs.add(tuple(sorted((edge.source, edge.target))))  # type: ignore[assignment]
        predicted_directed.add((edge.source, edge.target, edge.kind))

    truth_pairs: set[tuple[str, str]] = set()
    truth_directed: set[tuple[str, str, str]] = set()
    for rel in truth["directed_relations"]:
        truth_pairs.add(tuple(sorted((rel["source"], rel["target"]))))
        truth_directed.add((rel["source"], rel["target"], rel["kind"]))

    pair_hits = len(predicted_pairs & truth_pairs)
    dir_hits = len(predicted_directed & truth_directed)

    def pr(hits: int, pred: int, truth_n: int) -> dict:
        precision = round(hits / pred, 4) if pred else 0.0
        recall = round(hits / truth_n, 4) if truth_n else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "predicted": pred,
            "truth": truth_n,
            "hits": hits,
        }

    return pr(pair_hits, len(predicted_pairs), len(truth_pairs)), pr(
        dir_hits, len(predicted_directed), len(truth_directed)
    )


def run_graph_stage() -> int:
    _endpoints, text, truth = _load_fixtures()
    edges = _predict_edges(text)
    pair_layer, dir_layer = _layers(edges, truth)
    n = dir_layer["truth"]

    print(f"[r6] module pairs: {pair_layer}")
    print(f"[r6] directed relations: {dir_layer}")

    failures: list[str] = []
    if n == 0:
        failures.append("truth is empty")
    if pair_layer["precision"] < MIN_PAIR or pair_layer["recall"] < MIN_PAIR:
        failures.append(f"module-pair thresholds not met (min {MIN_PAIR})")
    if n >= 20:
        if dir_layer["precision"] < MIN_DIR_PRECISION_LARGE:
            failures.append(f"directed precision < {MIN_DIR_PRECISION_LARGE} at N>=20")
    else:
        if dir_layer["precision"] < 1.0:
            failures.append("directed precision must be 1.0 for N<20")
    if n and dir_layer["recall"] < MIN_DIR_RECALL:
        failures.append(f"directed recall < {MIN_DIR_RECALL}")
    if n and dir_layer["predicted"] == 0:
        failures.append("truth non-empty but prediction empty")

    if failures:
        for failure in failures:
            print(f"[r6][FAIL] {failure}")
        return 1
    print("[r6][PASS] graph stage")
    return 0


def run_selected_stage() -> int:
    endpoints, text, truth = _load_fixtures()
    edges = _predict_edges(text)
    graph = build_graph(endpoints, r6_edges=edges)
    planning = plan_paths(graph, endpoints, max_planned=8, l3b_budget=3)

    def _module_of(endpoint_identity: str) -> str:
        path = endpoint_identity.split(" ", 1)[1]
        ep = next(
            (
                e
                for e in endpoints
                if e.path == path and e.method.upper() == endpoint_identity.split(" ", 1)[0]
            ),
            None,
        )
        if ep is None:
            return ""
        tags = ep.tags or []
        if tags and str(tags[0]).strip():
            return str(tags[0]).strip().casefold()
        segments = [s for s in path.strip("/").split("/") if s and not s.startswith("{")]
        return segments[0].casefold() if segments else "root"

    selected_r6: list[tuple[str, str, str]] = []
    for contract in planning["selected"]:
        for hop in contract.business_hops:
            if hop.candidate is not None and hop.candidate.rule == "R6":
                selected_r6.append((hop.source, hop.target, hop.kind))
    print(f"[r6] selected R6 hops: {selected_r6}")
    allowed_modules = {(r["source"], r["target"]) for r in truth["directed_relations"]}
    allowed_endpoints = {(g["source"], g["target"]) for g in truth.get("allowed_groundings", [])}
    bad = [
        h
        for h in selected_r6
        if (_module_of(h[0]), _module_of(h[1])) not in allowed_modules
        and (h[0], h[1]) not in allowed_endpoints
    ]
    if bad:
        print(f"[r6][FAIL] selected R6 hops outside oracle: {bad}")
        return 1
    print("[r6][PASS] selected stage")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R6 benchmark")
    parser.add_argument("--stage", choices=["graph", "selected"], default="graph")
    args = parser.parse_args()
    if args.stage == "graph":
        return run_graph_stage()
    return run_selected_stage()


if __name__ == "__main__":
    raise SystemExit(main())
