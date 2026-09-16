"""Audit cross-module context loss in the current segmented generation.

Evidence gathering for the "module segmentation loses cross-module links" plan:
1. test_type distribution in the best production artifact (v4)
2. how many cases actually span more than one endpoint (real integration cases)
3. how many cross-module dependency edges are recoverable from the swagger
   spec by pure code (proving the link graph is cheap to build, IF the parser
   keeps response schemas -- today it does not)
4. token cost of a global signature vs the full endpoints text
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, cwd-independent

# ---------------------------------------------------------------- 1 + 2
cases = json.loads((ROOT / "output/ecommerce_testcases_v4.json").read_text(encoding="utf-8"))
print(f"[1] v4 artifact: {len(cases)} cases")

dist: dict[str, int] = {}
for c in cases:
    dist[c.get("test_type", "?")] = dist.get(c.get("test_type", "?"), 0) + 1
print("    test_type distribution:")
for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
    print(f"      {k:<12} {v:>3}  ({v / len(cases):.1%})")

_EP = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/[^\s,;]*)")


def endpoints_touched(case: dict) -> set[str]:
    blob = " ".join(
        list(case.get("steps") or [])
        + list(case.get("preconditions") or [])
        + list(case.get("expected_results") or [])
    )
    return {f"{m} {p.rstrip('.,;')}" for m, p in _EP.findall(blob)}


multi = [c for c in cases if len(endpoints_touched(c)) > 1]
print(f"[2] cases whose steps touch >1 endpoint: {len(multi)}/{len(cases)}")
for c in multi[:6]:
    print(f"      - {c['id']} [{c.get('test_type')}] {sorted(endpoints_touched(c))}")

# Cases that REFERENCE an id placeholder but never create it.
placeholders = re.compile(r"<([A-Z0-9_]+)>")
produced: set[str] = set()
consumed: dict[str, list[str]] = {}
for c in cases:
    text = " ".join(list(c.get("steps") or []) + list(c.get("expected_results") or []))
    for ph in placeholders.findall(text):
        consumed.setdefault(ph, []).append(c["id"])
    for m in re.finditer(r"(?:store|save|record|capture)[^.\n]*?as\s*<([A-Z0-9_]+)>", text, re.I):
        produced.add(m.group(1))
orphan = {p: ids for p, ids in consumed.items() if p not in produced and p not in ("TOKEN",)}
print(f"[3] placeholders referenced without a producing step: {len(orphan)}")
for p, ids in sorted(orphan.items(), key=lambda kv: -len(kv[1]))[:10]:
    print(f"      <{p}> used by {len(ids)} cases e.g. {ids[:4]}")

# ---------------------------------------------------------------- 4
spec = json.loads((ROOT / "examples/ecommerce_swagger.json").read_text(encoding="utf-8"))


def resolve(node: dict, depth: int = 0) -> dict:
    """Best-effort $ref resolution inside one document."""
    if depth > 6 or not isinstance(node, dict):
        return {}
    if "$ref" in node:
        ref = node["$ref"].split("/")[1:]
        target: object = spec
        for part in ref:
            target = target[part]  # type: ignore[index]
        return resolve(target, depth + 1)  # type: ignore[arg-type]
    return node


produces: dict[str, set[str]] = {}  # endpoint -> response field names
consumes: dict[str, set[str]] = {}  # endpoint -> path/query/body field names
tags_of: dict[str, str] = {}

for path, item in spec.get("paths", {}).items():
    for method, op in item.items():
        if not isinstance(op, dict) or method.lower() not in (
            "get",
            "post",
            "put",
            "patch",
            "delete",
        ):
            continue
        key = f"{method.upper()} {path}"
        tags_of[key] = (op.get("tags") or ["<none>"])[0]

        # fields this endpoint CONSUMES
        fields: set[str] = set()
        for p in list(item.get("parameters", [])) + list(op.get("parameters", [])):
            fields.add(str(p.get("name", "")))
        body = (op.get("requestBody") or {}).get("content") or {}
        for media in body.values():
            props = resolve(media.get("schema", {})).get("properties", {}) or {}
            fields.update(props.keys())
        consumes[key] = {f for f in fields if f}

        # fields this endpoint PRODUCES (200/201 response schema)
        out: set[str] = set()
        for code in ("200", "201"):
            resp = (op.get("responses") or {}).get(code)
            if not isinstance(resp, dict):
                continue
            for media in (resp.get("content") or {}).values():
                schema = resolve(media.get("schema", {}))
                props = schema.get("properties", {}) or {}
                # unwrap {data: {...}} envelopes one level
                out.update(props.keys())
                for sub in ("data", "result", "items"):
                    inner = resolve(props.get(sub, {})) if isinstance(props.get(sub), dict) else {}
                    out.update((inner.get("properties") or {}).keys())
        produces[key] = {f for f in out if f}

edges: list[tuple[str, str, str]] = []
for src, outs in produces.items():
    for dst, ins in consumes.items():
        if src == dst:
            continue
        shared = outs & ins
        # ignore generic fields that appear everywhere
        shared = {f for f in shared if f not in {"id", "name"}}
        for f in sorted(shared):
            edges.append((src, dst, f))

print(f"[4] endpoints: {len(tags_of)}; cross-endpoint data edges recoverable by code: {len(edges)}")
for src, dst, f in edges[:14]:
    cross = "  <-- CROSS-MODULE" if tags_of[src] != tags_of[dst] else ""
    print(f"      {src}  --{f}-->  {dst}   [{tags_of[src]}->{tags_of[dst]}]{cross}")
cross_edges = [e for e in edges if tags_of[e[0]] != tags_of[e[1]]]
print(f"    of which cross-module: {len(cross_edges)}")

# ---------------------------------------------------------------- 5
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))
from testagent.engine.prompt_builder import endpoints_to_signature  # noqa: E402
from testagent.parsers.swagger_parser import SwaggerParser  # noqa: E402

eps = SwaggerParser().parse(str(ROOT / "examples/ecommerce_swagger.json"))
full = SwaggerParser.endpoints_to_text(eps)
sig = endpoints_to_signature(eps)
print("[5] global context size on the 13-endpoint ecommerce example:")
print(f"      endpoints_to_text (full)  : {len(full):>5} chars")
print(
    f"      endpoints_to_signature    : {len(sig):>5} chars  ({len(sig) / len(full):.0%} of full)"
)
print(
    f"      per-batch (2 eps) text    : ~{len(full) // 7:>5} chars  -> sees {2}/{len(eps)} endpoints"
)
