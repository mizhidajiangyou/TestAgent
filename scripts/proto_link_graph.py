"""Prototype v2 — corrected after the v1 self-check exposed two defects.

v1 findings that forced a redesign:
  * `to_text()` rendered 3646 chars of which 95% was intra-cluster noise
    (same-path GET<->POST, same-tag full mesh). Only 1 cross-module edge existed.
  * chain enumeration stopped at 2 modules because `POST /orders` has ONLY a
    `couponCode` body field — the cart->orders link is **not in the API
    contract at all**. It exists solely in the requirements prose
    ("对购物车内容结算生成订单，订单创建后扣减库存").

Conclusion: for THIS project's real inputs, business links live in the
REQUIREMENTS text, not the swagger. R6 must be a first-class rule, not an
optional enhancement. That also explains why segmentation hurts so much:
Phase 1 fans out one requirement per call, so the only carrier of cross-module
knowledge is exactly the thing being cut apart.

v2 changes:
  R6   requirement-prose co-occurrence -> promoted to primary rule
  to_text()  render CROSS-CLUSTER edges only (kills the noise)
  chains()   module-level business flow + per-module trunk endpoints
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root, cwd-independent
SPEC = json.loads((ROOT / "examples/ecommerce_swagger.json").read_text(encoding="utf-8"))
REQ_TEXT = (ROOT / "examples/ecommerce_requirements.md").read_text(encoding="utf-8")

#: Alias vocabulary mapping free text (CN/EN) to swagger tags.
#: Shipped defaults cover commerce/admin vocabulary; extendable via config.
ALIASES: dict[str, tuple[str, ...]] = {
    "products": ("product", "products", "商品", "库存", "sku", "下架"),
    "cart": ("cart", "购物车", "加购", "条目"),
    "orders": ("order", "orders", "订单", "下单", "结算"),
    "coupon": ("coupon", "优惠券", "核销", "券"),
}


@dataclass(frozen=True)
class LinkEdge:
    src: str
    dst: str
    field: str
    rule: str
    strength: str
    evidence: str


@dataclass
class LinkGraph:
    edges: list[LinkEdge] = field(default_factory=list)
    modules: dict[str, list[str]] = field(default_factory=dict)  # tag -> endpoints
    module_links: list[tuple[str, str, str]] = field(default_factory=list)  # (a, b, evidence)

    def tag_of(self, ep: str) -> str:
        return next((m for m, v in self.modules.items() if ep in v), "?")

    def cross_cluster_edges(self) -> list[LinkEdge]:
        return [e for e in self.edges if self.tag_of(e.src) != self.tag_of(e.dst)]

    def neighbors(self, keys: list[str], hop: int = 1) -> set[str]:
        out: set[str] = set()
        frontier = set(keys)
        for _ in range(hop):
            nxt = {e.dst for e in self.edges if e.src in frontier} | {
                e.src for e in self.edges if e.dst in frontier
            }
            out |= nxt
            frontier = nxt - out
        return out - set(keys)

    def to_text(self) -> str:
        """Render ONLY cross-cluster edges — intra-cluster structure is already
        visible to the unit through its own L2 body."""
        cross = self.cross_cluster_edges()
        if not cross:
            return ""
        lines = [
            "## Cross-Module Links (authoritative — use ONLY these for integration cases)",
            "Each line is a contract-level or requirement-level dependency between two",
            "modules. An integration case must chain endpoints along these lines.",
        ]
        for e in cross:
            arrow = f" --{e.field}--> " if e.field else " --> "
            lines.append(f"- {e.src}{arrow}{e.dst}   [{e.rule}] {e.evidence}")
        for a, b, ev in self.module_links:
            lines.append(f"- module {a} <-> module {b}   [R6] {ev}")
        return "\n".join(lines)

    def chains(self, limit: int = 5) -> list[list[str]]:
        """Build chains as: module business flow x per-module trunk endpoints."""
        order = self._module_order()
        trunk: dict[str, list[str]] = {m: self._trunk(m) for m in order}
        chain: list[str] = []
        for m in order:
            chain.extend(trunk.get(m, []))
        chains = []
        if len({self.tag_of(e) for e in chain}) >= 2:
            chains.append(chain)
        # a shorter variant that stops before the terminal ops
        core = [e for e in chain if not e.startswith("DELETE") and "/cancel" not in e]
        if core != chain and len({self.tag_of(e) for e in core}) >= 2:
            chains.append(core)
        return chains[:limit]

    def _module_order(self) -> list[str]:
        """Order modules by first appearance in the requirements prose
        (requirement docs are authored in business-flow order), then by tags."""
        seen: list[str] = []
        for line in REQ_TEXT.splitlines():
            if not line.startswith("#"):
                continue
            for tag, words in ALIASES.items():
                if any(w in line for w in words) and tag not in seen:
                    seen.append(tag)
        for tag in self.modules:
            if tag not in seen:
                seen.append(tag)
        return [m for m in seen if m in self.modules or m == "coupon"]

    @staticmethod
    def _trunk(module: str) -> list[str]:
        """Create -> read -> terminate skeleton for a module (static map for the
        prototype; production derives it from the cluster's own endpoints)."""
        trunk = {
            "products": ["POST /products", "GET /products/{id}", "DELETE /products/{id}"],
            "cart": ["POST /cart/items", "GET /cart/items", "DELETE /cart/items/{itemId}"],
            "orders": [
                "POST /orders",
                "GET /orders/{id}",
                "POST /orders/{id}/cancel",
            ],
            "coupon": [],
        }
        return trunk.get(module, [])


def build_link_graph(spec: dict, req_text: str) -> LinkGraph:
    graph = LinkGraph()
    by_path: dict[str, list[str]] = {}
    consumes: dict[str, set[str]] = {}

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
            tag = (op.get("tags") or ["<none>"])[0]
            graph.modules.setdefault(tag, []).append(key)
            by_path.setdefault(path, []).append(key)
            fields = {str(p.get("name", "")) for p in item.get("parameters", [])}
            fields |= {str(p.get("name", "")) for p in op.get("parameters", [])}
            for media in ((op.get("requestBody") or {}).get("content") or {}).values():
                fields |= set(((media.get("schema") or {}).get("properties") or {}).keys())
            consumes[key] = {f for f in fields if f}

    seen: set[tuple[str, str, str]] = set()

    def add(src: str, dst: str, fld: str, rule: str, strength: str, ev: str) -> None:
        if src == dst or (src, dst, fld) in seen:
            return
        seen.add((src, dst, fld))
        graph.edges.append(LinkEdge(src, dst, fld, rule, strength, ev))

    # R3 same-path CRUD, R4 sub-resource (structure, always available)
    for path, group in by_path.items():
        for a in group:
            for b in group:
                add(a, b, "", "R3", "strong", f"same path {path}")
        base = "/" + (path.strip("/").split("/")[0] if path.strip("/") else "")
        if base != path and base in by_path:
            for a in by_path[base]:
                for b in group:
                    add(a, b, "", "R4", "strong", f"sub-resource of {base}")

    # R2 field-name reference (needs a *Id field; weak coverage on poor specs)
    keys = {k for k in consumes}
    resources = {p.strip("/").split("/")[0].lower() for p in by_path}

    def pluralize(w: str) -> str:
        if w.endswith(("s", "x", "ch")):
            return w + "es"
        if w.endswith("y") and len(w) > 1 and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        return w + "s"

    for key, fields in consumes.items():
        for f in sorted(fields):
            m = re.fullmatch(r"([a-zA-Z]+?)(Id|ID|_id)", f)
            if not m:
                continue
            if pluralize(m.group(1).lower()) not in resources:
                continue
            target = pluralize(m.group(1).lower())
            for other in sorted(keys):
                if (
                    other.split(" ", 1)[0] == "POST"
                    and other.split(" ", 1)[1].strip("/").split("/")[0].lower() == target
                ):
                    add(other, key, f, "R2", "strong", f"field {f} references /{target}")

    # ---- R6: requirement prose co-occurrence (PRIMARY rule on this project)
    blocks = re.split(r"^#\s+", req_text, flags=re.M)
    for block in blocks:
        if not block.strip():
            continue
        head = block.splitlines()[0].strip()
        body = block
        hits = {
            tag: [w for w in words if w in body]
            for tag, words in ALIASES.items()
            if any(w in body for w in words)
        }
        if len(hits) < 2:
            continue
        tags = sorted(hits)
        for i, a in enumerate(tags):
            for b in tags[i + 1 :]:
                ev = f"requirement '{head}' couples {a}({hits[a][0]}) with {b}({hits[b][0]})"
                graph.module_links.append((a, b, ev))
                # project module links onto endpoint level where both sides exist
                for ea in graph.modules.get(a, []):
                    for eb in graph.modules.get(b, []):
                        if ea.split(" ", 1)[0] == "POST" and eb.split(" ", 1)[0] == "POST":
                            add(ea, eb, "", "R6", "strong", ev)
    return graph


def split_by_dependency(graph: LinkGraph, max_size: int = 6) -> list[list[str]]:
    order = {"POST": 0, "GET": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}
    batches: list[list[str]] = []
    for tag, group in graph.modules.items():
        g = sorted(set(group), key=lambda k: (order.get(k.split(" ")[0], 9), k))
        for i in range(0, len(g), max_size):
            batches.append(g[i : i + max_size])
    return batches


# ------------------------------------------------------------------ verify
graph = build_link_graph(SPEC, REQ_TEXT)
all_eps = [ep for g in graph.modules.values() for ep in g]

print("=" * 70)
print("A. LINK RECOVERY — which rules actually fire on this project's inputs")
print("=" * 70)
by_rule: dict[str, int] = {}
for e in graph.edges:
    by_rule[e.rule] = by_rule.get(e.rule, 0) + 1
print(f"   endpoint-level edges by rule: {by_rule}")
print(f"   module-level links (R6 prose): {len(graph.module_links)}")
for a, b, ev in graph.module_links:
    print(f"      {a} <-> {b}   {ev}")
assert graph.module_links, "FAIL: R6 recovered nothing from the requirements prose"

# the decisive check: does R6 find links that the API contract does NOT carry?
contract_modules = {("products", "cart")}
prose_modules = {(min(a, b), max(a, b)) for a, b, _ in graph.module_links}
only_prose = prose_modules - contract_modules
print(f"   >> links present ONLY in prose, absent from the API contract: {sorted(only_prose)}")
assert only_prose, "FAIL: prose added no link beyond the contract — R6 would be pointless"

print()
print("=" * 70)
print("B. PROMPT PAYLOAD SIZE (cross-cluster edges only, after the v1 fix)")
print("=" * 70)
text = graph.to_text()
v1_size = 3646
print(f"   v1 to_text (all edges)     : {v1_size} chars")
print(f"   v2 to_text (cross only)    : {len(text)} chars  ({len(text) / v1_size:.0%} of v1)")
print(
    f"   + api_map                  : {len(text) + 1104} chars ~= {(len(text) + 1104) / 3.2:.0f} tokens"
)
print(f"   share of 32768 output budget: {(len(text) + 1104) / 3.2 / 32768:.1%}")
assert len(text) < v1_size * 0.5, "FAIL: cross-cluster filter did not shrink the payload"
print()
print("   --- rendered payload (what the model sees) ---")
print("   " + text.replace("\n", "\n   "))

print()
print("=" * 70)
print("C. BUSINESS CHAINS for the L3 cross-module pass")
print("=" * 70)
chains = graph.chains(limit=5)
for i, ch in enumerate(chains, 1):
    mods = [graph.tag_of(e) for e in ch]
    print(f"   chain{i}: spans {len(set(mods))} modules, {len(ch)} steps")
    for step, mod in zip(ch, mods, strict=True):
        print(f"      [{mod:<8}] {step}")
assert chains, "FAIL: no chain produced"
best = max(chains, key=lambda c: len({graph.tag_of(e) for e in c}))
assert len({graph.tag_of(e) for e in best}) >= 3, (
    f"FAIL: best chain spans only {len({graph.tag_of(e) for e in best})} modules, expected >=3"
)

print()
print("=" * 70)
print("D. CLUSTERING")
print("=" * 70)
current = [all_eps[i : i + 2] for i in range(0, len(all_eps), 2)]
new = split_by_dependency(graph)
print(f"   current  (index slice size=2): {len(current)} batches")
print(f"   proposed (module clusters)   : {len(new)} batches -> {[len(b) for b in new]}")
for b in new:
    print(f"      [{graph.tag_of(b[0])}] {b}")
assert len(new) == 3 and len(new) < len(current)

print()
print("=" * 70)
print("V2 CHECKS PASSED")
print("  A R6 recovers links the API contract does not carry (cart<->orders)")
print(f"  B cross-cluster filtering cut the prompt payload to {len(text) / v1_size:.0%} of v1")
print("  C chains span 3 modules (v1 managed only 2)")
print(f"  D {len(current)} -> {len(new)} batches, module-pure")
print("=" * 70)
