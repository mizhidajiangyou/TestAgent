"""Global context assembly + lifecycle batching (T12a, fix-plan T12 / plan-k §4.3).

**FROZEN INTERFACE (first builder owns it; plan-k §4.3 contract)** — later
tasks (LINK-S5 graph seeding, L3b) may ADD methods/adapter modules but must
not change these signatures or semantics:

- ``ContextBuilder(endpoints, relations=None, max_chars=8000,
  max_neighbors=4, l1_chars=1200, max_cluster=6)``
- ``build_l0() -> str`` — full-endpoint index for every call. The ONLY
  degradation criterion is the RENDERED CHARACTER COUNT (never an endpoint
  count proxy): detailed compact signatures while they fit ``max_chars``,
  then a module (first path segment) index, then a hard truncation marker.
- ``build_l1(batch: Sequence[str]) -> str`` — 1-hop neighbour summary for
  the batch endpoints (relations first, then same-resource siblings); at
  most ``max_neighbors`` neighbours and ``l1_chars`` characters.
- ``split_clusters() -> list[list[str]]`` — lifecycle clusters replacing
  blind slicing: endpoints sharing a path prefix (R3 same-path multi-method
  / R4 nested parent-child) stay in one cluster, capped at ``max_cluster``
  (oversized clusters split deterministically by method groups, then
  alphabetically).

Pure assembly: no LLM, no I/O, domain-light (reads APIEndpoint fields only).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from testagent.config.models import APIEndpoint

__all__ = ["ContextBuilder"]

_PATH_SEGMENTS = 1


def _minimal_signature(endpoints: Sequence[APIEndpoint]) -> str:
    """Layering-safe fallback renderer (no prompt_builder dependency)."""
    lines: list[str] = []
    for ep in endpoints:
        line = f"- {ep.method} {ep.path}"
        if ep.summary:
            line += f" - {ep.summary}"
        lines.append(line)
    return "\n".join(lines)


class ContextBuilder:
    """Assemble L0/L1 context and lifecycle clusters for one spec."""

    def __init__(
        self,
        endpoints: Sequence[APIEndpoint],
        relations: Sequence[tuple[str, str, str]] = (),
        max_chars: int = 8000,
        max_neighbors: int = 4,
        l1_chars: int = 1200,
        max_cluster: int = 6,
        signature_fn: Callable[[Sequence[APIEndpoint]], str] | None = None,
    ) -> None:
        # Layering (B4.11): pipeline must not import engine.prompt_builder,
        # so the rich-signature renderer is INJECTED by the host (the
        # generator passes ``endpoints_to_rich_signature``). None -> the
        # minimal built-in renderer below.
        self._signature_fn = signature_fn or _minimal_signature
        self._endpoints = list(endpoints)
        self._relations = list(relations)
        self._max_chars = max_chars
        self._max_neighbors = max_neighbors
        self._l1_chars = l1_chars
        self._max_cluster = max_cluster

    # -- L0 --------------------------------------------------------------

    def build_l0(self) -> str:
        """Full-endpoint index; character budget is the only degradation."""
        detailed = self._signature_fn(self._endpoints)
        if len(detailed) <= self._max_chars:
            return detailed
        modules = self._module_index()
        if len(modules) <= self._max_chars:
            return modules
        cut = self._max_chars - len("\n... [L0 truncated]")
        return modules[:cut] + "\n... [L0 truncated]"

    def _module_index(self) -> str:
        """Group endpoints by their first path segment (module view)."""
        groups: dict[str, list[str]] = {}
        for ep in self._endpoints:
            resource = ep.path.strip("/").split("/")[0] or "root"
            groups.setdefault(resource, []).append(f"{ep.method} {ep.path}")
        lines: list[str] = ["[L0 module index - spec too large for detail]"]
        for resource in sorted(groups):
            eps = sorted(groups[resource])
            lines.append(f"- /{resource} ({len(eps)}): {', '.join(eps)}")
        return "\n".join(lines)

    # -- L1 --------------------------------------------------------------

    def build_l1(self, batch: Sequence[str]) -> str:
        """1-hop neighbour summary: related endpoints for the batch."""
        batch_set = set(batch)
        neighbours: list[str] = []
        for src, dst, _kind in self._relations:
            if src in batch_set and dst not in batch_set and dst not in neighbours:
                neighbours.append(dst)
            if dst in batch_set and src not in batch_set and src not in neighbours:
                neighbours.append(src)
        for ep in self._endpoints:
            key = f"{ep.method} {ep.path}"
            if key in batch_set or key in neighbours:
                continue
            resource = ep.path.strip("/").split("/")[0]
            if any(b.split(" ", 1)[-1].strip("/").split("/")[0] == resource for b in batch):
                neighbours.append(key)
        neighbours = neighbours[: self._max_neighbors]
        if not neighbours:
            return ""
        lines = [f"[L1 neighbours] {n}" for n in neighbours]
        text = "\n".join(lines)
        while len(text) > self._l1_chars and len(lines) > 1:
            lines.pop()
            text = "\n".join(lines)
        return text[: self._l1_chars]

    # -- L3a: lifecycle clusters ------------------------------------------

    def split_clusters(self) -> list[list[str]]:
        """Lifecycle clusters (same resource stays together, size-capped)."""
        parent_of: dict[str, str] = {}

        def find(x: str) -> str:
            while parent_of[x] != x:
                parent_of[x] = parent_of[parent_of[x]]
                x = parent_of[x]
            return x

        def union(a: str, b: str) -> None:
            parent_of.setdefault(a, a)
            parent_of.setdefault(b, b)
            ra, rb = find(a), find(b)
            if ra != rb:
                parent_of[rb] = ra

        keys = [f"{ep.method} {ep.path}" for ep in self._endpoints]
        for key in keys:
            parent_of.setdefault(key, key)
        by_path: dict[str, list[str]] = {}
        resource_of: dict[str, str] = {}
        for ep in self._endpoints:
            key = f"{ep.method} {ep.path}"
            path = ep.path
            by_path.setdefault(path, []).append(key)
            resource = re.sub(r"\{.*\}", "", path).strip("/").split("/")[0]
            resource_of[key] = resource
        # R3: same path, multiple methods -> one lifecycle.
        for group in by_path.values():
            for other in group[1:]:
                union(group[0], other)
        # R4: nested child paths share the parent resource.
        by_resource: dict[str, list[str]] = {}
        for key, resource in resource_of.items():
            by_resource.setdefault(resource, []).append(key)
        for group in by_resource.values():
            for other in group[1:]:
                union(group[0], other)

        clusters: dict[str, list[str]] = {}
        for key in keys:
            clusters.setdefault(find(key), []).append(key)

        # Cap cluster size: split deterministically by method, then name.
        result: list[list[str]] = []
        for group in clusters.values():
            group = sorted(group)
            for i in range(0, len(group), self._max_cluster):
                result.append(group[i : i + self._max_cluster])
        result.sort(key=lambda c: c[0])
        return result

    # ------------------------------------------------------------------
    # S5a additions (LINK-S5a, plan-links-v15 §5.5): additive only — the
    # frozen constructor/L0/L1/L3a signatures above are untouched.
    # ------------------------------------------------------------------

    def build_l2(self, contract: Any) -> str:
        """Single-path endpoint skeleton (max 8 lines) + its binding
        contract. Overflow records ``context_shape_limit`` instead of
        silently dropping middle steps (v15 §5.1/§5.5)."""
        from testagent.pipeline.pathplanner import PathContract  # local: avoid cycle

        assert isinstance(contract, PathContract)
        lines = ["[L2 path skeleton]"]
        for endpoint in contract.endpoints:
            lines.append(f"- {endpoint}")
        if len(lines) - 1 > 8:
            return "context_shape_limit: path exceeds 8 endpoint lines; contract not sent unsliced"
        lines.append("[L2 binding contract]")
        for hop in contract.business_hops:
            binding = hop.candidate
            if hop.kind == "DATA_FLOW" and binding and (binding.producer or binding.consumer):
                lines.append(
                    f"- {hop.source} -> {hop.target}: {binding.producer or '?'} -> "
                    f"{binding.consumer or '?'} ({binding.mode})"
                )
            else:
                lines.append(
                    f"- {hop.source} -> {hop.target}: {hop.kind} (observable assertion required)"
                )
        return "\n".join(lines)

    def render_l1_for(self, batch: Sequence[str], relations: Sequence[tuple[str, str, str]]) -> str:
        """L1 with explicit graph relations (additive convenience for S5a
        wiring; delegates to the frozen build_l1)."""
        return self.build_l1(batch)


def assemble_l3b_prompts(planning: dict[str, Any], builder: ContextBuilder) -> list[dict[str, str]]:
    """S5a: one prompt-context bundle per selected path (v15 §5.5).

    Each bundle carries L0 (once per run), the L2 skeleton+contract and the
    path_id. Derived from the SAME immutable planning result (no re-plan).
    """
    bundles: list[dict[str, str]] = []
    l0 = builder.build_l0()
    for contract in planning["selected"]:
        bundles.append(
            {
                "path_id": contract.path_id,
                "l0": l0,
                "l2": builder.build_l2(contract),
                "static_class": contract.static_class,
            }
        )
    return bundles
