"""Requirement -> endpoint binding (T6, fix-plan RC-2).

Deterministic binding chain, strictest source wins:

1. ``explicit`` — the requirement document annotates the binding itself
   (e.g. ``"@ POST /auth/login"`` or any ``METHOD /path`` mention);
2. ``keyword`` — schema/field/verb matching: the requirement text mentions a
   body/parameter field declared by an endpoint, or a CRUD verb whose noun
   matches the endpoint path;
3. ``llm_suggestion`` — provided by the caller as CANDIDATES ONLY. LLM
   output can never be the sole range authority (the model must not decide
   its own input range).

A requirement whose binding ends up EMPTY produces an explicit input gap
(fed to the T4 findings stream) instead of silently binding every endpoint.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from testagent.pipeline.consistency import (
    ConsistencyKind,
    Finding,
    RequirementRef,
    extract_endpoint_mentions,
)
from testagent.pipeline.obligations import BindingBasis

__all__ = ["Binding", "bind_requirements", "extract_explicit_bindings"]

_EXPLICIT_RE = re.compile(
    r"@\s*((?P<method>GET|POST|PUT|PATCH|DELETE)\s+)?(?P<path>/[A-Za-z0-9_{}.\-/.]*)", re.IGNORECASE
)

_CRUD_VERBS = {
    "create": ("POST",),
    "add": ("POST",),
    "register": ("POST",),
    "list": ("GET",),
    "get": ("GET",),
    "read": ("GET",),
    "query": ("GET",),
    "update": ("PUT", "PATCH"),
    "modify": ("PUT", "PATCH"),
    "delete": ("DELETE",),
    "remove": ("DELETE",),
    "login": ("POST",),
}

_NOUN_RE = re.compile(
    r"\b(create|add|register|list|get|read|query|update|modify|delete|remove|login)s?\b"
    r"\s+(?:the\s+)?(?:an?\s+)?(?P<noun>[a-z][a-z_]{2,}?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Binding:
    """Result of the binding chain for one requirement."""

    requirement_id: str
    endpoints: tuple[str, ...]
    basis: BindingBasis
    gap: Finding | None = None


def extract_explicit_bindings(text: str) -> list[str]:
    """``@ METHOD /path`` (method optional) annotations in requirement text.

    A bare ``@ /path`` binds every method the spec declares for that path.
    """
    found: list[str] = []
    for m in _EXPLICIT_RE.finditer(text):
        method = (m.group("method") or "").upper()
        path = m.group("path")
        found.append(f"{method} {path}".strip())
    return found


def _spec_index(endpoints: Sequence[Any]) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """(path -> declared methods, endpoint -> known request fields)."""
    index: dict[str, list[str]] = {}
    fields_by_endpoint: dict[str, set[str]] = {}
    for ep in endpoints:
        path = str(getattr(ep, "path", ""))
        method = str(getattr(ep, "method", "")).upper()
        index.setdefault(path, []).append(method)
        fields: set[str] = set()
        for p in getattr(ep, "parameters", None) or []:
            if isinstance(p, dict) and isinstance(p.get("name"), str):
                fields.add(p["name"].lower())
        body = getattr(ep, "request_body", None)
        if isinstance(body, dict):
            props = (body.get("schema") or {}).get("properties", {})
            if isinstance(props, dict):
                fields.update(str(k).lower() for k in props)
        fields_by_endpoint[f"{method} {path}"] = fields
    return index, fields_by_endpoint


def bind_requirements(
    requirements: Sequence[RequirementRef],
    endpoints: Sequence[Any],
    llm_suggestions: dict[str, Sequence[str]] | None = None,
) -> list[Binding]:
    """Run the deterministic binding chain for every requirement.

    ``llm_suggestions`` maps requirement id -> candidate endpoint strings;
    they are only used when explicit + keyword matching yield nothing, and
    the resulting binding is marked ``llm_suggestion`` (candidate-grade).
    """
    path_index, fields_by_endpoint = _spec_index(endpoints)
    suggestions = llm_suggestions or {}
    bindings: list[Binding] = []

    for req in requirements:
        # Source 1: explicit annotations (highest authority). A bare
        # "@ /path" expands to every method the spec declares for it.
        explicit: list[str] = []
        for m in extract_explicit_bindings(req.text):
            if " " not in m:
                for method in path_index.get(m, []):
                    explicit.append(f"{method} {m}")
            elif _exists(m, path_index):
                explicit.append(m)
        if explicit:
            bindings.append(
                Binding(
                    requirement_id=req.id,
                    endpoints=tuple(dict.fromkeys(explicit)),
                    basis=BindingBasis.EXPLICIT,
                )
            )
            continue

        # Plain METHOD /path mentions count as explicit declarations too.
        mentioned = [m for m in extract_endpoint_mentions(req.text) if _exists(m, path_index)]
        if mentioned:
            bindings.append(
                Binding(
                    requirement_id=req.id,
                    endpoints=tuple(dict.fromkeys(mentioned)),
                    basis=BindingBasis.EXPLICIT,
                )
            )
            continue

        # Source 2: keyword matching (fields, then verb+noun heuristics).
        keyword: list[str] = []
        lowered = req.text.lower()
        for endpoint, fields in fields_by_endpoint.items():
            if fields and any(f in lowered for f in fields):
                keyword.append(endpoint)
        if not keyword:
            for nm in _NOUN_RE.finditer(req.text):
                verb = nm.group(0).split()[0].lower()
                noun = nm.group("noun").lower()
                for path, methods in path_index.items():
                    resource = path.strip("/").split("/")[0].replace("{id}", "")
                    if noun.rstrip("s") == resource.rstrip("s"):
                        keyword.extend(
                            f"{method} {path}"
                            for method in methods
                            if method in _CRUD_VERBS.get(verb, ())
                        )
        if keyword:
            bindings.append(
                Binding(
                    requirement_id=req.id,
                    endpoints=tuple(dict.fromkeys(keyword)),
                    basis=BindingBasis.KEYWORD,
                )
            )
            continue

        # Source 3: LLM suggestions are candidates only.
        sugg = [s for s in suggestions.get(req.id, []) if _exists(s, path_index)]
        if sugg:
            bindings.append(
                Binding(
                    requirement_id=req.id,
                    endpoints=tuple(dict.fromkeys(sugg)),
                    basis=BindingBasis.LLM_SUGGESTION,
                )
            )
            continue

        # Empty binding: explicit input gap — never bind everything as filler.
        bindings.append(
            Binding(
                requirement_id=req.id,
                endpoints=(),
                basis=BindingBasis.NONE,
                gap=Finding(
                    kind=ConsistencyKind.SPEC_GAP,
                    subject=f"binding:{req.id}",
                    detail=(
                        f"requirement {req.id} binds to no endpoint; "
                        "generate nothing for it or report the input gap"
                    ),
                    requirement_ref=req.id,
                ),
            )
        )
    return bindings


def _exists(mention: str, path_index: dict[str, list[str]]) -> bool:
    method, _, path = mention.partition(" ")
    if not path:
        return False
    if path not in path_index:
        return False
    return not method or method in path_index[path]
