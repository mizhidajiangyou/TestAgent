"""R6 prose-relation rules (LINK-S2a, plan-links-v15 §4.2-§4.4).

Pure functions over requirement text: clause splitting, alias matching with
scores, direction templates, and endpoint grounding. No LLM, no spec
mutation, no dict-order dependence — direction comes from character spans
of the alias hits, and three-module or negated clauses abstain with a
recorded reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from testagent.pipeline.module_aliases import (
    STRONG_SCORE,
    WEAK_SCORE,
    merge_aliases,
)

__all__ = [
    "Clause",
    "R6Edge",
    "extract_r6_edges",
    "ground_r6_edge",
    "match_direction",
    "match_modules",
    "split_clauses",
]

_CLAUSE_SPLIT_RE = re.compile(r"[。！？；.!?;\n]+")

_RAW_PATTERNS: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = (
    # kind, ordered regex fragments (with {s}/{t} alias slots)
    (
        "DATA_FLOW",
        ((r"(?P<t>{a})(?:[^。！？；\n.!?;]{0,12}?)(?:根据|基于|使用|通过)(?P<s>{b})", 1),),
    ),
    (
        "PRECONDITION",
        ((r"(?P<t>{a})[^。！？；\n.!?;]{0,12}?必须先[^。！？；\n.!?;]{0,20}?(?P<s>{b})", 1),),
    ),
    ("PRECONDITION", ((r"(?P<s>{b})前(?P<t>{a})", 1),)),
    (
        "PRECONDITION",
        ((r"(?P<t>{a})[^。！？；\n.!?;]{0,12}?必须以(?P<s>{b})为前提", 1),),
    ),
    (
        "SIDE_EFFECT",
        (
            (
                r"(?P<s>{b})[^。！？；\n.!?;]{0,8}?(?:之后|然后|随后|接着)[^。！？；\n.!?;]{0,12}?(?P<t>{a})",
                1,
            ),
        ),
    ),
    (
        "SIDE_EFFECT",
        ((r"(?P<t>{a})[^。！？；\n.!?;]{0,8}?在(?P<s>{b})[^。！？；\n.!?;]{0,8}?之后", 1),),
    ),
)

_NEGATION_RE = re.compile(r"不需要|无需|不依赖|禁止|并非|不能")
_AMBIGUOUS_MODULES = 3


@dataclass(frozen=True)
class Clause:
    """One clause with stable identity and provenance offsets."""

    clause_id: str
    text: str
    doc: str
    offset: int


@dataclass(frozen=True)
class R6Edge:
    """A directed module-level relation from one clause."""

    source: str
    target: str
    kind: str
    clause_id: str
    evidence: str
    score: float
    warnings: tuple[str, ...] = field(default_factory=tuple)


def split_clauses(text: str, doc: str = "requirements") -> list[Clause]:
    """Split on 。！？；.!?; and newlines; keep offsets; commas survive."""
    clauses: list[Clause] = []
    start = 0
    index = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text):
        piece = text[start : m.start()].strip()
        if piece:
            index += 1
            clauses.append(
                Clause(
                    clause_id=f"{doc}-C{index:03d}",
                    text=piece,
                    doc=doc,
                    offset=start + text[start : m.start()].index(piece[0]) if piece else start,
                )
            )
        start = m.end()
    tail = text[start:].strip()
    if tail:
        index += 1
        clauses.append(Clause(clause_id=f"{doc}-C{index:03d}", text=tail, doc=doc, offset=start))
    return clauses


def match_modules(
    clause: Clause,
    aliases: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    min_score: float = 1.0,
) -> dict[str, float]:
    """Score module hits in one clause.

    Returns ``module -> accumulated score`` for modules reaching
    ``min_score``. English keys match on word boundaries case-insensitively
    (singular+plural), Chinese on substrings. The longest alias wins a text
    span; equal-length cross-module conflicts are NOT recalled.
    """
    table = merge_aliases(aliases)
    candidates: list[tuple[int, int, str, float]] = []  # (start, -length, module, score)
    lowered = clause.text.lower()
    for module, (strong, weak) in table.items():
        # English canonical key: word-boundary match with plural variants.
        for variant in _english_variants(module):
            for m in re.finditer(rf"\b{re.escape(variant)}\b", lowered):
                candidates.append((m.start(), -(m.end() - m.start()), module, STRONG_SCORE))
        for word in strong:
            for m in re.finditer(re.escape(word), clause.text):
                candidates.append((m.start(), -(len(word)), module, STRONG_SCORE))
        for word in weak:
            for m in re.finditer(re.escape(word), clause.text):
                candidates.append((m.start(), -(len(word)), module, WEAK_SCORE))
    # Longest alias wins each overlapping span; equal-length cross-module
    # conflicts at the same span are dropped entirely.
    by_start: dict[int, list[tuple[int, int, str, float]]] = {}
    for start, neg_len, module, score in candidates:
        by_start.setdefault(start, []).append((start, neg_len, module, score))
    scores: dict[str, float] = {}
    for _, group in sorted(by_start.items()):
        best = min(group, key=lambda item: (item[1], item[0]))
        same_span = [g for g in group if g[1] == best[1]]
        modules_at_span = {g[2] for g in same_span}
        if len(modules_at_span) > 1:
            continue  # ambiguous alias: not recalled
        scores[best[2]] = scores.get(best[2], 0.0) + best[3]
    return {mod: sc for mod, sc in scores.items() if sc >= min_score}


def _english_variants(module: str) -> tuple[str, ...]:
    if not module.isascii():
        return (module,)
    singular = module
    variants = {singular}
    if singular.endswith("ies"):
        variants.add(singular[:-3] + "y")
    elif singular.endswith("s") and not singular.endswith("ss"):
        variants.add(singular[:-1])
    else:
        variants.add(singular + "s")
    return tuple(variants)


def match_direction(
    clause: Clause,
    source: str,
    target: str,
    aliases: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> tuple[str, str] | None:
    """Try the finite direction templates; return ``(source, target)`` or
    None. Overlapping same-kind shorter templates do not double-fire."""
    table = merge_aliases(aliases)
    source_spans = _alias_spans(clause.text, source, table)
    target_spans = _alias_spans(clause.text, target, table)
    if not source_spans or not target_spans:
        return None
    # Templates match ALIAS WORDS (the actual text spans), not module keys.
    t_alt = _alias_alternation(target, table)
    s_alt = _alias_alternation(source, table)
    for _kind, patterns in _RAW_PATTERNS:
        for pattern, _weight in patterns:
            regex = _render_pattern(pattern, t_alt, s_alt)
            for m in re.finditer(regex, clause.text):
                # The relation expression must not cross another module hit.
                matched = m.group(0)
                if _NEGATION_RE.search(matched):
                    return None  # unsupported negation: abstain
                if _crosses_other_module(m.start(), m.end(), clause, table, {source, target}):
                    return None
                return (source, target)
    return None


def _alias_alternation(
    module: str, table: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
) -> str:
    """Regex alternation of one module's alias words (EN + ZH), longest
    first so the longest alias wins a span."""
    words = {module}
    strong, weak = table.get(module, ((), ()))
    for variant in _english_variants(module):
        words.add(variant)
    words.update(strong)
    words.update(weak)
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


def _render_pattern(pattern: str, target_alt: str, source_alt: str) -> str:
    """Substitute the {a}/{b} alternation slots; inputs are pre-escaped
    alternations, so no further escaping here (regex quantifier braces in
    the template must survive)."""
    return pattern.replace("{a}", target_alt).replace("{b}", source_alt)


def _alias_spans(
    text: str, module: str, table: dict[str, tuple[tuple[str, ...], tuple[str, ...]]]
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    strong, weak = table.get(module, ((), ()))
    for variant in _english_variants(module):
        spans.extend(
            (m.start(), m.end())
            for m in re.finditer(rf"\b{re.escape(variant)}\b", text, re.IGNORECASE)
        )
    for word in (*strong, *weak):
        spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(word), text))
    return spans


def _crosses_other_module(
    start: int,
    end: int,
    clause: Clause,
    table: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    allowed: set[str],
) -> bool:
    for module in table:
        if module in allowed:
            continue
        for span_start, span_end in _alias_spans(clause.text, module, table):
            if span_start < end and start < span_end:
                return True
    return False


def extract_r6_edges(
    text: str,
    doc: str = "requirements",
    aliases: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
    min_score: float = 1.0,
) -> tuple[list[R6Edge], list[dict[str, str]]]:
    """Full R6 extraction over a document.

    Returns ``(edges, abstentions)``. Abstention reasons: unknown alias
    modules, fewer/more than two modules, negation, ambiguous direction,
    no matching template.
    """
    table = merge_aliases(aliases)
    edges: list[R6Edge] = []
    abstained: list[dict[str, str]] = []
    for clause in split_clauses(text, doc):
        hits = match_modules(clause, aliases, min_score)
        modules = sorted(hits)
        for module in modules:
            if module not in table:
                abstained.append(
                    {"clause": clause.clause_id, "reason": "unused_alias", "module": module}
                )
        modules = [m for m in modules if m in table]
        if len(modules) < 2:
            if modules:
                abstained.append({"clause": clause.clause_id, "reason": "single_module"})
            continue
        if len(modules) >= _AMBIGUOUS_MODULES:
            abstained.append({"clause": clause.clause_id, "reason": "r6_ambiguous_direction"})
            continue
        first, second = modules[0], modules[1]
        if _NEGATION_RE.search(clause.text):
            abstained.append({"clause": clause.clause_id, "reason": "unsupported_negation"})
            continue
        # Direction is DERIVED from the templates: try both module
        # assignments; alphabetical order is only the enumeration seed.
        direction: tuple[str, str] | None = None
        for cand in ((first, second), (second, first)):
            direction = match_direction(clause, cand[0], cand[1], aliases)
            if direction is not None:
                break
        if direction is None:
            abstained.append({"clause": clause.clause_id, "reason": "no_template_match"})
            continue
        source, target = direction
        edges.append(
            R6Edge(
                source=source,
                target=target,
                kind=_matched_kind(clause, direction, aliases),
                clause_id=clause.clause_id,
                evidence=clause.text,
                score=hits[source] + hits[target],
            )
        )
    return edges, abstained


def _matched_kind(
    clause: Clause,
    direction: tuple[str, str],
    aliases: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] | None,
) -> str:
    """Re-derive which kind matched (first pattern that fits the spans)."""
    source, target = direction
    table = merge_aliases(aliases)
    t_alt = _alias_alternation(target, table)
    s_alt = _alias_alternation(source, table)
    for kind, patterns in _RAW_PATTERNS:
        for pattern, _weight in patterns:
            regex = _render_pattern(pattern, t_alt, s_alt)
            if re.search(regex, clause.text):
                return str(kind)
    return "DATA_FLOW"


@dataclass(frozen=True)
class _GroundedPair:
    source: str
    target: str
    score: int


_WRITE = {"POST", "PUT", "PATCH", "DELETE"}
_TERMINAL = {"DELETE"}


def ground_r6_edge(
    edge: R6Edge,
    endpoint_modules: dict[str, list[Any_endpoint]],
    max_pairs: int = 2,
) -> list[tuple[str, str]]:
    """Map a module edge to endpoint pairs (§4.4 role-preference table).

    ``endpoint_modules``: module -> list of APIEndpoint-like objects. The
    module direction is preserved; pairs are scored by the fixed table and
    the top ``max_pairs`` (deterministic tie-break by full_path) returned.
    """
    sources = endpoint_modules.get(edge.source, [])
    targets = endpoint_modules.get(edge.target, [])
    scored: list[_GroundedPair] = []
    for src in sources:
        for dst in targets:
            src_write = str(getattr(src, "method", "")).upper() in _WRITE
            dst_write = str(getattr(dst, "method", "")).upper() in _WRITE
            if edge.kind == "DATA_FLOW":
                if not (src_write and dst_write) and not (src_write and not dst_write):
                    continue
                score = 10 if (src_write and dst_write) else 7
            elif edge.kind == "PRECONDITION":
                if not ((src_write and dst_write) or (not src_write and dst_write)):
                    continue
                score = 10 if (src_write and dst_write) else 7
            else:  # SIDE_EFFECT
                if not ((src_write and not dst_write) or (src_write and dst_write)):
                    continue
                score = 10 if (src_write and not dst_write) else 7
            if not re.search(r"\{\w+\}", str(getattr(src, "path", ""))) and src_write:
                score += 3
            if dst_write:
                score += 2
            score -= len(re.findall(r"\{\w+\}", str(getattr(src, "path", ""))))
            score -= len(re.findall(r"\{\w+\}", str(getattr(dst, "path", ""))))
            if str(getattr(src, "method", "")).upper() in _TERMINAL:
                score -= 4
            if str(getattr(dst, "method", "")).upper() in _TERMINAL:
                score -= 4
            scored.append(
                _GroundedPair(
                    f"{getattr(src, 'method', '')} {getattr(src, 'path', '')}",
                    f"{getattr(dst, 'method', '')} {getattr(dst, 'path', '')}",
                    score,
                )
            )
    scored.sort(key=lambda p: (-p.score, p.source, p.target))
    return [(p.source, p.target) for p in scored[:max_pairs]]


class Any_endpoint:  # noqa: N801 - structural typing placeholder name used in docs
    """Duck-typed endpoint (method/path attributes); never instantiated."""
