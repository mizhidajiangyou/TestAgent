"""Deterministic pre-generation consistency checking (T4, fix-plan §3.3).

Cross-checks requirements against the parsed spec (and the built-in error
contract) BEFORE generation, classifying every divergence into explicit
kinds so a downstream policy (T9 ``CONFLICT_POLICY``) can adjudicate —
the checker itself never decides who wins.

All functions are pure: text/dicts in, findings/strings out. No I/O, no
LLM, no imports from ``engine.prompt_builder`` (pipeline arch gate) — the
structured default error contract lives here as data.

Checks (fix-plan §3.3, all deterministic):
1. requirement-mentioned **endpoints** missing from the spec (spec_gap);
2. requirement-mentioned **fields** absent from every body/param schema (spec_gap);
3. **status-code** disagreement for the same trigger between spec,
   requirement and the built-in error contract (spec_req_conflict /
   contract_only);
4. requirement **rules with no spec support** (bounds, lockout durations —
   unsupported_requirement).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "DEFAULT_ERROR_CONTRACT",
    "ConflictPolicy",
    "ConsistencyKind",
    "Finding",
    "RequirementRef",
    "adjudicate_conflicts",
    "classify_findings",
    "extract_bound_rules",
    "extract_field_mentions",
    "extract_status_expectations",
    "find_endpoint_gaps",
    "find_field_gaps",
    "find_status_conflicts",
    "find_unsupported_rules",
    "render_authoritative_table",
    "render_gap_report",
]


class ConsistencyKind(StrEnum):
    """Finding classification (fix-plan §3.3 — classify first, then policy)."""

    CONSISTENT = "consistent"
    SPEC_GAP = "spec_gap"
    SPEC_REQ_CONFLICT = "spec_req_conflict"
    CONTRACT_ONLY = "contract_only"
    UNSUPPORTED_REQUIREMENT = "unsupported_requirement"


class ConflictPolicy(StrEnum):
    """Adjudication policy for SPEC_REQ_CONFLICT findings (T9 wires settings)."""

    STRICT = "strict"
    SPEC_FIRST = "spec_first"
    REQUIREMENT_FIRST = "requirement_first"


@dataclass(frozen=True)
class RequirementRef:
    """One requirement's identity and raw text (extraction input)."""

    id: str
    text: str


@dataclass(frozen=True)
class Finding:
    """One classified divergence between requirement / spec / contract."""

    kind: ConsistencyKind
    subject: str
    detail: str
    requirement_ref: str = ""
    endpoint_ref: str = ""
    # Status-conflict findings carry the structured sides so policy
    # adjudication never has to parse the subject/detail strings.
    spec_value: str = ""
    requirement_value: str = ""


@dataclass(frozen=True)
class _ContractRule:
    """Built-in error-contract rule: trigger wording -> default status."""

    trigger: str
    status: str
    patterns: tuple[str, ...]


#: Structured representation of the built-in error contract. T9 demotes this
#: to a fallback: it applies only where neither spec nor requirement speaks.
DEFAULT_ERROR_CONTRACT: tuple[_ContractRule, ...] = (
    _ContractRule(
        "duplicate_email",
        "409",
        (r"duplicate[ -]?email", r"重复邮箱", r"已存在的邮箱", r"邮箱已被(注册|使用)"),
    ),
    _ContractRule(
        "unauthorized",
        "401",
        (r"unauthorized", r"未(认证|授权|登录)", r"无效(的)?(令牌|token)", r"缺少(认证|令牌)"),
    ),
    _ContractRule("forbidden", "403", (r"forbidden", r"无(权限|权访问)", r"权限不足", r"越权")),
    _ContractRule("not_found", "404", (r"not[ -]?found", r"不存在", r"找不到")),
    _ContractRule(
        "validation_error",
        "400",
        (
            r"(参数|字段)?(校验|验证)?(错误|失败|非法)",
            r"invalid (request|field|parameter)",
            r"validation",
        ),
    ),
)

_ENDPOINT_MENTION_RE = re.compile(
    r"\b(?P<method>GET|POST|PUT|PATCH|DELETE)\s+(?P<path>/[A-Za-z0-9_{}.\-/.]*)",
    re.IGNORECASE,
)
_BARE_PATH_RE = re.compile(r"(?<![\w])(/[a-z][a-zA-Z0-9_\-{}.]*)")
_FIELD_MENTION_RE = re.compile(r"`([a-z_][a-zA-Z0-9_]*)`")
_STATUS_CODE_RE = re.compile(r"\b([1-5]\d{2})\b")
_MIN_BOUND_RE = re.compile(
    r"(?P<field>[a-zA-Z0-9_\u4e00-\u9fff]+?)\s*"
    r"(?:最小|至少|下限|min(?:imum)?|>=?)\s*(?P<value>\d+)"
)
_MAX_BOUND_RE = re.compile(
    r"(?P<field>[a-zA-Z0-9_\u4e00-\u9fff]+?)\s*"
    r"(?:最大|最多|上限|不超过|max(?:imum)?|<=?)\s*(?P<value>\d+)"
)
_DURATION_RE = re.compile(
    r"(?P<field>[a-zA-Z0-9_\u4e00-\u9fff]*?(?:锁定|lockout|lock)[a-zA-Z0-9_\u4e00-\u9fff]*?)"
    r"\s*(?:时长|duration|time)?\s*(?P<value>\d+)\s*(?P<unit>分钟|秒|小时|min(?:ute)?s?|seconds?|hours?)",
    re.IGNORECASE,
)


def extract_endpoint_mentions(text: str) -> list[str]:
    """Return normalized ``METHOD /path`` mentions (upper-cased method)."""
    found = [
        f"{m.group('method').upper()} {m.group('path')}"
        for m in _ENDPOINT_MENTION_RE.finditer(text)
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for f in found:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def extract_field_mentions(text: str) -> list[str]:
    """Return backtick-quoted field-like identifiers mentioned in text."""
    seen: set[str] = set()
    unique: list[str] = []
    for m in _FIELD_MENTION_RE.finditer(text):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def extract_status_expectations(text: str) -> list[tuple[str, str]]:
    """Extract ``(trigger, status)`` pairs: a contract trigger keyword with a
    status code asserted in the same sentence-ish window (up to 30 chars)."""
    expectations: list[tuple[str, str]] = []
    for rule in DEFAULT_ERROR_CONTRACT:
        for pattern in rule.patterns:
            for m in re.finditer(pattern, text, re.IGNORECASE):
                window = text[m.start() : m.end() + 30]
                codes = _STATUS_CODE_RE.findall(window)
                if codes:
                    expectations.append((rule.trigger, codes[0]))
    seen: set[tuple[str, str]] = set()
    unique: list[tuple[str, str]] = []
    for pair in expectations:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def extract_bound_rules(text: str) -> list[tuple[str, str, str]]:
    """Extract ``(field, rule_kind, value)`` bound rules from requirement text.

    ``rule_kind`` is one of ``min`` / ``max`` / ``duration``. Field names are
    normalized to lower case for ASCII identifiers.
    """
    rules: list[tuple[str, str, str]] = []
    for m in _MIN_BOUND_RE.finditer(text):
        rules.append((m.group("field").strip().lower(), "min", m.group("value")))
    for m in _MAX_BOUND_RE.finditer(text):
        rules.append((m.group("field").strip().lower(), "max", m.group("value")))
    for m in _DURATION_RE.finditer(text):
        field = m.group("field").strip().lower() or "lockout"
        rules.append((field, "duration", f"{m.group('value')} {m.group('unit').lower()}"))
    return rules


def _spec_schema_fields(endpoints: Iterable[Any]) -> set[str]:
    """Union of body property names and parameter names across endpoints."""
    fields: set[str] = set()
    for ep in endpoints:
        for p in getattr(ep, "parameters", None) or []:
            if isinstance(p, dict):
                name = p.get("name")
                if isinstance(name, str):
                    fields.add(name)
        body = getattr(ep, "request_body", None)
        if isinstance(body, dict):
            props = (body.get("schema") or {}).get("properties", {})
            if isinstance(props, dict):
                fields.update(k for k in props if isinstance(k, str))
    return fields


def _spec_endpoint_set(endpoints: Iterable[Any]) -> set[str]:
    return {f"{ep.method.upper()} {ep.path}" for ep in endpoints if getattr(ep, "method", None)}


def find_endpoint_gaps(
    requirements: Sequence[RequirementRef], endpoints: Sequence[Any]
) -> list[Finding]:
    """Check 1: requirement-mentioned endpoints missing from the spec."""
    spec_endpoints = _spec_endpoint_set(endpoints)
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for req in requirements:
        for mention in extract_endpoint_mentions(req.text):
            _method, _, path = mention.partition(" ")
            exists = mention in spec_endpoints or any(
                e == mention or (path and e.endswith(f" {path}")) for e in spec_endpoints
            )
            if exists:
                continue
            key = (path, req.id)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                Finding(
                    kind=ConsistencyKind.SPEC_GAP,
                    subject=path,
                    detail=f"requirement {req.id} mentions {mention} but the spec has no such endpoint",
                    requirement_ref=req.id,
                )
            )
    return findings


def find_field_gaps(
    requirements: Sequence[RequirementRef], endpoints: Sequence[Any]
) -> list[Finding]:
    """Check 2: requirement-mentioned fields absent from every schema."""
    spec_fields = _spec_schema_fields(endpoints)
    findings: list[Finding] = []
    seen: set[tuple[str, str]] = set()
    for req in requirements:
        for field in extract_field_mentions(req.text):
            if field in spec_fields:
                continue
            key = (field, req.id)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                Finding(
                    kind=ConsistencyKind.SPEC_GAP,
                    subject=field,
                    detail=(
                        f"requirement {req.id} mentions field `{field}` "
                        "but no endpoint schema declares it"
                    ),
                    requirement_ref=req.id,
                )
            )
    return findings


def _trigger_rule(trigger: str) -> _ContractRule | None:
    for rule in DEFAULT_ERROR_CONTRACT:
        if rule.trigger == trigger:
            return rule
    return None


def find_status_conflicts(
    requirements: Sequence[RequirementRef],
    endpoints: Sequence[Any],
    error_contract: tuple[_ContractRule, ...] = DEFAULT_ERROR_CONTRACT,
) -> list[Finding]:
    """Check 3: status-code disagreement for the same trigger.

    Codes are collected from three sources — the spec's declared response
    codes on endpoints mentioned by the requirement, the requirement text
    itself, and the built-in contract. Two or more distinct codes for one
    trigger yield ``spec_req_conflict``; a trigger asserted only by the
    contract yields ``contract_only``.
    """
    findings: list[Finding] = []
    spec_endpoints = list(endpoints)
    for req in requirements:
        mentions = extract_endpoint_mentions(req.text)
        expectations = extract_status_expectations(req.text)
        for trigger, req_code in expectations:
            rule = next((r for r in error_contract if r.trigger == trigger), None)
            contract_code = rule.status if rule else ""
            spec_codes: set[str] = set()
            endpoint_ref = ""
            if mentions:
                for mention in mentions:
                    method, _, path = mention.partition(" ")
                    for ep in spec_endpoints:
                        if (
                            getattr(ep, "method", "").upper() == method
                            and getattr(ep, "path", "") == path
                        ):
                            spec_codes.update(
                                str(c) for c in (getattr(ep, "responses", None) or [])
                            )
                            endpoint_ref = f"{method} {path}"
            codes: dict[str, str] = {}
            if contract_code:
                codes["contract"] = contract_code
            codes["requirement"] = req_code
            # Only ERROR-class codes (4xx/5xx) participate on the spec side:
            # 2xx success codes on the same endpoint are not candidates for an
            # error-trigger disagreement (duplicate_email(400 vs 409), never
            # (201 vs 400 vs 409)).
            error_codes = sorted(c for c in spec_codes if c[:1] in {"4", "5"})
            for i, sc in enumerate(error_codes):
                codes[f"spec:{endpoint_ref or 'declared'}" if i == 0 else f"spec#{i}"] = sc
            distinct = set(codes.values())
            if len(distinct) > 1:
                subject = f"{trigger}({' vs '.join(sorted(distinct))})"
                findings.append(
                    Finding(
                        kind=ConsistencyKind.SPEC_REQ_CONFLICT,
                        subject=subject,
                        detail=f"sources disagree on the status code: {codes}",
                        requirement_ref=req.id,
                        endpoint_ref=endpoint_ref,
                        spec_value=error_codes[0] if error_codes else "",
                        requirement_value=req_code,
                    )
                )
            elif distinct and distinct == {contract_code} and contract_code and not spec_codes:
                findings.append(
                    Finding(
                        kind=ConsistencyKind.CONTRACT_ONLY,
                        subject=trigger,
                        detail=f"only the built-in contract defines this trigger (status {contract_code})",
                        requirement_ref=req.id,
                    )
                )
    return findings


def _declared_bound(field: str, kind: str, endpoints: Sequence[Any]) -> tuple[bool, str | None]:
    """Is a min/max rule for ``field`` supported by any spec schema?"""
    for ep in endpoints:
        pairs: list[tuple[str, dict[str, Any]]] = []
        for p in getattr(ep, "parameters", None) or []:
            if isinstance(p, dict) and isinstance(p.get("name"), str):
                schema = p.get("schema")
                schema = schema if isinstance(schema, dict) else {}
                pairs.append((str(p["name"]), schema))
        body = getattr(ep, "request_body", None)
        if isinstance(body, dict):
            props = (body.get("schema") or {}).get("properties", {})
            if isinstance(props, dict):
                pairs.extend((k, v) for k, v in props.items() if isinstance(v, dict))
        for name, schema in pairs:
            if name.lower() != field:
                continue
            key = "minimum" if kind == "min" else "maximum"
            if schema.get(key) is not None:
                return True, str(schema[key])
    return False, None


def find_unsupported_rules(
    requirements: Sequence[RequirementRef], endpoints: Sequence[Any]
) -> list[Finding]:
    """Check 4: requirement rules (bounds, lockout durations) without spec support."""
    findings: list[Finding] = []
    seen: set[tuple[str, str, str]] = set()
    for req in requirements:
        for field, kind, value in extract_bound_rules(req.text):
            if kind == "duration":
                key = (field, kind, value)
                if key not in seen:
                    seen.add(key)
                    findings.append(
                        Finding(
                            kind=ConsistencyKind.UNSUPPORTED_REQUIREMENT,
                            subject=f"{field or 'lockout'} {value}",
                            detail=(
                                f"requirement {req.id} asserts a {kind} rule "
                                f"({field or 'lockout'} {value}) with no spec support"
                            ),
                            requirement_ref=req.id,
                        )
                    )
                continue
            supported, declared = _declared_bound(field, kind, endpoints)
            if supported:
                continue
            key = (field, kind, value)
            if key in seen:
                continue
            seen.add(key)
            label = "上限" if kind == "max" else "下限"
            subject = f"{field} {label}"
            findings.append(
                Finding(
                    kind=ConsistencyKind.UNSUPPORTED_REQUIREMENT,
                    subject=subject,
                    detail=(
                        f"requirement {req.id} asserts {field} {kind}={value} "
                        "but no schema declares it"
                        + (f" (spec declares {declared})" if declared else "")
                    ),
                    requirement_ref=req.id,
                )
            )
    return findings


def classify_findings(
    requirements: Sequence[RequirementRef],
    endpoints: Sequence[Any],
    error_contract: tuple[_ContractRule, ...] = DEFAULT_ERROR_CONTRACT,
) -> list[Finding]:
    """Run all four deterministic checks and return the combined findings."""
    findings: list[Finding] = []
    findings.extend(find_endpoint_gaps(requirements, endpoints))
    findings.extend(find_field_gaps(requirements, endpoints))
    findings.extend(find_status_conflicts(requirements, endpoints, error_contract))
    findings.extend(find_unsupported_rules(requirements, endpoints))
    return findings


def adjudicate_conflicts(findings: Sequence[Finding], policy: ConflictPolicy) -> dict[str, str]:
    """Policy adjudication for SPEC_REQ_CONFLICT subjects (T9 contract).

    Returns ``subject -> decision`` for every conflict. ``strict`` never
    picks a side: the decision is ``conflict_unresolved:<subject>``.
    Adjudication reads the structured ``spec_value`` / ``requirement_value``
    sides of the finding — never parses subject strings.
    """
    decisions: dict[str, str] = {}
    for f in findings:
        if f.kind is not ConsistencyKind.SPEC_REQ_CONFLICT:
            continue
        if policy is ConflictPolicy.STRICT:
            decisions[f.subject] = f"conflict_unresolved:{f.subject}"
        elif policy is ConflictPolicy.SPEC_FIRST:
            decisions[f.subject] = f"spec_wins:{f.spec_value or 'unspecified'}"
        else:
            decisions[f.subject] = f"requirement_wins:{f.requirement_value or 'unspecified'}"
    return decisions


def render_gap_report(findings: Sequence[Finding]) -> str:
    """Render the gap report (grouped by kind, stable order)."""
    if not findings:
        return "No consistency findings: requirements and spec agree.\n"
    lines: list[str] = ["# Consistency gap report", ""]
    for kind in ConsistencyKind:
        group = [f for f in findings if f.kind is kind]
        if not group:
            continue
        lines.append(f"## {kind.value} ({len(group)})")
        for f in group:
            ref = f" [req={f.requirement_ref}]" if f.requirement_ref else ""
            lines.append(f"- {f.subject}{ref}: {f.detail}")
        lines.append("")
    return "\n".join(lines)


def render_authoritative_table(findings: Sequence[Finding], policy: ConflictPolicy) -> str:
    """Render the run's authoritative value table (generation AND review
    share this exact table; T9 injects it into prompts)."""
    decisions = adjudicate_conflicts(findings, policy)
    lines: list[str] = [
        "# Authoritative value table",
        "",
        f"Policy: {policy.value}",
        "",
        "| subject | kind | decision |",
        "|---|---|---|",
    ]
    conflict_subjects = {f.subject for f in findings if f.kind is ConsistencyKind.SPEC_REQ_CONFLICT}
    for f in findings:
        if f.kind is ConsistencyKind.CONSISTENT:
            continue
        if f.subject in conflict_subjects:
            decision = decisions.get(f.subject, "conflict_unresolved")
        elif f.kind is ConsistencyKind.SPEC_GAP:
            decision = "allowed with spec_support=missing (gap reported)"
        elif f.kind is ConsistencyKind.CONTRACT_ONLY:
            decision = "built-in contract applies as fallback"
        else:
            decision = "no spec support - mark case as out_of_spec"
        lines.append(f"| {f.subject} | {f.kind.value} | {decision} |")
    lines.append("")
    return "\n".join(lines)
