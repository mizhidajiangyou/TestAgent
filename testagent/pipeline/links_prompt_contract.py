"""LINK-S6a materials: prompt contract constants and JSON-schema additions
(plan-links-v15 §6.1 / §7.2).

Single source of truth for the L3b / phase prompt contract text; the
tasks/testcase schema additions derive from ``LINKS_CASE_SCHEMA_ADDITIONS``.
Preparation only - wiring happens in S6b (gated by FH2.4).
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "BINDS_V2_CONTRACT",
    "LINKS_CASE_SCHEMA_ADDITIONS",
    "PATH_CONTRACT_INSTRUCTION",
    "render_schema_additions_json",
]

BINDS_V2_CONTRACT = """BINDS CONTRACT (v2):
- Optional top-level `binds` object. Key = placeholder identifier without
  angle brackets, matching [A-Z][A-Z0-9_]{0,30}.
- Each value has exactly: `producer` (required), `consumer` (optional),
  `reason` (optional, ignored by the checker).
- producer: `METHOD /path response.field[.field...]` - the endpoint MUST be
  copied verbatim from the API Endpoints list and executed in steps.
- consumer: `METHOD /path (body|params).field[.field...]` - the endpoint and
  field must exist in the spec; the request must actually pass this value.
- A placeholder consumed in steps must be produced earlier (store/capture
  verbs, `<ID> = value` assignment, or a binds producer endpoint) - otherwise
  the case is a DRAFT (orphan / forward_reference).
"""

PATH_CONTRACT_INSTRUCTION = """PATH CONTRACT:
- Execute the endpoints in the L2 skeleton IN ORDER, one canonical
  `METHOD /path` per step; pass parameters separately (`params.id=<ID>` or
  a JSON body). Do not substitute concrete ids for the template.
- If the contract marks a DATA_FLOW binding, the target request field MUST
  consume the same `<IDENTIFIER>` produced by the source step.
- For PRECONDITION/SIDE_EFFECT hops, the target step MUST carry an
  observable assertion (`assert/expect/验证/断言 response.<field> <op> <value>`).
- Keep `path_id` out of your output: the program stamps identity fields.
"""

LINKS_CASE_SCHEMA_ADDITIONS: dict[str, Any] = {
    "path_id": {"type": "string", "default": "", "title": "Program-stamped path identity"},
    "source_stage": {
        "type": "string",
        "enum": ["", "phase1", "phase2", "l3b"],
        "default": "",
        "title": "Program-stamped pipeline stage",
    },
    "binds": {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "producer": {"type": "string", "minLength": 1},
                "consumer": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["producer"],
            "additionalProperties": False,
        },
        "default": {},
    },
    "executability": {"type": "string", "default": "", "title": "links executability class"},
}


def render_schema_additions_json() -> str:
    """Schema fragment for the future tasks/testcase JSON schema (S6b)."""
    return json.dumps(LINKS_CASE_SCHEMA_ADDITIONS, ensure_ascii=False, indent=2)
