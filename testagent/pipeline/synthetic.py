"""Synthetic template-context construction (plan-c B3.4).

Pure function over a manifest — NO I/O, no Registry dependency (that is the
B4.3 consumption side; keeping construction pure is what broke the v3
circular dependency B3.4 <-> B4.3).

The synthetic context lets ``tasks validate`` render templates with
``StrictUndefined`` against a REPRESENTATIVE context instead of an empty
one, so a template referencing a variable the pipeline forgets to inject
fails at validation time — the empty-context smoke render could not catch
that (optional variables guard with ``default('')`` and mandatory ones
silently render empty).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from testagent.pipeline.manifest import Manifest

#: Sample values per input kind, used when the manifest does not declare an
#: explicit ``template_context`` entry for the variable. int/bool use native
#: types — templates legitimately do arithmetic on them (B5.1: k6 think-time
#: ``{{ think_time_ms // 1000 }}``).
_KIND_SAMPLES: dict[str, Any] = {
    "swagger": "GET /items - list items\nPOST /items - create item",
    "requirements": "- REQ-001: sample requirement",
    "file": "sample file content",
    "text": "sample text",
    "choice": "sample-choice",
    "int": 10,
    "bool": True,
}


def build_synthetic_context(manifest: Manifest) -> dict[str, Any]:
    """Construct a representative template context for validation renders.

    Priority per variable name:
    1. explicit ``manifest.template_context[name]``;
    2. derived from the declared input's ``kind`` sample;
    3. fallback empty string (the variable may be pipeline-injected, not
       input-derived — the empty fallback still catches StrictUndefined
       references to variables that are declared NOWHERE).

    Well-known pipeline-injected variables get plausible samples too, so
    templates referencing them render meaningfully.
    """
    ctx: dict[str, Any] = dict(manifest.template_context)
    for spec in manifest.inputs:
        var = spec.context_var or spec.name
        if var in ctx:
            continue
        default = spec.default
        if isinstance(default, str) and not default.startswith("from_settings:"):
            ctx[var] = default
        else:
            ctx[var] = _KIND_SAMPLES.get(spec.kind, "")

    # Derived views the pipeline always provides alongside parsed inputs.
    derived_defaults = {
        "endpoints_text": "GET /items - list items",
        "endpoints_signature": "- GET /items",
        "requirements_text": "- REQ-001: sample requirement",
        "output_language": "english",
        "json_mode": False,
        "historical_cases": "",
        "already_covered": "",
        # Review-stage injected variables (plan-d B5.0): the review template
        # renders with the serialized artifact and the round number. The
        # script_* trio covers templates migrated verbatim from the legacy
        # script_review_prompt.j2.
        "artifact": "[]",
        "round": 1,
        "script_kind": "k6",
        "script": "script content",
        "context_text": "## Review Context",
    }
    for key, value in derived_defaults.items():
        ctx.setdefault(key, value)
    return ctx
