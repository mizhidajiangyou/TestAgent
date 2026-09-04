"""ReviewHooks — the pipeline's review protocol (plan-d v3 R6 / plan-e E5).

Review is a POST-PROCESS protocol (``artifact → review loop → artifact'``),
not a pipeline stage. This module pins its four responsibilities so the
review runtime is a stable independent protocol, not a pile of executor
branches:

1. ``build_prompt`` — how the review INPUT becomes the user prompt.
2. ``parse``        — how the LLM output is parsed (``None`` marks the
   attempt unparseable → contributes to REVIEW_FAILED).
3. ``apply``        — how the review output becomes the accepted artifact.
   **E5 semantic**: ``apply(original, reviewed) -> accepted_artifact``. The
   ReviewLoop does NOT own artifact-merge semantics; ``apply`` is the ONLY
   place where the domain decides how a review output replaces/merges the
   original — no double merging. Both factories ship the identity form
   (``reviewed`` IS the full new artifact).
4. ``retention_check`` — the rejection guard (``False`` → REVIEW_REJECTED
   keeps the original).

The factories close over a task-specific prompt renderer; parse/apply/
retention are the artifact-type behaviors (list vs text).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from testagent.engine.review import MIN_RETENTION_RATIO
from testagent.pipeline.validators import strip_fences

#: Default review system prompt when the manifest declares none. Real task
#: packages should declare one (``review.system_prompt``) so the request
#: fingerprint is fully task-controlled (plan-e I2).
DEFAULT_REVIEW_SYSTEM_PROMPT = (
    "You are a strict artifact reviewer. Return the complete reviewed "
    "artifact in exactly the same format as the input. Output only the "
    "artifact itself, no commentary."
)

#: Review lifecycle terminal states (R2). REVIEW_FAILED (review did not
#: run to completion — exception or unparseable output) is semantically
#: DISTINCT from REVIEW_REJECTED (ran, but the retention guard refused).
#: REVIEW_DISABLED covers "did not run at all" (switch off, no rounds, or
#: an empty text artifact — legacy ``if review_enabled and script``).
REVIEWED = "REVIEWED"
REVIEW_REJECTED = "REVIEW_REJECTED"
REVIEW_FAILED = "REVIEW_FAILED"
REVIEW_DISABLED = "REVIEW_DISABLED"


@dataclass
class ReviewHooks[T]:
    """The four review responsibilities for one artifact type (R6)."""

    build_prompt: Callable[[T, int], str]
    parse: Callable[[str], T | None]
    apply: Callable[[T, T], T]
    retention_check: Callable[[T, T], bool]


def _parse_json_list(raw: str) -> list[dict[str, Any]] | None:
    """Parse a review response into a list of dicts (fences + envelope)."""
    from testagent.pipeline.runtime import _extract_json_list

    return _extract_json_list(raw)


def _parse_text(raw: str) -> str | None:
    """Parse a script review response (fences stripped; empty → None)."""
    text = strip_fences(raw)
    return text if text.strip() else None


def make_list_hooks(
    render_review: Callable[[str, int], str],
) -> ReviewHooks[list[dict[str, Any]]]:
    """List-shaped artifacts (test cases): JSON serialize in, JSON parse out.

    Retention mirrors the legacy guard exactly (MIN_RETENTION_RATIO on the
    item count; empty original never rejects — first-round semantics).
    """
    return ReviewHooks(
        build_prompt=lambda current, round_idx: render_review(
            json.dumps(current, ensure_ascii=False), round_idx
        ),
        parse=_parse_json_list,
        # E5: the review output IS the full new artifact (identity apply).
        apply=lambda _original, reviewed: reviewed,
        retention_check=lambda original, reviewed: (
            True if not original else len(reviewed) >= len(original) * MIN_RETENTION_RATIO
        ),
    )


def make_text_hooks(
    render_review: Callable[[str, int], str],
    validate: Callable[[str], bool] | None = None,
) -> ReviewHooks[str]:
    """Text-shaped artifacts (k6/jmeter/gui scripts): raw script in/out.

    Scripts are exempt from the retention guard (legacy semantics — a
    legitimately tightened script may be any length).

    ``validate`` mirrors the legacy script-review parse guards (B5.1): the
    legacy perf generator validated each review candidate per format
    (jmeter: ``_validate_jmx``; k6: none) so a bad review answer can never
    corrupt a good first-pass script. The pipeline wires the manifest's
    artifact validators (``when``-guarded by script format) as that guard;
    a failing candidate parses to ``None`` → the round fails.
    """

    def parse(raw: str) -> str | None:
        text = _parse_text(raw)
        if text is None:
            return None
        return text if (validate is None or validate(text)) else None

    return ReviewHooks(
        build_prompt=lambda current, round_idx: render_review(current, round_idx),
        parse=parse,
        apply=lambda _original, reviewed: reviewed,
        retention_check=lambda _original, _reviewed: True,
    )
