"""LLM request fingerprints for migration parity (plan-c B4.10, review P2-11).

Final-output parity alone can pass by coincidence (a fake LLM returning the
same canned response). The fingerprint captures the REQUEST side — model,
prompt hashes and every composed parameter (budget param name, temperature,
effort fragments / extra_body) — so old-vs-new equivalence means both the
artifacts AND the request sequences match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RequestFingerprint:
    """One LLM request's observable contract."""

    model: str
    system_prompt_sha: str
    user_prompt_sha: str
    params: dict[str, Any]  # everything sent besides model/messages
    label: str = ""  # CALL_LABEL (unit attribution)

    def as_tuple(self) -> tuple[Any, ...]:
        return (
            self.model,
            self.system_prompt_sha,
            self.user_prompt_sha,
            json.dumps(self.params, sort_keys=True),
            self.label,
        )


@dataclass
class FingerprintLog:
    """Ordered record of request fingerprints for parity comparison."""

    entries: list[RequestFingerprint] = field(default_factory=list)

    def record(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        params: dict[str, Any],
        label: str = "",
    ) -> None:
        self.entries.append(
            RequestFingerprint(
                model=model,
                system_prompt_sha=_sha(system_prompt),
                user_prompt_sha=_sha(user_prompt),
                params=params,
                label=label,
            )
        )

    def signatures(self) -> list[tuple[Any, ...]]:
        return [e.as_tuple() for e in self.entries]


def diff_fingerprints(old: FingerprintLog, new: FingerprintLog) -> list[str]:
    """Differences between two request sequences; empty = equivalent."""
    a, b = old.signatures(), new.signatures()
    if len(a) != len(b):
        return [f"request count mismatch: old={len(a)} new={len(b)}"]
    diffs: list[str] = []
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        if x != y:
            diffs.append(f"request[{i}]: old={x!r} != new={y!r}")
    return diffs
