"""FH-H parity harness (plan-k section 8.2 / plan-e E1+I2+E2).

One shared harness for both parity lines (testcase / perf+gui). Provides:

- **E1 fixture schema**: six elements per fixture — input, request_trace
  (fingerprint + response text), artifact, failure_semantics, events, meta —
  serialized as one JSON file under ``tests/fixtures/migration/<task>/``.
- **I2 fingerprint**: SHA-256 over (model, system prompt, user prompt,
  params, logical label). Whitelist fields ONLY — session ids, timestamps
  and correlation ids are structurally excluded (never recorded).
- **Failure-semantics observable normalizer**: maps internal unit statuses
  to the observable class the E2 action contract speaks about (empty /
  timeout / provider_error / validation / recovered / ok), so parity
  compares observable behavior, not internal enum identity.
- **Minimal diff**: top-level key diff between fixture and replay; empty
  dict means parity.

Recording discipline: fixtures are recorded BEFORE the change under test
(provenance: recorded_utc + git sha); replay after a change must diff=0.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "TERMINAL_EVENTS",
    "Fingerprint",
    "Fixture",
    "fixture_path",
    "load_fixture",
    "minimal_diff",
    "observable_failure_class",
    "record_fixture",
    "replay_fixture",
]

FIXTURE_ROOT = Path(__file__).parents[1] / "tests" / "fixtures" / "migration"

#: Terminal EngineEvent names (plan-e R5): exactly one per trajectory.
TERMINAL_EVENTS = {"done", "fail"}

#: Observable failure classes (E2 action contract, plan-e): what an outside
#: observer can distinguish. INTERNAL statuses map INTO these.
_OBSERVABLE_MAP = {
    "SUCCESS": "ok",
    "EMPTY": "empty",
    "INVALID": "validation",
    "VALIDATION_ERROR": "validation",
    "TIMEOUT": "timeout",
    "PROVIDER_ERROR": "provider_error",
    "CANCELLED": "cancelled",
}


@dataclass(frozen=True)
class Fingerprint:
    """I2: whitelist-only request fingerprint (no session/time/correlation)."""

    model: str
    system_sha256: str
    user_sha256: str
    params: dict[str, Any]
    label: str

    @classmethod
    def of(
        cls,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        params: dict[str, Any] | None = None,
        label: str = "",
    ) -> Fingerprint:
        return cls(
            model=model,
            system_sha256=hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            user_sha256=hashlib.sha256(user_prompt.encode("utf-8")).hexdigest(),
            params=dict(params or {}),
            label=label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "system_sha256": self.system_sha256,
            "user_sha256": self.user_sha256,
            "params": self.params,
            "label": self.label,
        }


def observable_failure_class(status: str) -> str:
    """Map an internal terminal status to its observable class (E2)."""
    return _OBSERVABLE_MAP.get(status, status.lower())


@dataclass
class Fixture:
    """E1 six-element fixture (plus provenance meta)."""

    name: str
    task: str
    input: dict[str, Any]
    request_trace: list[dict[str, Any]] = field(default_factory=list)
    artifact: Any = None
    failure_semantics: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_version": 1,
            "name": self.name,
            "task": self.task,
            "input": self.input,
            "request_trace": self.request_trace,
            "artifact": self.artifact,
            "failure_semantics": self.failure_semantics,
            "events": self.events,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Fixture:
        return cls(
            name=payload["name"],
            task=payload["task"],
            input=payload.get("input", {}),
            request_trace=payload.get("request_trace", []),
            artifact=payload.get("artifact"),
            failure_semantics=payload.get("failure_semantics", {}),
            events=payload.get("events", []),
            meta=payload.get("meta", {}),
        )


def fixture_path(task: str, name: str) -> Path:
    return FIXTURE_ROOT / task / f"{name}.json"


def load_fixture(task: str, name: str) -> Fixture:
    return Fixture.from_dict(json.loads(fixture_path(task, name).read_text(encoding="utf-8")))


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[1],
            check=True,
        ).stdout.strip()[:12]
    except Exception:
        return "unknown"


def record_fixture(fixture: Fixture) -> Path:
    """Write the fixture with provenance; recording is legitimate only
    BEFORE the change under test (golden discipline). Credentials guard runs
    first - fixtures get committed, so no secret may enter."""
    ensure_no_credentials(fixture.to_dict())
    fixture.meta.setdefault("recorded_utc", _utc_now())
    fixture.meta.setdefault("git_sha", _git_sha())
    path = fixture_path(fixture.task, fixture.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(fixture.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def replay_diff(recorded: Fixture, replayed: Fixture) -> dict[str, Any]:
    """Minimal diff: top-level element comparison (input / request_trace /
    artifact / failure_semantics / events). Empty dict = parity."""
    diff: dict[str, Any] = {}
    a, b = recorded.to_dict(), replayed.to_dict()
    for key in ("input", "request_trace", "artifact", "failure_semantics", "events"):
        if a.get(key) != b.get(key):
            diff[key] = {"recorded": a.get(key), "replayed": b.get(key)}
    return diff


def minimal_diff(recorded: Fixture, replayed: Fixture) -> dict[str, Any]:
    return replay_diff(recorded, replayed)


def replay_fixture(
    task: str,
    name: str,
    scenario: Callable[[dict[str, Any]], Fixture],
) -> dict[str, Any]:
    """Load the recorded fixture, run the scenario fresh, return the
    minimal diff (empty = parity)."""
    recorded = load_fixture(task, name)
    replayed = scenario(recorded.input)
    replayed.name = recorded.name
    replayed.task = recorded.task
    return replay_diff(recorded, replayed)


def ensure_no_credentials(payload: dict[str, Any]) -> None:
    """Safety: no api_key-looking values anywhere in a serialized fixture
    (recording discipline - fixtures get committed)."""
    text = json.dumps(payload, ensure_ascii=False)
    for marker in ("sk-", "api_key", "OPENAI_API_KEY"):
        assert marker not in text, f"credential marker {marker!r} leaked into fixture"


def repo_env() -> dict[str, str]:
    """Environment for subprocess runners (deterministic tests)."""
    env = dict(os.environ)
    env.setdefault("OPENAI_VERIFY_MODEL", "false")
    return env
