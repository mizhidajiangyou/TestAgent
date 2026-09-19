"""Capability resolution + settings override (v4 §3.3).

Priority: explicit CLI > resume-saved capability values > current Settings
> field defaults. The override mutates ONLY the selected values on the
container's settings provider (copy-then-override); output_dir, LLM
endpoint, keys, quality switches and test injections stay untouched.
Precondition: consumers of the settings provider not yet resolved — the
caller must invoke this BEFORE resolving llm/generators.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ResolvedCapabilities",
    "apply_capabilities_override",
    "parse_capability_options",
    "resolve_capabilities",
]

_CAPABILITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ResolvedCapabilities:
    """Effective capability values with their provenance."""

    split_mode: str
    concurrency: int
    split_mode_source: str  # cli | resume | settings | default
    concurrency_source: str

    def as_saved(self) -> dict[str, Any]:
        """session record payload: resolved values only (never raw None)."""
        return {
            "schema_version": _CAPABILITY_SCHEMA_VERSION,
            "resolved_split_mode": self.split_mode,
            "resolved_concurrency": self.concurrency,
        }


def _valid_concurrency(value: Any) -> int:
    """Non-bool int >= 1; bool/NaN/negative must not reach the loop."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"concurrency must be an integer >= 1, got {value!r}")
    if value < 1:
        raise ValueError(f"concurrency must be >= 1, got {value}")
    return value


def resolve_capabilities(
    *,
    cli_split: str | None,
    cli_concurrency: int | None,
    saved: dict[str, Any] | None,
    settings: Any,
) -> ResolvedCapabilities:
    """Resolve the effective values by priority chain (v4 §3.3)."""
    settings_split = getattr(settings, "split_mode", "auto")
    settings_conc = _valid_concurrency(
        getattr(getattr(settings, "llm", None), "max_concurrency", 5)
    )

    split_mode = settings_split
    split_source = "settings" if settings_split != "auto" else "default"
    concurrency = settings_conc
    conc_source = "settings" if settings_conc != 5 else "default"

    if saved:
        version = saved.get("schema_version")
        if version != _CAPABILITY_SCHEMA_VERSION:
            raise ValueError(f"unsupported capability_options schema_version: {version!r}")
        saved_split = saved.get("resolved_split_mode")
        saved_conc = saved.get("resolved_concurrency")
        if saved_split is not None:
            if saved_split not in ("auto", "single"):
                raise ValueError(f"corrupt saved split mode: {saved_split!r}")
            split_mode = saved_split
            split_source = "resume"
        if saved_conc is not None:
            concurrency = _valid_concurrency(saved_conc)
            conc_source = "resume"

    if cli_split is not None:
        if cli_split not in ("auto", "single"):
            raise ValueError(f"invalid split mode: {cli_split!r}")
        split_mode = cli_split
        split_source = "cli"
    if cli_concurrency is not None:
        concurrency = _valid_concurrency(cli_concurrency)
        conc_source = "cli"

    return ResolvedCapabilities(
        split_mode=split_mode,
        concurrency=concurrency,
        split_mode_source=split_source,
        concurrency_source=conc_source,
    )


def parse_capability_options(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Read ``capability_options`` from a session record. Absent object ->
    None (legacy record: current Settings apply); corrupt container ->
    explicit error (never silent replacement)."""
    if record is None:
        return None
    options = record.get("capability_options")
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError(f"capability_options must be an object, got {type(options).__name__}")
    return options


def apply_capabilities_override(container: Any, resolved: ResolvedCapabilities) -> None:
    """Override ONLY the selected settings values on the container's
    settings provider (copy of the already-injected Settings object).

    Preconditions (caller obligation, proven by tests):
    - llm/generators depending on these values are NOT yet resolved;
    - never rebuild the Container, never reset resolved singletons,
      never touch private fields.
    """
    current = container.settings()
    replacement = copy.deepcopy(current)
    replacement.split_mode = resolved.split_mode
    replacement.llm.max_concurrency = resolved.concurrency
    container.settings.override(replacement)
