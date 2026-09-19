"""Modular capabilities orchestration (plan-modular-capabilities-v4).

Layer boundary (v4 §1.2): may import models/parsers/generators PUBLIC
interfaces; never touches engine/pipeline internals; never mutates frozen
generators. CLI and Container remain the composition roots.
"""

from testagent.orchestration.input_adapter import resolve_requirements, resolve_split_mode
from testagent.orchestration.settings_override import (
    ResolvedCapabilities,
    apply_capabilities_override,
    parse_capability_options,
    resolve_capabilities,
)

__all__ = [
    "ResolvedCapabilities",
    "apply_capabilities_override",
    "parse_capability_options",
    "resolve_capabilities",
    "resolve_requirements",
    "resolve_split_mode",
]
