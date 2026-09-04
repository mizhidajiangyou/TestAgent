"""Cross-layer shared constants (plan-e B5.2a).

Values borrowed across architectural layers live here so no layer imports a
sibling's internals just for a constant. Origin: ``DEFAULT_TARGET_URL`` was
defined in ``generators/gui_test_generator.py`` and imported by
``engine/conversation.py`` — a reverse dependency that would break when the
GUI generator is deleted (plan-d B5.4). The value is frozen; behaviour is
unchanged.
"""

#: Fallback target URL for GUI test generation when the caller provides none.
#: Historical value (pre-B5.2a it lived in gui_test_generator.py) — do NOT
#: change without a migration-parity decision: prompts embed this default.
DEFAULT_TARGET_URL = "https://example.com"
