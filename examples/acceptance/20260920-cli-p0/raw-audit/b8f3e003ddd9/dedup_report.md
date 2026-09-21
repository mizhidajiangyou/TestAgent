# Scenario dedup report

Removed duplicates: 6
Identity-missing cases kept: 24

- removed TC-001 (key=POST /BOOKS||create||valid-data) in favor of TC-016 [stronger assertion] (global)
- removed TC-017 (key=POST /BOOKS||error-handling||missing-required:title) in favor of TC-002 [duplicate identity; earlier kept] (global)
- removed TC-022 (key=POST /CART/ITEMS||auth||missing-token) in favor of TC-014 [duplicate identity; earlier kept] (global)
- removed TC-036 (key=POST /BOOKS||error-handling||missing-required:title) in favor of TC-002 [duplicate identity; earlier kept] (global)
- removed TC-038 (key=GET /BOOKS||boundary||max-limit) in favor of TC-021 [duplicate identity; earlier kept] (global)
- removed TC-039 (key=POST /CART/ITEMS||create||valid-item) in favor of TC-006 [duplicate identity; earlier kept] (global)

scenario_identity_missing (kept): TC-026, TC-027, TC-028, TC-029, TC-030, TC-031, TC-032, TC-033, TC-034, TC-043, TC-044, TC-045, TC-046, TC-047, TC-048, TC-049, TC-050, TC-051, TC-052, TC-053, TC-054, TC-055, TC-056, TC-057
