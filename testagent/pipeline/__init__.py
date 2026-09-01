"""Task-package pipeline package (plan-c Step 1-3).

Modules:
- :mod:`testagent.pipeline.manifest` — strict pydantic schema for manifest.json
- :mod:`testagent.pipeline.parity` — artifact comparison for migration parity
- :mod:`testagent.pipeline.status` — UnitStatus taxonomy + v10 Outcome mapping
- :mod:`testagent.pipeline.synthetic` — synthetic template-context construction
- :mod:`testagent.pipeline.inputs` / ``split`` / ``validators`` / ``merge`` /
  ``writers`` — the five pipeline stages' building blocks
- :mod:`testagent.pipeline.registry` — TASKS_DIR discovery + conflict detection
- :mod:`testagent.pipeline.executor` — PipelineExecutor + pre-review snapshots
- :mod:`testagent.pipeline.clicommand` — dynamic CLI command registration
- :mod:`testagent.pipeline.fingerprint` — LLM request fingerprints for parity

Architecture boundary (plan-c B4.11): nothing in this package may import
from ``testagent.generators``, ``testagent.engine.prompt_builder`` or
``testagent.engine.conversation`` — the pipeline is a peer layer, not a
consumer of the legacy generators.
"""
