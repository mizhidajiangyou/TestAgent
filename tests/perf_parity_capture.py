"""Shared machinery for the perf fingerprint-parity gates.

Legacy-free by construction: ``test_perf_parity_oracle.py`` replays the recorded
cells through the task-package chain only, and that is the gate which has to
survive plan-k B5.4 deleting ``PerformanceGenerator``. The live
legacy-vs-pipeline comparison (and the one place that may RECORD cells) is
``tests/test_perf_parity_legacy.py``.

Cells come from the B5.1 matrix: script format x review on/off x output
language x fenced-or-plain answer, plus the three named cases that pin
settings-defaults, CLI overrides and the empty-answer behavior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from testagent.engine.llm_client import LLMResponse
from testagent.pipeline.executor import PipelineExecutor
from testagent.pipeline.inputs import parse_inputs
from testagent.pipeline.manifest import load_manifest
from testagent.pipeline.registry import TaskPackage
from testagent.pipeline.runtime import build_generate_unit, build_review_runner

REPO = Path(__file__).parents[1]
TASK_ROOT = REPO / "tasks" / "perf"
SWAGGER = REPO / "examples" / "ecommerce_swagger.json"

#: One source of truth for the perf parameters on both sides.
PERF_KW: dict[str, Any] = {
    "base_url": "https://api.example.com",
    "virtual_users": 100,
    "duration_seconds": 300,
    "ramp_up_seconds": 60,
    "think_time_ms": 500,
    "auth_type": "none",
}

#: Script constants double as EXPECTED artifacts — both the legacy
#: ``_extract_script`` and the pipeline ``strip_fences`` strip surrounding
#: whitespace, so the constants are the post-strip forms.
K6_SCRIPT = "import http from 'k6/http';\nexport default function () {}"
K6_SCRIPT_FENCED = f"```javascript\n{K6_SCRIPT}\n```"
K6_REVISED = "import http from 'k6/http';\nexport default function () { sleep(1); }"
JMX_SCRIPT = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<jmeterTestPlan version="1.2">\n</jmeterTestPlan>'
)
JMX_SCRIPT_FENCED = f"```xml\n{JMX_SCRIPT}\n```"
JMX_REVISED = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<jmeterTestPlan version="1.2">\n'
    "  <hashTree/>\n"
    "</jmeterTestPlan>"
)

#: ``legacy_raises_on_invalid_jmx`` is the registered divergence (B5.3): the
#: legacy chain raises ValueError, the pipeline records an INVALID unit.
OVERRIDE_KW: dict[str, Any] = {
    "base_url": "https://staging.example.dev",
    "virtual_users": 42,
    "duration_seconds": 120,
    "ramp_up_seconds": 15,
    "think_time_ms": 1500,
    "auth_type": "bearer",
}


def sha12(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class _SubClient:
    def __init__(self, script: list[Any], role: str, sink: list[tuple[str, str, str]]) -> None:
        self._script = list(script)
        self._role = role
        self._sink = sink
        self.model_name = f"{role}-m"

    def _respond(self, system: str, user: str) -> Any:
        self._sink.append((self._role, system, user))
        idx = sum(1 for r, *_ in self._sink if r == self._role) - 1
        return self._script[min(idx, len(self._script) - 1)]

    @staticmethod
    def _as_response(item: Any) -> LLMResponse:
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, Exception):
            raise item
        return LLMResponse(text=str(item), finish_reason="stop")

    def chat(self, system: str, user: str) -> str:
        return self._as_response(self._respond(system, user)).text

    def chat_with_meta(self, system: str, user: str) -> LLMResponse:
        return self._as_response(self._respond(system, user))

    async def achat_with_meta(self, system: str, user: str) -> LLMResponse:
        return self.chat_with_meta(system, user)


class ParityFakeLLM:
    """Multi-model-shaped double: primary + secondary sub-clients.

    Records ``(role, system, user)`` on every path — the legacy generator calls
    sync ``chat``/``chat_with_meta``, the pipeline calls ``achat_with_meta`` —
    so the same object shows whether the two chains ask the same things of the
    same roles.
    """

    intent_capable = True
    primary_model = "primary-m"

    def __init__(self, primary_script: list[Any], secondary_script: list[Any] | None = None):
        self.calls: list[tuple[str, str, str]] = []
        self.primary = _SubClient(primary_script, "primary", self.calls)
        self._secondary = _SubClient(
            secondary_script if secondary_script is not None else primary_script,
            "secondary",
            self.calls,
        )
        self.session_id = ""

    @property
    def pairs(self) -> list[tuple[str, str]]:
        return [(system, user) for _, system, user in self.calls]

    @property
    def roles(self) -> list[str]:
        return [role for role, *_ in self.calls]

    def secondary_client(self) -> _SubClient:
        return self._secondary

    def set_session_id(self, sid: str) -> None:
        self.session_id = sid

    async def averify(self) -> None:
        return None

    def chat(self, system: str, user: str) -> str:
        return self.primary.chat(system, user)

    def chat_with_meta(self, system: str, user: str) -> LLMResponse:
        return self.primary.chat_with_meta(system, user)

    async def achat_with_meta(self, system: str, user: str) -> LLMResponse:
        return await self.primary.achat_with_meta(system, user)


class Settings:
    """Plain settings double for the perf chain (every field the package reads)."""

    def __init__(
        self,
        output_dir: str,
        *,
        review_enabled: bool = False,
        output_language: str = "english",
        script_format: str = "k6",
    ) -> None:
        self.output_dir = output_dir
        self.script_format = script_format
        self.output_language = output_language
        self.review_enabled = review_enabled
        self.review_max_rounds = 2
        self.llm = _LLM()
        self.perf = _Perf()


class _LLM:
    max_concurrency = 2
    json_mode = False


class _Perf:
    base_url = PERF_KW["base_url"]
    virtual_users = PERF_KW["virtual_users"]
    duration_seconds = PERF_KW["duration_seconds"]
    ramp_up_seconds = PERF_KW["ramp_up_seconds"]
    think_time_ms = PERF_KW["think_time_ms"]
    auth_type = PERF_KW["auth_type"]


def task_package() -> TaskPackage:
    manifest = load_manifest(TASK_ROOT / "manifest.json")
    return TaskPackage(manifest.name, manifest, TASK_ROOT)


async def run_pipeline(tmp_path: Path, cell: dict[str, Any]) -> dict[str, Any]:
    """Drive ``tasks/perf`` for one cell and return the observable boundary."""
    settings = Settings(
        str(tmp_path),
        review_enabled=cell["review_enabled"],
        output_language=cell["output_language"],
        script_format=cell["script_format"],
    )
    fake = ParityFakeLLM(cell["primary_script"], cell["secondary_script"])
    task = task_package()
    raw: dict[str, Any] = {"swagger": str(SWAGGER), "script_format": cell["script_format"]}
    if cell.get("overrides"):
        raw.update(cell["overrides"])
    elif cell.get("omit_format"):
        raw.pop("script_format")
    ctx = parse_inputs(task.manifest, raw, settings)
    executor = PipelineExecutor(
        fake,  # type: ignore[arg-type]
        settings,  # type: ignore[arg-type]
        generate_unit=build_generate_unit(fake),  # type: ignore[arg-type]
        review_runner=build_review_runner(fake),  # type: ignore[arg-type]
    )
    result = await executor.arun(task, ctx, session_id=f"parity-{cell['name']}")
    return {
        "pairs": [[system, user] for system, user in fake.pairs],
        "roles": list(fake.roles),
        "artifact": result.artifact,
        "review_status": (result.review_meta or {}).get("status"),
    }


def _matrix_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for script_format in ("k6", "jmeter"):
        gen = JMX_SCRIPT_FENCED if script_format == "jmeter" else K6_SCRIPT_FENCED
        plain = JMX_SCRIPT if script_format == "jmeter" else K6_SCRIPT
        revised = JMX_REVISED if script_format == "jmeter" else K6_REVISED
        for fenced, gen_answer in ((False, plain), (True, gen)):
            for review_enabled in (False, True):
                for output_language in ("english", "chinese"):
                    name = "-".join(
                        (
                            script_format,
                            "review" if review_enabled else "noreview",
                            output_language,
                            "fenced" if fenced else "plain",
                        )
                    )
                    cells.append(
                        {
                            "name": name,
                            "script_format": script_format,
                            "review_enabled": review_enabled,
                            "output_language": output_language,
                            "primary_script": [gen_answer, plain]
                            if review_enabled
                            else [gen_answer],
                            "secondary_script": [revised] if review_enabled else None,
                            "expected_artifact": plain if review_enabled else gen_answer.strip(),
                        }
                    )
    return cells


CELLS: list[dict[str, Any]] = [
    *_matrix_cells(),
    {
        "name": "defaults-jmeter",
        "script_format": "jmeter",
        "review_enabled": False,
        "output_language": "english",
        "primary_script": [JMX_SCRIPT],
        "secondary_script": None,
        "omit_format": True,
        "expected_artifact": JMX_SCRIPT,
    },
    {
        "name": "overrides-k6",
        "script_format": "k6",
        "review_enabled": False,
        "output_language": "english",
        "primary_script": [K6_SCRIPT],
        "secondary_script": None,
        "overrides": OVERRIDE_KW,
        "expected_artifact": K6_SCRIPT,
    },
    {
        "name": "empty-k6",
        "script_format": "k6",
        "review_enabled": True,
        "output_language": "english",
        "primary_script": [""],
        "secondary_script": [],
        "expected_artifact": "",
    },
]

CELL_NAMES = [cell["name"] for cell in CELLS]
