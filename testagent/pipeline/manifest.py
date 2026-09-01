"""Manifest schema for task packages (plan-c B1.4/B4.2, manifest_version 1).

Strictness rules (review P0-4):
- Every model uses ``extra="forbid"`` — a typo'd field name fails validation
  with a field path instead of being silently ignored.
- ``manifest_version`` accepts exactly one value (1); unknown future
  versions fail fast with an actionable message.
- ``name``/``aliases`` must match ``[a-z0-9_-]+``; aliases are unique within
  the manifest, distinct from ``name`` and not reserved CLI commands.

Stage dataflow contract (plan-c B1.5): v1 pipelines are FAN-OUT + MERGE only
(input -> N independent stages -> merge). There is NO stage-to-stage
dependency — a stage cannot consume another stage's artifact. Supporting
stage chains requires a manifest_version bump and explicit design.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: CLI commands the dynamic registration must never shadow (plan-c B3.5).
RESERVED_COMMANDS: frozenset[str] = frozenset(
    {"tasks", "checkpoint", "config", "chat", "serve", "version", "help", "run"}
)

#: The only manifest schema version this code understands.
SUPPORTED_MANIFEST_VERSIONS: tuple[int, ...] = (1,)

_NAME_RE = re.compile(r"[a-z0-9_-]+")

InputKind = Literal["swagger", "requirements", "file", "text", "choice", "int", "bool"]
SplitBy = Literal["single", "per_input", "batch"]
ArtifactType = Literal["structured", "text"]
ValidatorKind = Literal["python_compile", "xml", "regex", "contains"]
ContractKind = Literal["json_list", "text"]


class StrictModel(BaseModel):
    """Base for every manifest model: unknown fields are validation errors."""

    model_config = ConfigDict(extra="forbid")


def _validate_name(value: str) -> str:
    if not _NAME_RE.fullmatch(value):
        raise ValueError(f"name {value!r} must match [a-z0-9_-]+")
    if value in RESERVED_COMMANDS:
        raise ValueError(f"name {value!r} collides with a reserved CLI command")
    return value


class InputSpec(StrictModel):
    name: str
    kind: InputKind
    option: list[str] = Field(default_factory=list)  # empty = ["--<name>"]
    required: bool = False
    help: str = ""
    context_var: str = ""  # empty = name
    parse: str = ""  # kind=file: testcase_history|json|text
    choices: list[str] = Field(default_factory=list)  # kind=choice
    default: str | int | bool | None = None  # "from_settings:KEY" supported

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        return _validate_name(v)

    @model_validator(mode="after")
    def _choice_needs_choices(self) -> InputSpec:
        if self.kind == "choice" and not self.choices:
            raise ValueError(f"input {self.name!r}: kind=choice requires choices")
        return self


class WhenSpec(StrictModel):
    input_present: list[str] = Field(default_factory=list)  # AND within list
    input_absent: list[str] = Field(default_factory=list)


class SplitSpec(StrictModel):
    by: SplitBy = "single"
    input: str = ""
    batch_size: int = 2


class StageOutput(StrictModel):
    contract: ContractKind = "json_list"
    unwrap_keys: list[str] = Field(default_factory=list)


class StageSpec(StrictModel):
    name: str
    template: str
    system_prompt: str  # "file:<path>" | "inline:<text>"
    when: WhenSpec = Field(default_factory=WhenSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)
    inject: dict[str, str] = Field(default_factory=dict)
    output: StageOutput = Field(default_factory=StageOutput)

    @model_validator(mode="after")
    def _split_input_required(self) -> StageSpec:
        if self.split.by in ("per_input", "batch") and not self.split.input:
            raise ValueError(
                f"stage {self.name!r}: split.by={self.split.by!r} requires split.input"
            )
        return self


class DedupSpec(StrictModel):
    keys: list[str] = Field(default_factory=list)
    normalize: Literal["lower", "none"] = "none"


class RenumberSpec(StrictModel):
    field: str = "id"
    format: str = "TC-{i:03d}"


class MergeSpec(StrictModel):
    baseline_input: str = ""
    dedup: DedupSpec = Field(default_factory=DedupSpec)
    renumber: RenumberSpec = Field(default_factory=RenumberSpec)


class TruncationSpec(StrictModel):
    enabled: bool = True
    policy: str = "from_settings"
    slim_continue: bool = True
    scope_key_field: str = ""


class ValidatorSpec(StrictModel):
    kind: ValidatorKind
    root: str = ""  # xml: expected root tag
    pattern: str = ""  # regex / contains
    when: dict[str, str] = Field(default_factory=dict)  # {"format": "jmeter"}


class ArtifactSpec(StrictModel):
    type: ArtifactType = "structured"
    item_schema: dict[str, Any] | None = None  # inline JSON Schema
    schema_ref: str = ""  # "$ref"-style relative path, e.g. "schema/xxx.json"
    list_wrapper_keys: list[str] = Field(default_factory=list)
    validators: list[ValidatorSpec] = Field(default_factory=list)


class ReviewSpec(StrictModel):
    enabled: bool | str = "from_settings"
    max_rounds: int | str = "from_settings"
    template: str = ""
    context: list[str] = Field(default_factory=list)
    call_via_executor: bool = True


class OutputSpec(StrictModel):
    default_path: str = ""
    formats: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SessionSpec(StrictModel):
    record: bool = True
    #: RESERVED (plan-c B3.3): accepted for forward compatibility, IGNORED by
    #: the executor with a warning. True pipeline resume (completed_units
    #: level) is a future manifest_version feature.
    resume: bool = True


class PipelineSpec(StrictModel):
    verify_model: bool | str = "from_settings"
    json_mode: bool | str = "from_settings"
    max_concurrency: int | str = "from_settings"
    output_language: str | str = "from_settings"
    stages: list[StageSpec]
    merge: MergeSpec = Field(default_factory=MergeSpec)
    truncation: TruncationSpec = Field(default_factory=TruncationSpec)
    fan_out_recover: bool = True

    @field_validator("stages")
    @classmethod
    def _stages_non_empty(cls, v: list[StageSpec]) -> list[StageSpec]:
        if not v:
            raise ValueError("pipeline.stages must contain at least one stage")
        return v


class Manifest(StrictModel):
    manifest_version: int = 1
    name: str
    display_name: str = ""
    description: str = ""
    version: str = "0.0.0"
    aliases: list[str] = Field(default_factory=list)
    examples: str = ""
    inputs: list[InputSpec] = Field(default_factory=list)
    require_any: list[str] = Field(default_factory=list)
    pipeline: PipelineSpec
    artifact: ArtifactSpec = Field(default_factory=ArtifactSpec)
    review: ReviewSpec = Field(default_factory=ReviewSpec)
    output: OutputSpec = Field(default_factory=OutputSpec)
    session: SessionSpec = Field(default_factory=SessionSpec)
    #: Synthetic template context (plan-c B3.4): variable name -> sample value
    #: used by ``tasks validate`` to smoke-render templates with
    #: StrictUndefined, catching variables the pipeline forgets to inject
    #: (empty-context renders cannot).
    template_context: dict[str, str] = Field(default_factory=dict)

    @field_validator("manifest_version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v not in SUPPORTED_MANIFEST_VERSIONS:
            raise ValueError(
                f"unsupported manifest_version {v}; supported: {list(SUPPORTED_MANIFEST_VERSIONS)}"
            )
        return v

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        return _validate_name(v)

    @field_validator("aliases")
    @classmethod
    def _valid_aliases(cls, v: list[str]) -> list[str]:
        for alias in v:
            if not _NAME_RE.fullmatch(alias):
                raise ValueError(f"alias {alias!r} must match [a-z0-9_-]+")
            if alias in RESERVED_COMMANDS:
                raise ValueError(f"alias {alias!r} collides with a reserved CLI command")
        if len(set(v)) != len(v):
            raise ValueError(f"duplicate aliases in {v!r}")
        return v

    @model_validator(mode="after")
    def _alias_not_name(self) -> Manifest:
        if self.name in self.aliases:
            raise ValueError(f"alias {self.name!r} duplicates the manifest name")
        return self

    def template_refs(self) -> list[str]:
        """Every template this package declares (stages + review)."""
        refs = [s.template for s in self.pipeline.stages]
        for stage in self.pipeline.stages:
            if stage.system_prompt.startswith("file:"):
                refs.append(stage.system_prompt.removeprefix("file:"))
        if self.review.template:
            refs.append(self.review.template)
        return refs


def load_manifest(path: Path) -> Manifest:
    """Parse + validate one manifest.json; pydantic errors carry field paths."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)
