"""TASKS_DIR discovery + Registry with global conflict detection
(plan-c B4.3 + B3.5, review P1-10).

Conflict detection happens HERE, at construction time — all four collision
classes (name/name, name/alias, alias/name, alias/alias) plus reserved
commands are reported once, instead of the CLI registration layer deciding
winner-by-insertion-order at runtime (click 8.4.2 silently OVERWRITES
same-named commands — verified in the v2 plan's environment probe).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import ValidationError

from testagent.pipeline.manifest import RESERVED_COMMANDS, load_manifest
from testagent.pipeline.synthetic import build_synthetic_context

logger = logging.getLogger(__name__)


class TaskPackage:
    """One loaded task package: manifest + private jinja2 environment."""

    def __init__(self, name: str, manifest: Any, root: Path) -> None:
        self.name = name
        self.manifest = manifest
        self.root = root
        # StrictUndefined (review P1-9/#10): a template referencing a
        # variable the pipeline never provides fails LOUDLY at validation
        # time instead of rendering an empty string at generation time.
        self._env = Environment(
            loader=FileSystemLoader(str(root)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )
        self._synthetic_context = build_synthetic_context(manifest)

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with the run context. Synthetic-context values
        act as defaults for variables the run did not provide, so a partial
        context still renders (validation already proved the full shape)."""
        merged = {**self._synthetic_context, **context}
        return self._env.get_template(template_name).render(**merged)

    def system_prompt(self, spec: str, context: dict[str, Any] | None = None) -> str:
        """Resolve a system_prompt spec.

        Three forms:

        - ``file:<path>`` — a jinja template inside the package, rendered with
          ``context`` (B5.1: the legacy perf/gui system prompts vary with the
          script format and output language, so a static string cannot
          reproduce that and the fingerprint gate catches the drift);
        - ``generation:<name>`` — one of the shared system-prompt bases from
          ``config/prompt_contract``, run through THE composition order every
          chain uses (base → historical-baseline suffix → language hint → error
          contract → JSON-mode wrapper). A task package that spelled only the
          base would send prompts without the error contract, which is a
          quality regression no artifact diff can see;
        - ``inline:<text>`` — the text verbatim.

        Rendered output is stripped: system prompts are logical text, and
        trailing file newlines must not leak into request fingerprints.
        """
        if spec.startswith("file:"):
            return self.render(spec.removeprefix("file:"), context or {}).strip()
        if spec.startswith("generation:"):
            from testagent.config.prompt_contract import compose_named_system_prompt

            ctx = context or {}
            return compose_named_system_prompt(
                spec.removeprefix("generation:"),
                output_language=str(ctx.get("output_language") or "english"),
                json_mode=bool(ctx.get("json_mode")),
                historical_cases=str(ctx.get("historical_cases") or ""),
            )
        if spec.startswith("inline:"):
            return spec.removeprefix("inline:")
        return spec

    def validate_renderable(self) -> list[str]:
        """Smoke-render every declared template with the SYNTHETIC context
        (plan-c B3.4). Catches missing files, syntax errors AND undefined
        variables (StrictUndefined) — the v2 empty-context render could not
        catch the last class.
        """
        problems: list[str] = []
        for ref in self.manifest.template_refs():
            try:
                self._env.get_template(ref).render(**self._synthetic_context)
            except Exception as exc:
                problems.append(f"{ref}: {exc}")
        return problems

    def validate_references(self) -> list[str]:
        """Check declared file references exist (templates + schema refs)."""
        problems: list[str] = []
        for ref in self.manifest.template_refs():
            if not (self.root / ref).exists():
                problems.append(f"template not found: {ref}")
        schema_ref = getattr(self.manifest.artifact, "schema_ref", "")
        if schema_ref and not (self.root / schema_ref).exists():
            problems.append(f"schema not found: {schema_ref}")
        return problems


class Registry:
    """Name-indexed set of task packages with construction-time conflict
    detection (plan-c B3.5)."""

    def __init__(self, packages: list[TaskPackage]) -> None:
        self._by_name: dict[str, TaskPackage] = {}
        self._by_alias: dict[str, str] = {}
        self.conflicts: list[str] = []
        for pkg in packages:
            self._register(pkg)

    def _register(self, pkg: TaskPackage) -> None:
        name = pkg.name
        if name in self._by_name:
            self.conflicts.append(f"name/name: {name!r} declared by two packages")
            return
        if name in self._by_alias:
            self.conflicts.append(f"alias/name: {name!r} collides with an existing alias")
            return
        if name in RESERVED_COMMANDS:
            self.conflicts.append(f"name {name!r} is a reserved command")
            return
        for alias in pkg.manifest.aliases:
            if alias in self._by_name:
                self.conflicts.append(
                    f"alias/name: {alias!r} (of {name!r}) collides with a task name"
                )
                continue
            if alias in self._by_alias:
                self.conflicts.append(f"alias/alias: {alias!r} (of {name!r}) already registered")
                continue
            if alias in RESERVED_COMMANDS:
                self.conflicts.append(f"alias {alias!r} is a reserved command")
                continue
            self._by_alias[alias] = name
        self._by_name[name] = pkg

    def tasks(self, *, include_hidden: bool = False) -> list[TaskPackage]:
        """Visible packages (underscore-prefixed names hidden by default)."""
        return [p for p in self._by_name.values() if include_hidden or not p.name.startswith("_")]

    def get(self, name: str) -> TaskPackage:
        if name in self._by_name:
            return self._by_name[name]
        target = self._by_alias.get(name)
        if target is not None:
            return self._by_name[target]
        raise KeyError(f"Unknown task package {name!r}")

    def __contains__(self, name: str) -> bool:
        return name in self._by_name or name in self._by_alias


class DirectorySource:
    """Scans TASKS_DIR for task packages (plan-c B4.3).

    The base dir is resolved to an ABSOLUTE path at construction: task
    packages register their templates against it, and a later cwd change
    (tests, daemonized runs) must not invalidate the loader.
    """

    def __init__(self, base_dir: Path | str) -> None:
        self._base = Path(base_dir).resolve()

    def discover(self) -> list[TaskPackage]:
        packages: list[TaskPackage] = []
        if not self._base.exists():
            logger.info("TASKS_DIR %s does not exist; no task packages.", self._base)
            return packages
        for child in sorted(self._base.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "manifest.json"
            if not manifest_path.exists():
                continue
            try:
                manifest = load_manifest(manifest_path)
            except (json.JSONDecodeError, ValidationError, OSError) as exc:
                logger.warning("Task package %s skipped: invalid manifest (%s)", child.name, exc)
                continue
            if manifest.name != child.name:
                logger.warning(
                    "Task package %s skipped: manifest name %r != directory name",
                    child.name,
                    manifest.name,
                )
                continue
            pkg = TaskPackage(manifest.name, manifest, child)
            problems = pkg.validate_renderable() + pkg.validate_references()
            if problems:
                # Load but flag (design doc §5.2: degrade, don't crash) —
                # `tasks validate --strict` is the fail-loud mode.
                logger.warning("Task package %s has broken references: %s", child.name, problems)
            packages.append(pkg)
        return packages


def get_registry(base_dir: Path | str) -> Registry:
    return Registry(DirectorySource(base_dir).discover())
