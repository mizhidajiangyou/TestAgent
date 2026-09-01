"""TestAgent CLI package.

Re-exports ``main`` (the click root group) and imports :mod:`testagent.cli.commands`
to register all subcommands. The ``testagent.cli:main`` entry point declared in
``pyproject.toml`` resolves to ``main`` here, preserving the original
single-module import contract (``from testagent.cli import main``).

Task-package commands (plan-c B4.7) register AFTER the legacy commands so
the collision policy in :func:`testagent.pipeline.clicommand.register_tasks`
can keep legacy commands authoritative during migration (click 8.4.2
silently overwrites same-named commands).
"""

# Importing the commands package triggers each subcommand's @main.command(...)
# decorator, registering them on the root group before any CLI invocation.
from testagent.cli import commands  # noqa: F401  (registration side-effect)
from testagent.cli.base import main
from testagent.config.settings import get_settings
from testagent.pipeline.clicommand import register_tasks
from testagent.pipeline.registry import get_registry

register_tasks(main, get_registry(get_settings().tasks_dir))

__all__ = ["main"]
