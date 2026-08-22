"""TestAgent CLI package.

Re-exports ``main`` (the click root group) and imports :mod:`testagent.cli.commands`
to register all subcommands. The ``testagent.cli:main`` entry point declared in
``pyproject.toml`` resolves to ``main`` here, preserving the original
single-module import contract (``from testagent.cli import main``).
"""

# Importing the commands package triggers each subcommand's @main.command(...)
# decorator, registering them on the root group before any CLI invocation.
from testagent.cli import commands  # noqa: F401  (registration side-effect)
from testagent.cli.base import main

__all__ = ["main"]
