"""CLI subcommand modules.

Importing this package triggers each command module's ``@main.command(...)``
registration onto the root :func:`testagent.cli.base.main` group. The modules
themselves are imported here purely for this registration side-effect.
"""

from testagent.cli.commands import (  # noqa: F401  (registration side-effect)
    chat,
    checkpoint,
    config,
    generate_gui,
    generate_perf,
    generate_tests,
    review,
    serve,
    tasks,
)
