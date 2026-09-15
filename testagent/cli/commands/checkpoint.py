"""``testagent checkpoint`` — inspect and recover pre-review snapshots
(plan-c B4.8, review P0-3).

These are ARTIFACT snapshots, not pipeline checkpoints: recovery hands back
the pre-review artifact of a session whose review exploded. There is no
``--resume`` — mid-pipeline resume is a future manifest_version feature and
the naming refuses to imply it.
"""

import json

import click

from testagent.cli.base import console, main
from testagent.pipeline.executor import list_snapshots, recover_snapshot


@main.group()
def checkpoint() -> None:
    """Inspect and recover pre-review snapshots."""


@checkpoint.command("list")
@click.pass_context
def checkpoint_list(ctx: click.Context) -> None:
    """List recoverable snapshots (newest first)."""
    out = ctx.obj["container"].settings().output_dir
    rows = list_snapshots(out)
    if not rows:
        console.print("[yellow]No snapshots found.[/]")
        return
    for row in rows:
        console.print(
            f"[cyan]{row['session_id']}[/]  task={row['task']}  "
            f"items={row['count']}  created={row['created_at']}"
        )


@checkpoint.command("recover")
@click.argument("session_id")
@click.option("--save-as", default=None, help="Write the recovered artifact to this path.")
@click.pass_context
def checkpoint_recover(ctx: click.Context, session_id: str, save_as: str | None) -> None:
    """Recover a session's pre-review artifact."""
    out = ctx.obj["container"].settings().output_dir
    payload = recover_snapshot(session_id, out)
    if payload is None:
        console.print(f"[red]No snapshot for session {session_id}. Use 'checkpoint list'.[/]")
        raise SystemExit(1)
    artifact = payload["artifact"]
    if save_as:
        with open(save_as, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        console.print(f"[green]Recovered {payload['count']} item(s) -> {save_as}[/]")
    else:
        console.print_json(json.dumps(artifact, ensure_ascii=False))
