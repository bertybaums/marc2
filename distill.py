"""Phase 2: Distill reasoning traces into language-complete descriptions.

For each correctly solved task, a Claude Code subagent converts the reasoning
trace into a structured see/do/grid description suitable for the MARC pipeline.

Usage:
    python distill.py distill --batch-size 10
    python distill.py report
    python distill.py dry-run              # test on 3 tasks
"""

import asyncio
import json
import time

import click

from db import (
    init_db, get_solved_tasks, get_best_solve_trial, get_description,
    insert_description, get_all_tasks,
)
from prompts import build_distill_prompt
from tasks import load_config, get_arc_dir, load_arc_task


async def run_subagent(prompt, timeout_seconds=600):
    """Run a Claude Code subagent and return its text output.

    Returns (output_text, elapsed_seconds, error).
    """
    cmd = [
        "claude",
        "-p", prompt,
        "--tools", "",
        "--model", "opus",
        "--output-format", "text",
        "--no-session-persistence",
    ]

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_seconds
        )
        elapsed = time.monotonic() - start
        output = stdout.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            err_text = stderr.decode("utf-8", errors="replace").strip()
            return None, elapsed, f"Exit code {proc.returncode}: {err_text[:500]}"

        return output, elapsed, None

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return None, elapsed, f"Timeout after {timeout_seconds}s"
    except Exception as e:
        elapsed = time.monotonic() - start
        return None, elapsed, str(e)


def parse_description(output):
    """Parse subagent output to extract see/do/grid description.

    Expects JSON with see_description, do_description, grid_description.
    Returns (see, do, grid, error).
    """
    if not output:
        return None, None, None, "Empty response"

    text = output.strip()

    # Try to find JSON in the response
    json_text = text

    # Strip markdown code blocks
    if "```json" in json_text:
        try:
            start = json_text.index("```json") + 7
            end = json_text.index("```", start)
            json_text = json_text[start:end].strip()
        except ValueError:
            pass
    elif "```" in json_text:
        try:
            start = json_text.index("```") + 3
            end = json_text.index("```", start)
            json_text = json_text[start:end].strip()
        except ValueError:
            pass

    # Try parsing as JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        # Try finding JSON object in the full text
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                return None, None, None, "Could not parse JSON from response"
        else:
            return None, None, None, "No JSON found in response"

    see = data.get("see_description", "")
    do = data.get("do_description", "")
    grid = data.get("grid_description", "")

    if not see or not do:
        return None, None, None, f"Missing fields: see={bool(see)}, do={bool(do)}"

    return see, do, grid, None


async def distill_task(conn, config, task_id, arc_dir):
    """Distill a single task's reasoning into a description.

    Returns (task_id, success, error).
    """
    # Get the best solve trial
    trial = get_best_solve_trial(conn, task_id)
    if not trial or not trial["correct"]:
        return task_id, False, "No correct solve trial"

    reasoning = trial["reasoning"]
    if not reasoning:
        return task_id, False, "No reasoning trace"

    # Get task info
    task_row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    arc_name = task_row["arc_name"]
    source = task_row["source"]

    try:
        arc_task = load_arc_task(arc_dir, arc_name, source)
    except FileNotFoundError as e:
        return task_id, False, str(e)

    # Build prompt and run subagent
    prompt = build_distill_prompt(arc_task, reasoning)
    output, elapsed, error = await run_subagent(prompt)

    if error:
        return task_id, False, error

    # Parse response
    see, do, grid, parse_error = parse_description(output)
    if parse_error:
        return task_id, False, parse_error

    # Store description
    insert_description(conn, task_id, trial["trial_id"], see, do, grid)
    return task_id, True, None


async def distill_batch(conn, config, task_ids, arc_dir, batch_size):
    """Distill a batch of tasks with limited concurrency."""
    semaphore = asyncio.Semaphore(batch_size)

    async def bounded_distill(task_id):
        async with semaphore:
            return await distill_task(conn, config, task_id, arc_dir)

    coros = [bounded_distill(tid) for tid in task_ids]

    total = len(task_ids)
    success_count = 0
    error_count = 0
    done_count = 0

    for coro in asyncio.as_completed(coros):
        task_id, success, error = await coro
        done_count += 1

        if success:
            success_count += 1
            status = "OK"
        else:
            error_count += 1
            status = f"FAIL: {error[:60]}"

        print(f"  [{done_count}/{total}] task {task_id}: {status}")

    return success_count, error_count


@click.group()
def cli():
    """Phase 2: Distill reasoning traces into language-complete descriptions."""
    pass


@cli.command()
@click.option("--batch-size", default=10, help="Max concurrent subagents")
@click.option("--limit", default=None, type=int, help="Max tasks to process")
def distill(batch_size, limit):
    """Distill descriptions for all solved tasks without descriptions."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    # Get solved task_ids that don't have descriptions yet
    solved = {r["task_id"] for r in get_solved_tasks(conn)}
    has_desc = {r["task_id"] for r in conn.execute(
        "SELECT task_id FROM descriptions"
    ).fetchall()}

    task_ids = sorted(solved - has_desc)

    if limit:
        task_ids = task_ids[:limit]

    if not task_ids:
        click.echo("No tasks to distill (all solved tasks already have descriptions)")
        return

    click.echo(f"Distilling {len(task_ids)} tasks (batch_size={batch_size})")
    success, errors = asyncio.run(
        distill_batch(conn, config, task_ids, arc_dir, batch_size)
    )

    click.echo(f"\nResults: {success} distilled, {errors} errors")
    conn.close()


@cli.command()
def report():
    """Show distillation statistics."""
    conn = init_db()

    solved = len(get_solved_tasks(conn))
    total_desc = conn.execute("SELECT COUNT(*) FROM descriptions").fetchone()[0]
    validated = conn.execute(
        "SELECT COUNT(*) FROM descriptions WHERE validated=1"
    ).fetchone()[0]
    failed = conn.execute(
        "SELECT COUNT(*) FROM descriptions WHERE validated=-1"
    ).fetchone()[0]
    untested = conn.execute(
        "SELECT COUNT(*) FROM descriptions WHERE validated=0"
    ).fetchone()[0]

    click.echo(f"Solved tasks: {solved}")
    click.echo(f"Descriptions: {total_desc}/{solved}")
    click.echo(f"  Validated: {validated}")
    click.echo(f"  Failed validation: {failed}")
    click.echo(f"  Untested: {untested}")

    if total_desc > 0:
        avg_see = conn.execute(
            "SELECT AVG(length(see_description)) FROM descriptions"
        ).fetchone()[0]
        avg_do = conn.execute(
            "SELECT AVG(length(do_description)) FROM descriptions"
        ).fetchone()[0]
        avg_grid = conn.execute(
            "SELECT AVG(length(grid_description)) FROM descriptions"
        ).fetchone()[0]
        click.echo(f"\nAvg lengths: see={avg_see:.0f}, do={avg_do:.0f}, grid={avg_grid:.0f} chars")

    conn.close()


@cli.command("dry-run")
@click.option("--count", default=3, help="Number of tasks to test")
def dry_run(count):
    """Test distillation on a few tasks."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    solved = [r["task_id"] for r in get_solved_tasks(conn)][:count]

    click.echo(f"Dry run: distilling {len(solved)} tasks sequentially")
    click.echo()

    for task_id in solved:
        trial = get_best_solve_trial(conn, task_id)
        task_row = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()

        click.echo(f"--- Task {task_id} ({task_row['arc_name']}) ---")
        click.echo(f"  Reasoning: {len(trial['reasoning'])} chars")

        arc_task = load_arc_task(arc_dir, task_row['arc_name'], task_row['source'])
        prompt = build_distill_prompt(arc_task, trial['reasoning'])
        click.echo(f"  Prompt: {len(prompt)} chars")

        click.echo("  Running subagent...")
        output, elapsed, error = asyncio.run(run_subagent(prompt, timeout_seconds=300))

        if error:
            click.echo(f"  ERROR: {error}")
            continue

        click.echo(f"  Response: {len(output)} chars in {elapsed:.1f}s")

        see, do, grid, parse_error = parse_description(output)
        if parse_error:
            click.echo(f"  PARSE ERROR: {parse_error}")
            click.echo(f"  Response: {output[:300]}...")
            continue

        click.echo(f"  See ({len(see)} chars): {see[:100]}...")
        click.echo(f"  Do  ({len(do)} chars): {do[:100]}...")
        click.echo(f"  Grid ({len(grid)} chars): {grid[:100]}...")

        # Store
        insert_description(conn, task_id, trial['trial_id'], see, do, grid)
        click.echo(f"  Stored description")
        click.echo()

    # Verify DB
    click.echo("=== DB verification ===")
    descs = conn.execute(
        "SELECT task_id, length(see_description) as see_len, "
        "length(do_description) as do_len, length(grid_description) as grid_len "
        "FROM descriptions ORDER BY task_id"
    ).fetchall()
    for d in descs:
        click.echo(f"  task {d['task_id']}: see={d['see_len']}, do={d['do_len']}, grid={d['grid_len']} chars")

    conn.close()


if __name__ == "__main__":
    cli()
