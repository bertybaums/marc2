"""Phase 1: Solve ARC-AGI2 tasks via Claude Code subagents.

Orchestrates parallel Claude Code CLI invocations to solve ARC-AGI2 tasks,
capturing full reasoning traces for every attempt.

Usage:
    python solve.py solve --split training --batch-size 10
    python solve.py solve --split evaluation --batch-size 10
    python solve.py retry --batch-size 10
    python solve.py report
    python solve.py dry-run              # test on 3 tasks
"""

import asyncio
import json
import sys
import time

import click

from db import (
    init_db, get_all_tasks, insert_solve_trial, update_solve_trial,
    get_solved_tasks, get_unsolved_tasks, get_best_solve_trial,
)
from grids import compare_grids, grid_to_text, parse_response_grid
from prompts import build_solve_prompt, build_solve_retry_prompt
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


async def solve_task(conn, config, task_row, attempt, arc_dir):
    """Solve a single task via subagent. Returns (task_id, correct, cell_accuracy, error)."""
    task_id = task_row["task_id"]
    arc_name = task_row["arc_name"]
    source = task_row["source"]

    try:
        arc_task = load_arc_task(arc_dir, arc_name, source)
    except FileNotFoundError as e:
        return task_id, None, None, str(e)

    expected = arc_task["test"][0]["output"]

    # Build prompt
    if attempt == 1:
        prompt = build_solve_prompt(arc_task)
    else:
        # For retry, get previous attempt's reasoning and grid
        prev = get_best_solve_trial(conn, task_id)
        if prev and prev["predicted_grid"]:
            try:
                prev_grid = json.loads(prev["predicted_grid"])
                prev_grid_text = grid_to_text(prev_grid)
            except (json.JSONDecodeError, KeyError):
                prev_grid_text = "(could not render previous grid)"
            prev_reasoning = prev["reasoning"] or "(no reasoning captured)"
            prompt = build_solve_retry_prompt(arc_task, prev_reasoning, prev_grid_text)
        else:
            prompt = build_solve_prompt(arc_task)

    # Insert trial row
    prompt_store = prompt[:10000]  # truncate for storage
    trial_id = insert_solve_trial(conn, task_id, attempt, prompt_store)

    # Run subagent
    output, elapsed, error = await run_subagent(prompt)

    if error:
        update_solve_trial(conn, trial_id, None, None, None, None, error=error)
        return task_id, None, None, error

    # Parse response
    grid, reasoning, parse_error = parse_response_grid(output)

    if grid is None:
        update_solve_trial(conn, trial_id, output, None, 0, 0.0,
                          error=parse_error)
        return task_id, False, 0.0, parse_error

    # Compare
    correct, cell_accuracy = compare_grids(grid, expected)
    grid_json = json.dumps(grid)

    update_solve_trial(conn, trial_id, output, grid_json, int(correct),
                      cell_accuracy)

    return task_id, correct, cell_accuracy, None


async def solve_batch(conn, config, tasks, attempt, arc_dir, batch_size):
    """Solve a batch of tasks with limited concurrency."""
    semaphore = asyncio.Semaphore(batch_size)
    results = []

    async def bounded_solve(task_row):
        async with semaphore:
            return await solve_task(conn, config, task_row, attempt, arc_dir)

    coros = [bounded_solve(t) for t in tasks]

    total = len(tasks)
    correct_count = 0
    done_count = 0

    for coro in asyncio.as_completed(coros):
        task_id, correct, cell_acc, error = await coro
        done_count += 1

        if correct:
            correct_count += 1
            status = "CORRECT"
        elif error:
            status = f"ERROR: {error[:60]}"
        else:
            acc_str = f"{cell_acc:.0%}" if cell_acc is not None else "?"
            status = f"WRONG ({acc_str} cells)"

        print(f"  [{done_count}/{total}] task {task_id}: {status}")
        results.append((task_id, correct, cell_acc, error))

    return results


@click.group()
def cli():
    """Phase 1: Solve ARC-AGI2 tasks via Claude Code subagents."""
    pass


@cli.command()
@click.option("--split", type=click.Choice(["training", "evaluation"]),
              default="training")
@click.option("--batch-size", default=10, help="Max concurrent subagents")
@click.option("--limit", default=None, type=int, help="Max tasks to process")
def solve(split, batch_size, limit):
    """Solve tasks (attempt 1)."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    # Get tasks that don't have an attempt-1 trial yet
    existing = {r["task_id"] for r in conn.execute(
        "SELECT DISTINCT task_id FROM solve_trials WHERE attempt=1"
    ).fetchall()}

    all_tasks = get_all_tasks(conn, source=split)
    tasks = [t for t in all_tasks if t["task_id"] not in existing]

    if limit:
        tasks = tasks[:limit]

    if not tasks:
        click.echo(f"No remaining tasks to solve for {split}")
        return

    click.echo(f"Solving {len(tasks)} {split} tasks (batch_size={batch_size})")
    results = asyncio.run(solve_batch(conn, config, tasks, 1, arc_dir, batch_size))

    correct = sum(1 for _, c, _, _ in results if c)
    errors = sum(1 for _, _, _, e in results if e)
    click.echo(f"\nResults: {correct}/{len(results)} correct, {errors} errors")
    conn.close()


@cli.command()
@click.option("--batch-size", default=10)
@click.option("--limit", default=None, type=int)
def retry(batch_size, limit):
    """Retry failed tasks (attempt 2)."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    # Get tasks that failed attempt 1 and don't have attempt 2
    unsolved = {r["task_id"] for r in get_unsolved_tasks(conn)}
    has_attempt_2 = {r["task_id"] for r in conn.execute(
        "SELECT DISTINCT task_id FROM solve_trials WHERE attempt=2"
    ).fetchall()}

    task_ids = unsolved - has_attempt_2
    tasks = [r for r in get_all_tasks(conn) if r["task_id"] in task_ids]

    if limit:
        tasks = tasks[:limit]

    if not tasks:
        click.echo("No tasks to retry")
        return

    click.echo(f"Retrying {len(tasks)} failed tasks (batch_size={batch_size})")
    results = asyncio.run(solve_batch(conn, config, tasks, 2, arc_dir, batch_size))

    correct = sum(1 for _, c, _, _ in results if c)
    click.echo(f"\nRetry results: {correct}/{len(results)} newly correct")
    conn.close()


@cli.command()
def report():
    """Show solve statistics."""
    conn = init_db()

    total_tasks = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    total_trials = conn.execute("SELECT COUNT(*) FROM solve_trials").fetchone()[0]

    attempt1 = conn.execute(
        "SELECT COUNT(*) FROM solve_trials WHERE attempt=1"
    ).fetchone()[0]
    attempt1_correct = conn.execute(
        "SELECT COUNT(*) FROM solve_trials WHERE attempt=1 AND correct=1"
    ).fetchone()[0]

    attempt2 = conn.execute(
        "SELECT COUNT(*) FROM solve_trials WHERE attempt=2"
    ).fetchone()[0]
    attempt2_correct = conn.execute(
        "SELECT COUNT(*) FROM solve_trials WHERE attempt=2 AND correct=1"
    ).fetchone()[0]

    solved = len(get_solved_tasks(conn))
    errors = conn.execute(
        "SELECT COUNT(*) FROM solve_trials WHERE error IS NOT NULL"
    ).fetchone()[0]

    click.echo(f"Tasks: {total_tasks} total")
    click.echo(f"Trials: {total_trials} total")
    click.echo(f"  Attempt 1: {attempt1_correct}/{attempt1} correct "
               f"({attempt1_correct/attempt1*100:.1f}%)" if attempt1 else "  Attempt 1: 0")
    if attempt2:
        click.echo(f"  Attempt 2: {attempt2_correct}/{attempt2} correct "
                   f"({attempt2_correct/attempt2*100:.1f}%)")
    click.echo(f"  Errors: {errors}")
    click.echo(f"Solved (any attempt): {solved}/{total_tasks} "
               f"({solved/total_tasks*100:.1f}%)" if total_tasks else "")

    # Per-split breakdown
    for source in ["training", "evaluation"]:
        n = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE source=?", (source,)
        ).fetchone()[0]
        s = conn.execute(
            """SELECT COUNT(DISTINCT st.task_id) FROM solve_trials st
               JOIN tasks t ON st.task_id = t.task_id
               WHERE st.correct=1 AND t.source=?""",
            (source,),
        ).fetchone()[0]
        if n:
            click.echo(f"  {source}: {s}/{n} ({s/n*100:.1f}%)")

    conn.close()


@cli.command("dry-run")
@click.option("--count", default=3, help="Number of tasks to test")
def dry_run(count):
    """Test on a few tasks to verify the pipeline works end-to-end."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    tasks = get_all_tasks(conn, source="training")[:count]

    click.echo(f"Dry run: solving {len(tasks)} tasks sequentially")
    click.echo()

    for task_row in tasks:
        task_id = task_row["task_id"]
        arc_name = task_row["arc_name"]
        click.echo(f"--- Task {task_id} ({arc_name}) ---")

        arc_task = load_arc_task(arc_dir, arc_name, "training")
        expected = arc_task["test"][0]["output"]
        click.echo(f"  Train examples: {len(arc_task['train'])}")
        click.echo(f"  Expected output: {len(expected)}x{len(expected[0])}")

        prompt = build_solve_prompt(arc_task)
        click.echo(f"  Prompt length: {len(prompt)} chars")

        click.echo("  Running subagent...")
        output, elapsed, error = asyncio.run(run_subagent(prompt, timeout_seconds=300))

        if error:
            click.echo(f"  ERROR: {error}")
            # Store error trial
            trial_id = insert_solve_trial(conn, task_id, 1, prompt[:10000])
            update_solve_trial(conn, trial_id, None, None, None, None, error=error)
            continue

        click.echo(f"  Response: {len(output)} chars in {elapsed:.1f}s")

        # Parse
        grid, reasoning, parse_error = parse_response_grid(output)
        if grid is None:
            click.echo(f"  PARSE ERROR: {parse_error}")
            click.echo(f"  Response tail: ...{output[-200:]}")
            trial_id = insert_solve_trial(conn, task_id, 1, prompt[:10000])
            update_solve_trial(conn, trial_id, output, None, 0, 0.0,
                              error=parse_error)
            continue

        correct, cell_acc = compare_grids(grid, expected)
        grid_json = json.dumps(grid)

        status = "CORRECT" if correct else f"WRONG ({cell_acc:.0%} cells)"
        click.echo(f"  Result: {status}")
        click.echo(f"  Predicted: {len(grid)}x{len(grid[0])}")

        # Store
        trial_id = insert_solve_trial(conn, task_id, 1, prompt[:10000])
        update_solve_trial(conn, trial_id, output, grid_json,
                          int(correct), cell_acc)
        click.echo(f"  Stored as trial_id={trial_id}")
        click.echo()

    # Verify DB
    click.echo("=== DB verification ===")
    trials = conn.execute("SELECT * FROM solve_trials ORDER BY trial_id").fetchall()
    for t in trials:
        status = "CORRECT" if t["correct"] else "WRONG" if t["correct"] == 0 else "ERROR"
        has_reasoning = "yes" if t["reasoning"] else "no"
        click.echo(f"  trial {t['trial_id']}: task {t['task_id']} attempt {t['attempt']} "
                   f"→ {status}  reasoning={has_reasoning}  "
                   f"cell_acc={t['cell_accuracy']}")

    conn.close()


if __name__ == "__main__":
    cli()
