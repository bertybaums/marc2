"""Phase 3: Validate descriptions via fresh Claude Code subagents.

A fresh subagent receives only the see/do/grid description + test input (no
training examples, no prior context). If it produces the correct output, the
description is validated as language-complete.

Failed descriptions can be revised and re-validated (up to 2 revisions).

Usage:
    python validate.py validate --batch-size 10
    python validate.py revise --batch-size 10
    python validate.py report
    python validate.py dry-run
"""

import asyncio
import json
import time

import click

from db import (
    init_db, get_description, get_unvalidated_descriptions,
    update_description_validation, insert_description, increment_revision,
    get_best_solve_trial,
)
from distill import parse_description
from grids import compare_grids, grid_to_text, parse_response_grid
from prompts import build_validate_prompt, build_revise_prompt
from tasks import load_config, get_arc_dir, load_arc_task


async def run_subagent(prompt, timeout_seconds=600):
    """Run a Claude Code subagent and return its text output."""
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


async def validate_task(conn, config, desc_row, arc_dir):
    """Validate a single description. Returns (task_id, passed, error)."""
    task_id = desc_row["task_id"]

    task_row = conn.execute(
        "SELECT * FROM tasks WHERE task_id=?", (task_id,)
    ).fetchone()
    arc_name = task_row["arc_name"]
    source = task_row["source"]

    try:
        arc_task = load_arc_task(arc_dir, arc_name, source)
    except FileNotFoundError as e:
        return task_id, None, str(e)

    expected = arc_task["test"][0]["output"]
    test_input = arc_task["test"][0]["input"]

    # Build prompt — description + test input only, NO training examples
    prompt = build_validate_prompt(
        desc_row["see_description"],
        desc_row["do_description"],
        desc_row["grid_description"],
        test_input,
    )

    output, elapsed, error = await run_subagent(prompt)

    if error:
        return task_id, None, error

    # Parse the predicted grid
    grid, reasoning, parse_error = parse_response_grid(output)

    if grid is None:
        update_description_validation(conn, task_id, -1, 0)
        return task_id, False, parse_error

    correct, cell_accuracy = compare_grids(grid, expected)
    update_description_validation(conn, task_id, 1 if correct else -1, int(correct))

    return task_id, correct, None


async def validate_batch(conn, config, desc_rows, arc_dir, batch_size):
    """Validate a batch of descriptions with limited concurrency."""
    semaphore = asyncio.Semaphore(batch_size)

    async def bounded_validate(desc_row):
        async with semaphore:
            return await validate_task(conn, config, desc_row, arc_dir)

    coros = [bounded_validate(d) for d in desc_rows]

    total = len(desc_rows)
    passed = 0
    failed = 0
    errors = 0
    done_count = 0

    for coro in asyncio.as_completed(coros):
        task_id, correct, error = await coro
        done_count += 1

        if correct is True:
            passed += 1
            status = "PASS"
        elif correct is False:
            failed += 1
            status = f"FAIL"
        else:
            errors += 1
            status = f"ERROR: {error[:60]}"

        print(f"  [{done_count}/{total}] task {task_id}: {status}")

    return passed, failed, errors


async def revise_task(conn, config, desc_row, arc_dir):
    """Revise a failed description and re-validate. Returns (task_id, passed, error)."""
    task_id = desc_row["task_id"]

    task_row = conn.execute(
        "SELECT * FROM tasks WHERE task_id=?", (task_id,)
    ).fetchone()

    try:
        arc_task = load_arc_task(arc_dir, task_row["arc_name"], task_row["source"])
    except FileNotFoundError as e:
        return task_id, None, str(e)

    expected = arc_task["test"][0]["output"]
    test_input = arc_task["test"][0]["input"]

    # Get the original reasoning trace
    trial = get_best_solve_trial(conn, task_id)
    reasoning_trace = trial["reasoning"] if trial else ""

    # Build revision prompt — includes what went wrong
    # First, get what the validator produced (re-validate to get wrong output)
    val_prompt = build_validate_prompt(
        desc_row["see_description"],
        desc_row["do_description"],
        desc_row["grid_description"],
        test_input,
    )
    val_output, _, val_error = await run_subagent(val_prompt)

    if val_error:
        return task_id, None, f"Re-validation error: {val_error}"

    wrong_grid, _, _ = parse_response_grid(val_output)
    wrong_text = grid_to_text(wrong_grid) if wrong_grid else "(could not parse grid)"
    correct_text = grid_to_text(expected)

    # Now build revision prompt
    revise_prompt = build_revise_prompt(
        arc_task, reasoning_trace,
        desc_row["see_description"],
        desc_row["do_description"],
        desc_row["grid_description"],
        wrong_text, correct_text,
    )

    output, elapsed, error = await run_subagent(revise_prompt)
    if error:
        return task_id, None, error

    see, do, grid, parse_error = parse_description(output)
    if parse_error:
        return task_id, None, parse_error

    # Update description with revision
    increment_revision(conn, task_id)
    insert_description(conn, task_id, desc_row["source_trial_id"], see, do, grid)

    # Validate the revised description
    new_desc = get_description(conn, task_id)
    return await validate_task(conn, config, new_desc, arc_dir)


async def revise_batch(conn, config, desc_rows, arc_dir, batch_size):
    """Revise and re-validate a batch of failed descriptions."""
    semaphore = asyncio.Semaphore(batch_size)

    async def bounded_revise(desc_row):
        async with semaphore:
            return await revise_task(conn, config, desc_row, arc_dir)

    coros = [bounded_revise(d) for d in desc_rows]

    total = len(desc_rows)
    passed = 0
    failed = 0
    errors = 0
    done_count = 0

    for coro in asyncio.as_completed(coros):
        task_id, correct, error = await coro
        done_count += 1

        if correct is True:
            passed += 1
            status = "REVISED → PASS"
        elif correct is False:
            failed += 1
            status = "REVISED → FAIL"
        else:
            errors += 1
            status = f"ERROR: {error[:60]}"

        print(f"  [{done_count}/{total}] task {task_id}: {status}")

    return passed, failed, errors


@click.group()
def cli():
    """Phase 3: Validate descriptions via fresh subagents."""
    pass


@cli.command()
@click.option("--batch-size", default=10, help="Max concurrent subagents")
@click.option("--limit", default=None, type=int, help="Max tasks to process")
def validate(batch_size, limit):
    """Validate all untested descriptions."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    descs = get_unvalidated_descriptions(conn)
    if limit:
        descs = descs[:limit]

    if not descs:
        click.echo("No descriptions to validate")
        return

    click.echo(f"Validating {len(descs)} descriptions (batch_size={batch_size})")
    passed, failed, errors = asyncio.run(
        validate_batch(conn, config, descs, arc_dir, batch_size)
    )

    click.echo(f"\nResults: {passed} passed, {failed} failed, {errors} errors")
    conn.close()


@cli.command()
@click.option("--batch-size", default=10, help="Max concurrent subagents")
@click.option("--max-revisions", default=2, help="Max revision attempts per description")
@click.option("--limit", default=None, type=int, help="Max tasks to process")
def revise(batch_size, max_revisions, limit):
    """Revise failed descriptions and re-validate."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    failed = conn.execute(
        "SELECT * FROM descriptions WHERE validated=-1 AND revision_count < ?",
        (max_revisions,),
    ).fetchall()

    if limit:
        failed = failed[:limit]

    if not failed:
        click.echo("No descriptions to revise")
        return

    click.echo(f"Revising {len(failed)} failed descriptions (batch_size={batch_size})")
    passed, still_failed, errors = asyncio.run(
        revise_batch(conn, config, failed, arc_dir, batch_size)
    )

    click.echo(f"\nResults: {passed} now pass, {still_failed} still fail, {errors} errors")
    conn.close()


@cli.command()
def report():
    """Show validation statistics."""
    conn = init_db()

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
    revised = conn.execute(
        "SELECT COUNT(*) FROM descriptions WHERE revision_count > 0"
    ).fetchone()[0]

    click.echo(f"Descriptions: {total_desc}")
    click.echo(f"  Validated (passed): {validated} ({validated/total_desc*100:.1f}%)" if total_desc else "")
    click.echo(f"  Failed: {failed}")
    click.echo(f"  Untested: {untested}")
    if revised:
        click.echo(f"  Revised: {revised}")

    conn.close()


@cli.command("dry-run")
@click.option("--count", default=3, help="Number of tasks to test")
def dry_run(count):
    """Test validation on a few tasks."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    descs = conn.execute(
        "SELECT * FROM descriptions ORDER BY task_id LIMIT ?", (count,)
    ).fetchall()

    click.echo(f"Dry run: validating {len(descs)} descriptions")
    click.echo()

    for desc in descs:
        task_id = desc["task_id"]
        task_row = conn.execute(
            "SELECT * FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()

        click.echo(f"--- Task {task_id} ({task_row['arc_name']}) ---")
        click.echo(f"  See: {desc['see_description'][:80]}...")
        click.echo(f"  Do:  {desc['do_description'][:80]}...")

        arc_task = load_arc_task(arc_dir, task_row['arc_name'], task_row['source'])
        expected = arc_task["test"][0]["output"]
        test_input = arc_task["test"][0]["input"]

        prompt = build_validate_prompt(
            desc["see_description"],
            desc["do_description"],
            desc["grid_description"],
            test_input,
        )
        click.echo(f"  Prompt: {len(prompt)} chars (no training examples)")

        click.echo("  Running fresh subagent...")
        output, elapsed, error = asyncio.run(run_subagent(prompt, timeout_seconds=300))

        if error:
            click.echo(f"  ERROR: {error}")
            continue

        click.echo(f"  Response: {len(output)} chars in {elapsed:.1f}s")

        grid, reasoning, parse_error = parse_response_grid(output)
        if grid is None:
            click.echo(f"  PARSE ERROR: {parse_error}")
            update_description_validation(conn, task_id, -1, 0)
            continue

        correct, cell_acc = compare_grids(grid, expected)
        status = "PASS" if correct else f"FAIL ({cell_acc:.0%} cells)"
        click.echo(f"  Result: {status}")
        click.echo(f"  Predicted: {len(grid)}x{len(grid[0])}, Expected: {len(expected)}x{len(expected[0])}")

        update_description_validation(conn, task_id, 1 if correct else -1, int(correct))
        click.echo()

    click.echo("=== Summary ===")
    for row in conn.execute(
        "SELECT validated, COUNT(*) as n FROM descriptions GROUP BY validated"
    ):
        label = {0: "untested", 1: "passed", -1: "failed"}[row["validated"]]
        click.echo(f"  {label}: {row['n']}")

    conn.close()


if __name__ == "__main__":
    cli()
