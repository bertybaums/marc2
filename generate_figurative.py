"""Phase 6: Generate figurative descriptions for MARC-eligible tasks.

For tasks classified as language_sufficient or both_required, Claude Code
subagents generate metaphorical see/do/grid descriptions from the validated
literal descriptions.

Usage:
    python generate_figurative.py generate --batch-size 10
    python generate_figurative.py report
    python generate_figurative.py dry-run
"""

import asyncio
import json
import time

import click

from db import (
    init_db, get_subsets, get_description, get_task, get_figurative,
    insert_figurative,
)
from prompts import build_figurative_generation
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


def parse_figurative(output):
    """Parse subagent output to extract figurative description.

    Expects JSON with metaphor, see, do, grid.
    Returns (metaphor, see, do, grid, error).
    """
    if not output:
        return None, None, None, None, "Empty response"

    text = output.strip()

    # Strip markdown code blocks
    json_text = text
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

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                data = json.loads(text[brace_start:brace_end + 1])
            except json.JSONDecodeError:
                return None, None, None, None, "Could not parse JSON"
        else:
            return None, None, None, None, "No JSON found"

    metaphor = data.get("metaphor", "")
    see = data.get("see", "")
    do = data.get("do", "")
    grid = data.get("grid", "")

    if not metaphor or not do:
        return None, None, None, None, f"Missing fields: metaphor={bool(metaphor)}, do={bool(do)}"

    return metaphor, see, do, grid, None


def jaccard_similarity(text1, text2):
    """Word-level Jaccard similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


async def generate_task(conn, config, task_id, arc_dir):
    """Generate a figurative description for one task.

    Returns (task_id, success, error).
    """
    desc = get_description(conn, task_id)
    if not desc:
        return task_id, False, "No description"

    task_row = get_task(conn, task_id)
    try:
        arc_task = load_arc_task(arc_dir, task_row["arc_name"], task_row["source"])
    except FileNotFoundError as e:
        return task_id, False, str(e)

    prompt = build_figurative_generation(desc, arc_task)
    output, elapsed, error = await run_subagent(prompt)

    if error:
        return task_id, False, error

    metaphor, fig_see, fig_do, fig_grid, parse_error = parse_figurative(output)
    if parse_error:
        return task_id, False, parse_error

    # Quality check: Jaccard similarity to literal do_description
    sim = jaccard_similarity(fig_do, desc["do_description"])
    if sim > 0.6:
        return task_id, False, f"Too similar to literal (Jaccard={sim:.2f})"

    insert_figurative(
        conn, task_id, "claude-subagent", metaphor, fig_see, fig_do, fig_grid,
        generation_prompt=prompt[:5000], raw_generation=output,
    )

    return task_id, True, None


async def generate_batch(conn, config, task_ids, arc_dir, batch_size):
    """Generate figurative descriptions for a batch of tasks."""
    semaphore = asyncio.Semaphore(batch_size)

    async def bounded_generate(task_id):
        async with semaphore:
            return await generate_task(conn, config, task_id, arc_dir)

    coros = [bounded_generate(tid) for tid in task_ids]

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
    """Phase 6: Generate figurative descriptions."""
    pass


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", default="gpt-oss-120b",
              help="Subject model whose subsets to use")
@click.option("--batch-size", default=10, help="Max concurrent subagents")
@click.option("--limit", default=None, type=int, help="Max tasks to process")
def generate(db_path, model_name, batch_size, limit):
    """Generate figurative descriptions for MARC-eligible tasks."""
    config = load_config()
    conn = init_db(db_path)
    arc_dir = get_arc_dir(config)

    # Get MARC-eligible task_ids
    subsets = get_subsets(conn, model_name=model_name)
    eligible = {s["task_id"] for s in subsets
                if s["subset"] in ("language_sufficient", "both_required")}

    # Skip tasks that already have figurative descriptions
    has_fig = {r["task_id"] for r in conn.execute(
        "SELECT DISTINCT task_id FROM figurative_descriptions"
    ).fetchall()}

    task_ids = sorted(eligible - has_fig)

    if limit:
        task_ids = task_ids[:limit]

    if not task_ids:
        click.echo("No tasks to generate figurative descriptions for")
        return

    click.echo(f"Generating figurative descriptions for {len(task_ids)} tasks "
               f"(batch_size={batch_size})")
    success, errors = asyncio.run(
        generate_batch(conn, config, task_ids, arc_dir, batch_size)
    )

    click.echo(f"\nResults: {success} generated, {errors} errors")
    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
def report(db_path):
    """Show figurative generation statistics."""
    conn = init_db(db_path)

    total = conn.execute("SELECT COUNT(*) FROM figurative_descriptions").fetchone()[0]
    click.echo(f"Figurative descriptions: {total}")

    if total > 0:
        avg_meta = conn.execute(
            "SELECT AVG(length(metaphor)) FROM figurative_descriptions"
        ).fetchone()[0]
        avg_do = conn.execute(
            "SELECT AVG(length(figurative_do)) FROM figurative_descriptions"
        ).fetchone()[0]
        click.echo(f"  Avg metaphor length: {avg_meta:.0f} chars")
        click.echo(f"  Avg figurative_do length: {avg_do:.0f} chars")

    conn.close()


@cli.command("dry-run")
@click.option("--count", default=3)
def dry_run(count):
    """Test figurative generation on a few tasks."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    subsets = get_subsets(conn, model_name="gpt-oss-120b")
    eligible = [s["task_id"] for s in subsets
                if s["subset"] in ("language_sufficient", "both_required")][:count]

    click.echo(f"Dry run: generating figurative descriptions for {len(eligible)} tasks")
    click.echo()

    for task_id in eligible:
        desc = get_description(conn, task_id)
        task_row = get_task(conn, task_id)

        click.echo(f"--- Task {task_id} ({task_row['arc_name']}) ---")
        click.echo(f"  Literal do: {desc['do_description'][:80]}...")

        arc_task = load_arc_task(arc_dir, task_row['arc_name'], task_row['source'])
        prompt = build_figurative_generation(desc, arc_task)
        click.echo(f"  Prompt: {len(prompt)} chars")

        click.echo("  Running subagent...")
        output, elapsed, error = asyncio.run(run_subagent(prompt, timeout_seconds=300))

        if error:
            click.echo(f"  ERROR: {error}")
            continue

        click.echo(f"  Response: {len(output)} chars in {elapsed:.1f}s")

        metaphor, fig_see, fig_do, fig_grid, parse_error = parse_figurative(output)
        if parse_error:
            click.echo(f"  PARSE ERROR: {parse_error}")
            continue

        sim = jaccard_similarity(fig_do, desc["do_description"])
        click.echo(f"  Metaphor: {metaphor}")
        click.echo(f"  Fig see: {fig_see[:80]}...")
        click.echo(f"  Fig do:  {fig_do[:80]}...")
        click.echo(f"  Jaccard similarity: {sim:.2f} {'(TOO SIMILAR)' if sim > 0.6 else '(OK)'}")

        if sim <= 0.6:
            insert_figurative(
                conn, task_id, "claude-subagent", metaphor, fig_see, fig_do, fig_grid,
                generation_prompt=prompt[:5000], raw_generation=output,
            )
            click.echo(f"  Stored")
        click.echo()

    conn.close()


if __name__ == "__main__":
    cli()
