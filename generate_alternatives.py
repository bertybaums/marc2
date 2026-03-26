"""Phase 8: Generate domain-diverse alternative figurative clues for MARC-verified puzzles.

For each MARC-verified puzzle, generates 15 alternative metaphors across diverse
source domains, with opacity guidance to maximize MARC yield.

Round 1: Generate 15 alternatives per puzzle (Claude subagents)
Round 2 (optional): Generate more for puzzles below target MARC-valid count

Usage:
    python generate_alternatives.py generate --batch-size 10
    python generate_alternatives.py verify --model gpt-oss-120b --concurrency 24
    python generate_alternatives.py report --model gpt-oss-120b
    python generate_alternatives.py dry-run
"""

import asyncio
import json
import time

import click

from db import (
    init_db, get_task, get_description, get_figurative, get_all_variants,
    insert_figurative, insert_figurative_trial, update_figurative_response,
    update_figurative_evaluation,
)
from grids import (
    COLOR_KEY, format_examples_text, parse_response_grid, compare_grids,
)
from models import call_llm_two_pass
from prompts import build_prompt, build_extraction_messages
from tasks import load_config, get_arc_dir, load_arc_task
from utils import find_model_config, get_extraction_model_config, load_arc, serialize_prompt

from concurrent.futures import ThreadPoolExecutor, as_completed

SOURCE_DOMAINS = [
    "biology", "cooking", "music", "sports", "weather",
    "architecture", "warfare", "theater", "gardening",
    "astronomy", "ocean/sailing", "electronics",
    "mythology", "dance", "geology",
]


def build_alternatives_prompt(desc_row, arc_task, existing_metaphors):
    """Build prompt for generating multiple domain-diverse figurative alternatives.

    Includes opacity guidance to target the MARC sweet spot.
    """
    train = arc_task["train"]

    existing_list = "\n".join(f'- "{m}"' for m in existing_metaphors) if existing_metaphors else "(none yet)"

    domains_str = ", ".join(SOURCE_DOMAINS)

    prompt = f"""\
You are creating metaphorical descriptions for a grid transformation puzzle.
Your goal is to produce figurative clues that hit a specific "sweet spot":

**CRITICAL OPACITY REQUIREMENT:**
- The metaphor must NOT be solvable on its own — it must be ambiguous enough
  that a solver needs to see training examples to decode it
- But it SHOULD be evocative enough that seeing examples creates an "aha!" moment
- Think of it like a riddle: the answer is obvious once you see the examples,
  but impossible to guess from the riddle alone

{COLOR_KEY}

## The Puzzle

{format_examples_text(train)}

## Literal Description (for your reference only — do NOT copy this)

What you see: {desc_row['see_description']}
What to do: {desc_row['do_description']}
Grid details: {desc_row['grid_description']}

## Existing Metaphors (generate DIFFERENT ones)

{existing_list}

## Your Task

Generate 15 metaphorical descriptions, each from a DIFFERENT source domain.
Use these domains: {domains_str}

For each metaphor:
1. Assign imaginative roles to grid elements through the lens of that domain
2. Describe the transformation as a scenario from that domain
3. Keep it evocative but AMBIGUOUS — someone reading just the metaphor should NOT
   be able to solve the puzzle without seeing examples
4. The metaphor should make someone say "Oh, THAT's what it means!" when they
   see the examples

Respond with ONLY this JSON array (no other text):
[
  {{"domain": "biology", "metaphor": "...", "see": "...", "do": "...", "grid": "..."}},
  {{"domain": "cooking", "metaphor": "...", "see": "...", "do": "...", "grid": "..."}},
  ...
]"""

    return prompt


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


def parse_alternatives(output):
    """Parse subagent output to extract array of alternative figurative descriptions.

    Returns (list_of_dicts, error).
    """
    if not output:
        return None, "Empty response"

    text = output.strip()

    # Strip markdown code blocks
    if "```json" in text:
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            pass
    elif "```" in text:
        try:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            pass

    # Try parsing as JSON array
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data, None
    except json.JSONDecodeError:
        pass

    # Try finding array in the text
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    if bracket_start >= 0 and bracket_end > bracket_start:
        try:
            data = json.loads(text[bracket_start:bracket_end + 1])
            if isinstance(data, list):
                return data, None
        except json.JSONDecodeError:
            pass

    return None, "Could not parse JSON array from response"


def get_marc_verified_task_ids(conn, model_name):
    """Get task_ids that have verified MARC property."""
    rows = conn.execute('''
        SELECT DISTINCT ft.task_id
        FROM figurative_trials ft
        JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
        WHERE ft.model_name=?
          AND ft.correct=1 AND ft.num_examples > 0
          AND fd.variant = 'original'
          AND ft.task_id NOT IN (
            SELECT task_id FROM baseline_trials
            WHERE model_name=? AND condition='examples_only' AND correct=1
          )
          AND ft.fig_id NOT IN (
            SELECT fig_id FROM figurative_trials
            WHERE model_name=? AND num_examples=0 AND correct=1
          )
        ORDER BY ft.task_id
    ''', (model_name, model_name, model_name)).fetchall()
    return [r["task_id"] for r in rows]


async def generate_for_task(conn, config, task_id, arc_dir):
    """Generate 15 alternative figurative descriptions for one task.

    Returns (task_id, num_stored, error).
    """
    desc = get_description(conn, task_id)
    if not desc:
        return task_id, 0, "No description"

    task_row = get_task(conn, task_id)
    try:
        arc_task = load_arc_task(arc_dir, task_row["arc_name"], task_row["source"])
    except FileNotFoundError as e:
        return task_id, 0, str(e)

    # Get existing metaphors to avoid duplicates
    existing = get_all_variants(conn, task_id)
    existing_metaphors = [v["metaphor"] for v in existing]

    prompt = build_alternatives_prompt(desc, arc_task, existing_metaphors)
    output, elapsed, error = await run_subagent(prompt, timeout_seconds=120)

    if error:
        return task_id, 0, error

    alternatives, parse_error = parse_alternatives(output)
    if parse_error:
        return task_id, 0, parse_error

    # Store each alternative
    stored = 0
    # Count existing alt variants
    alt_count = sum(1 for v in existing if v["variant"] != "original")

    for alt in alternatives:
        domain = alt.get("domain", "unknown")
        metaphor = alt.get("metaphor", "")
        fig_see = alt.get("see", "")
        fig_do = alt.get("do", "")
        fig_grid = alt.get("grid", "")

        if not metaphor:
            continue

        alt_count += 1
        variant_name = f"alt-{alt_count}"

        fig_id = insert_figurative(
            conn, task_id, "claude-subagent", metaphor, fig_see, fig_do, fig_grid,
            variant=variant_name, source_domain=domain,
        )
        if fig_id:
            stored += 1

    return task_id, stored, None


async def generate_batch(conn, config, task_ids, arc_dir, batch_size):
    """Generate alternatives for a batch of tasks."""
    semaphore = asyncio.Semaphore(batch_size)

    async def bounded_generate(task_id):
        async with semaphore:
            return await generate_for_task(conn, config, task_id, arc_dir)

    coros = [bounded_generate(tid) for tid in task_ids]

    total = len(task_ids)
    total_stored = 0
    error_count = 0
    done_count = 0

    for coro in asyncio.as_completed(coros):
        task_id, stored, error = await coro
        done_count += 1

        if error:
            error_count += 1
            status = f"FAIL: {error[:60]}"
        else:
            total_stored += stored
            status = f"OK ({stored} variants)"

        print(f"  [{done_count}/{total}] task {task_id}: {status}")

    return total_stored, error_count


@click.group()
def cli():
    """Phase 8: Domain-diverse alternative figurative clues."""
    pass


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", default="gpt-oss-120b",
              help="Subject model whose MARC results to use")
@click.option("--batch-size", default=10, help="Max concurrent subagents")
@click.option("--limit", default=None, type=int, help="Max tasks to process")
def generate(db_path, model_name, batch_size, limit):
    """Generate 15 domain-diverse alternatives per MARC-verified puzzle."""
    config = load_config()
    conn = init_db(db_path)
    arc_dir = get_arc_dir(config)

    marc_tasks = get_marc_verified_task_ids(conn, model_name)

    # Skip tasks that already have alternatives
    has_alts = set()
    for tid in marc_tasks:
        variants = get_all_variants(conn, tid)
        alt_count = sum(1 for v in variants if v["variant"] != "original")
        if alt_count >= 10:  # already has enough
            has_alts.add(tid)

    task_ids = [t for t in marc_tasks if t not in has_alts]

    if limit:
        task_ids = task_ids[:limit]

    if not task_ids:
        click.echo("No MARC-verified tasks need alternatives")
        return

    click.echo(f"Generating alternatives for {len(task_ids)} MARC-verified tasks "
               f"(batch_size={batch_size})")
    total_stored, errors = asyncio.run(
        generate_batch(conn, config, task_ids, arc_dir, batch_size)
    )

    click.echo(f"\nResults: {total_stored} alternatives stored, {errors} errors")
    conn.close()


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True)
@click.option("--concurrency", default=24, type=int)
@click.option("--limit", default=None, type=int, help="Limit to first N variants")
@click.option("--dry-run", is_flag=True)
def verify(config_path, db_path, model_name, concurrency, limit, dry_run):
    """Verify MARC property for alternative figurative clues via MindRouter."""
    config = load_config(config_path)
    model_config = find_model_config(config, model_name)
    extraction_config = get_extraction_model_config(config, model_name)

    conn = init_db(db_path)

    # Get alt variants
    alts = conn.execute('''
        SELECT fd.fig_id, fd.task_id, fd.variant, fd.source_domain
        FROM figurative_descriptions fd
        WHERE fd.variant != 'original'
        ORDER BY fd.task_id, fd.variant
    ''').fetchall()

    if limit:
        alts = alts[:limit]

    # Build trial plan: k=0 and k=1,...,num_train for each alt
    trial_plan = []
    for alt in alts:
        task = get_task(conn, alt["task_id"])
        for k in range(0, task["num_train"] + 1):
            trial_plan.append((alt["fig_id"], alt["task_id"], k, alt["variant"]))

    click.echo(f"Verifying {len(alts)} alternatives ({len(trial_plan)} trials)")
    click.echo(f"  Model: {model_name} | Concurrency: {concurrency}")

    if dry_run:
        for fig_id, tid, k, variant in trial_plan[:20]:
            click.echo(f"  task={tid} {variant} k={k}")
        if len(trial_plan) > 20:
            click.echo(f"  ... and {len(trial_plan) - 20} more")
        conn.close()
        return

    conn.close()

    def _run_trial(fig_id, tid, k, variant):
        tconn = init_db(db_path)
        try:
            task = get_task(tconn, tid)
            fig = tconn.execute(
                "SELECT * FROM figurative_descriptions WHERE fig_id=?", (fig_id,)
            ).fetchone()
            desc = get_description(tconn, tid)
            arc_task = load_arc(config, task["arc_name"])

            if k == 0:
                messages = build_prompt("figurative_only", desc, arc_task, fig_row=fig)
            else:
                messages = build_prompt("figurative_with_examples", desc, arc_task,
                                        num_examples=k, fig_row=fig)

            prompt_text = serialize_prompt(messages)
            trial_id = insert_figurative_trial(tconn, fig_id, tid, model_name, k, 1, prompt_text)

            row = tconn.execute(
                "SELECT response_text, error FROM figurative_trials WHERE trial_id=?",
                (trial_id,)
            ).fetchone()
            if row["response_text"] or row["error"]:
                return ("skip", tid, variant, k, "")

            max_attempts = 3
            for attempt in range(max_attempts):
                raw1, reasoning, raw2, extracted, total_ms = call_llm_two_pass(
                    model_config, messages, build_extraction_messages,
                    extraction_model_config=extraction_config
                )
                combined_raw = json.dumps({"pass1": json.loads(raw1) if raw1 else None,
                                           "pass2": json.loads(raw2) if raw2 else None})
                predicted_grid, _, parse_error = parse_response_grid(extracted)
                if predicted_grid is None:
                    predicted_grid, _, _ = parse_response_grid(reasoning)
                if predicted_grid is not None or attempt == max_attempts - 1:
                    break

            update_figurative_response(tconn, trial_id, combined_raw, extracted, total_ms)

            if predicted_grid is not None:
                expected = arc_task["test"][0]["output"]
                correct, cell_acc = compare_grids(predicted_grid, expected)
                update_figurative_evaluation(
                    tconn, trial_id, json.dumps(predicted_grid), reasoning,
                    1 if correct else 0, cell_acc,
                )
                status = "CORRECT" if correct else f"wrong ({cell_acc:.2f})"
            else:
                update_figurative_evaluation(tconn, trial_id, None, reasoning, 0, 0.0)
                status = f"parse_error"

            return ("ok", tid, variant, k, f"{status} ({total_ms}ms)")
        except Exception as e:
            return ("error", tid, variant, k, str(e))
        finally:
            tconn.close()

    completed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_trial, fig_id, tid, k, variant): (tid, variant, k)
            for fig_id, tid, k, variant in trial_plan
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            result_type, tid, variant, k, msg = future.result()
            if result_type == "skip":
                skipped += 1
            elif result_type == "ok":
                completed += 1
                if i % 100 == 0 or "CORRECT" in msg:
                    click.echo(f"[{i}/{total}] task={tid} {variant} k={k} ... {msg}")
            else:
                errors += 1
                click.echo(f"[{i}/{total}] task={tid} {variant} k={k} ... ERROR: {msg}")
            if i % 200 == 0:
                click.echo(f"  --- progress: {completed} done, {skipped} skipped, {errors} errors ---")

    click.echo(f"\nDone. completed={completed} skipped={skipped} errors={errors}")


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", default="gpt-oss-120b")
def report(db_path, model_name):
    """Report on alternative figurative clues and MARC verification."""
    conn = init_db(db_path)

    alts = conn.execute('''
        SELECT fd.task_id, fd.variant, fd.source_domain, fd.metaphor, fd.fig_id
        FROM figurative_descriptions fd
        WHERE fd.variant != 'original'
        ORDER BY fd.task_id, fd.variant
    ''').fetchall()

    click.echo(f"\n=== Alternative Figurative Clues ===\n")
    click.echo(f"Total alternatives: {len(alts)}")

    marc_valid = 0
    marc_invalid_transparent = 0
    marc_invalid_opaque = 0
    untested = 0

    for alt in alts:
        fig_id = alt["fig_id"]
        tid = alt["task_id"]

        trials = conn.execute(
            "SELECT num_examples, correct FROM figurative_trials WHERE fig_id=? AND model_name=?",
            (fig_id, model_name),
        ).fetchall()

        if not trials:
            untested += 1
            continue

        # Condition 2: figurative alone must fail
        fig_only = [t for t in trials if t["num_examples"] == 0 and t["correct"] == 1]
        if fig_only:
            marc_invalid_transparent += 1
            continue

        # Condition 3: figurative + examples must succeed
        fig_ex = [t for t in trials if t["num_examples"] > 0 and t["correct"] == 1]
        if fig_ex:
            marc_valid += 1
        else:
            marc_invalid_opaque += 1

    click.echo(f"MARC-valid: {marc_valid}")
    click.echo(f"Too transparent (figurative alone works): {marc_invalid_transparent}")
    click.echo(f"Too opaque (never works): {marc_invalid_opaque}")
    click.echo(f"Untested: {untested}")

    if marc_valid + marc_invalid_transparent + marc_invalid_opaque > 0:
        total_tested = marc_valid + marc_invalid_transparent + marc_invalid_opaque
        click.echo(f"\nMARC yield rate: {marc_valid}/{total_tested} ({marc_valid/total_tested*100:.1f}%)")

    # Domain distribution
    domains = {}
    for alt in alts:
        d = alt["source_domain"] or "untagged"
        domains[d] = domains.get(d, 0) + 1

    if domains:
        click.echo(f"\nSource domains ({len(domains)}):")
        for domain, count in sorted(domains.items(), key=lambda x: -x[1]):
            click.echo(f"  {domain}: {count}")

    # Per-task coverage
    by_task = {}
    for alt in alts:
        tid = alt["task_id"]
        by_task[tid] = by_task.get(tid, 0) + 1

    if by_task:
        click.echo(f"\nPer-task: {len(by_task)} tasks, "
                   f"avg {sum(by_task.values())/len(by_task):.1f} alts/task, "
                   f"min={min(by_task.values())}, max={max(by_task.values())}")

    conn.close()


@cli.command("dry-run")
@click.option("--count", default=2)
def dry_run(count):
    """Test alternative generation on a few tasks."""
    config = load_config()
    conn = init_db()
    arc_dir = get_arc_dir(config)

    marc_tasks = get_marc_verified_task_ids(conn, "gpt-oss-120b")[:count]

    click.echo(f"Dry run: generating alternatives for {len(marc_tasks)} tasks")
    click.echo()

    for task_id in marc_tasks:
        task_row = get_task(conn, task_id)
        desc = get_description(conn, task_id)
        existing = get_all_variants(conn, task_id)

        click.echo(f"--- Task {task_id} ({task_row['arc_name']}) ---")
        click.echo(f"  Original metaphor: {existing[0]['metaphor'][:70]}...")

        arc_task = load_arc_task(arc_dir, task_row['arc_name'], task_row['source'])
        existing_metaphors = [v["metaphor"] for v in existing]
        prompt = build_alternatives_prompt(desc, arc_task, existing_metaphors)
        click.echo(f"  Prompt: {len(prompt)} chars")

        click.echo("  Running subagent...")
        output, elapsed, error = asyncio.run(run_subagent(prompt, timeout_seconds=120))

        if error:
            click.echo(f"  ERROR: {error}")
            continue

        click.echo(f"  Response: {len(output)} chars in {elapsed:.1f}s")

        alternatives, parse_error = parse_alternatives(output)
        if parse_error:
            click.echo(f"  PARSE ERROR: {parse_error}")
            continue

        click.echo(f"  Generated {len(alternatives)} alternatives:")
        for alt in alternatives:
            domain = alt.get("domain", "?")
            metaphor = alt.get("metaphor", "")
            click.echo(f"    [{domain}] {metaphor[:70]}")

        click.echo()

    conn.close()


if __name__ == "__main__":
    cli()
