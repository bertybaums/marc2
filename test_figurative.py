"""Phase 7: Test figurative descriptions and verify the MARC property.

Tests each figurative description at k=0 (figurative alone) and k=1,...,num_train
(figurative + k examples) on subject models via MindRouter.

Usage:
    python test_figurative.py test --model gpt-oss-120b --concurrency 24
    python test_figurative.py verify-marc --model gpt-oss-120b
    python test_figurative.py report --model gpt-oss-120b
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from db import (
    init_db, get_task, get_figurative,
    insert_figurative_trial, update_figurative_response,
    update_figurative_evaluation, get_subsets,
)
from grids import parse_response_grid, compare_grids
from models import call_llm_two_pass, configure_rate_limit_from_config
from prompts import build_prompt, build_extraction_messages
from tasks import load_config
from utils import find_model_config, get_extraction_model_config, load_arc, serialize_prompt


def _run_single_figurative_trial(model_config, extraction_config, config, db_path,
                                  fig_id, tid, k, model_name):
    """Run a single figurative trial. Thread-safe: opens its own DB connection."""
    conn = init_db(db_path)
    try:
        task = get_task(conn, tid)
        fig = conn.execute(
            "SELECT * FROM figurative_descriptions WHERE fig_id=?", (fig_id,)
        ).fetchone()

        arc_task = load_arc(config, task["arc_name"])
        desc = conn.execute(
            "SELECT * FROM descriptions WHERE task_id=?", (tid,)
        ).fetchone()

        if k == 0:
            condition = "figurative_only"
            messages = build_prompt(condition, desc, arc_task, fig_row=fig)
        else:
            condition = "figurative_with_examples"
            messages = build_prompt(condition, desc, arc_task, num_examples=k, fig_row=fig)

        prompt_text = serialize_prompt(messages)
        trial_id = insert_figurative_trial(conn, fig_id, tid, model_name, k, 1, prompt_text)

        # Skip if already done
        row = conn.execute(
            "SELECT response_text, error FROM figurative_trials WHERE trial_id=?",
            (trial_id,)
        ).fetchone()
        if row["response_text"] or row["error"]:
            return ("skip", tid, k, "")

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

        update_figurative_response(conn, trial_id, combined_raw, extracted, total_ms)

        if predicted_grid is not None:
            expected = arc_task["test"][0]["output"]
            correct, cell_acc = compare_grids(predicted_grid, expected)
            update_figurative_evaluation(
                conn, trial_id, json.dumps(predicted_grid), reasoning,
                1 if correct else 0, cell_acc,
            )
            status = "CORRECT" if correct else f"wrong (cell_acc={cell_acc:.2f})"
        else:
            update_figurative_evaluation(conn, trial_id, None, reasoning, 0, 0.0)
            status = f"parse_error: {parse_error}"

        retries = f" (retries={attempt})" if attempt > 0 else ""
        return ("ok", tid, k, f"{status} ({total_ms}ms){retries}")

    except Exception as e:
        try:
            update_figurative_response(conn, trial_id, None, None, 0, error=str(e))
        except Exception:
            pass
        return ("error", tid, k, str(e))
    finally:
        conn.close()


@click.group()
def cli():
    """Phase 7: Test figurative descriptions for the MARC property."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True)
@click.option("--concurrency", default=24, type=int)
@click.option("--limit", default=None, type=int, help="Limit to first N tasks")
@click.option("--dry-run", is_flag=True)
def test(config_path, db_path, model_name, concurrency, limit, dry_run):
    """Test figurative descriptions across example counts (k=0,1,...,num_train)."""
    config = load_config(config_path)
    configure_rate_limit_from_config(config)
    model_config = find_model_config(config, model_name)
    extraction_config = get_extraction_model_config(config, model_name)

    conn = init_db(db_path)

    # Get all figurative descriptions
    figs = conn.execute(
        "SELECT * FROM figurative_descriptions ORDER BY task_id"
    ).fetchall()

    trial_plan = []
    for fig in figs:
        tid = fig["task_id"]
        task = get_task(conn, tid)
        for k in range(0, task["num_train"] + 1):
            trial_plan.append((tid, fig["fig_id"], k))

    if limit:
        seen_tasks = []
        limited_plan = []
        for tid, fig_id, k in trial_plan:
            if tid not in seen_tasks:
                seen_tasks.append(tid)
            if len(seen_tasks) <= limit:
                limited_plan.append((tid, fig_id, k))
        trial_plan = limited_plan

    click.echo(f"Testing {len(trial_plan)} trials across {len(figs)} figurative descriptions")
    click.echo(f"  Model: {model_name} | Concurrency: {concurrency}")

    if dry_run:
        for tid, fig_id, k in trial_plan[:20]:
            click.echo(f"  task={tid} fig={fig_id} k={k}")
        if len(trial_plan) > 20:
            click.echo(f"  ... and {len(trial_plan) - 20} more")
        conn.close()
        return

    conn.close()

    completed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_single_figurative_trial, model_config, extraction_config,
                config, db_path, fig_id, tid, k, model_name
            ): (tid, fig_id, k)
            for tid, fig_id, k in trial_plan
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            result_type, tid, k, msg = future.result()
            if result_type == "skip":
                skipped += 1
            elif result_type == "ok":
                completed += 1
                click.echo(f"[{i}/{total}] task={tid} k={k} ... {msg}")
            else:
                errors += 1
                click.echo(f"[{i}/{total}] task={tid} k={k} ... ERROR: {msg}")
            if i % 100 == 0:
                click.echo(f"  --- progress: {completed} done, {skipped} skipped, {errors} errors ---")

    click.echo(f"\nDone. completed={completed} skipped={skipped} errors={errors}")


@cli.command("verify-marc")
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True)
def verify_marc(db_path, model_name):
    """Check which tasks satisfy the MARC property.

    MARC property:
    1. Examples alone fail (baseline examples_only)
    2. Figurative alone fails (k=0)
    3. Figurative + sufficient examples succeeds (some k>0)
    """
    conn = init_db(db_path)

    figs = conn.execute(
        "SELECT * FROM figurative_descriptions ORDER BY task_id"
    ).fetchall()

    marc_puzzles = []
    non_marc = []

    for fig in figs:
        tid = fig["task_id"]

        # Condition 1: examples alone must fail
        ex_only = conn.execute(
            """SELECT MAX(correct) as best FROM baseline_trials
               WHERE task_id=? AND model_name=? AND condition='examples_only'""",
            (tid, model_name),
        ).fetchone()
        if ex_only and ex_only["best"] == 1:
            non_marc.append((tid, "examples_alone_succeeds"))
            continue

        # Condition 2: figurative alone must fail
        fig_only = conn.execute(
            """SELECT correct FROM figurative_trials
               WHERE fig_id=? AND model_name=? AND num_examples=0""",
            (fig["fig_id"], model_name),
        ).fetchone()
        if fig_only and fig_only["correct"] == 1:
            non_marc.append((tid, "figurative_alone_succeeds"))
            continue

        # Condition 3: figurative + examples must succeed for some k
        fig_success = conn.execute(
            """SELECT MIN(num_examples) as min_k FROM figurative_trials
               WHERE fig_id=? AND model_name=? AND num_examples>0 AND correct=1""",
            (fig["fig_id"], model_name),
        ).fetchone()
        if fig_success and fig_success["min_k"] is not None:
            marc_puzzles.append((tid, fig_success["min_k"], fig["metaphor"]))
        else:
            non_marc.append((tid, "figurative_never_succeeds"))

    click.echo(f"\n=== MARC Property Verification ({model_name}) ===\n")
    click.echo(f"Valid MARC puzzles: {len(marc_puzzles)}")
    for tid, min_k, metaphor in marc_puzzles[:20]:
        click.echo(f"  task={tid} min_k={min_k}: \"{metaphor[:70]}\"")
    if len(marc_puzzles) > 20:
        click.echo(f"  ... and {len(marc_puzzles) - 20} more")

    click.echo(f"\nNon-MARC: {len(non_marc)}")
    reasons = {}
    for tid, reason in non_marc:
        reasons[reason] = reasons.get(reason, 0) + 1
    for reason, count in sorted(reasons.items()):
        click.echo(f"  {reason}: {count}")

    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True)
def report(db_path, model_name):
    """Show figurative trial statistics."""
    conn = init_db(db_path)

    total = conn.execute(
        "SELECT COUNT(*) FROM figurative_trials WHERE model_name=?",
        (model_name,),
    ).fetchone()[0]

    k0_total = conn.execute(
        "SELECT COUNT(*) FROM figurative_trials WHERE model_name=? AND num_examples=0",
        (model_name,),
    ).fetchone()[0]
    k0_correct = conn.execute(
        "SELECT COUNT(*) FROM figurative_trials WHERE model_name=? AND num_examples=0 AND correct=1",
        (model_name,),
    ).fetchone()[0]

    kn_total = conn.execute(
        "SELECT COUNT(*) FROM figurative_trials WHERE model_name=? AND num_examples>0",
        (model_name,),
    ).fetchone()[0]
    kn_correct = conn.execute(
        "SELECT COUNT(*) FROM figurative_trials WHERE model_name=? AND num_examples>0 AND correct=1",
        (model_name,),
    ).fetchone()[0]

    errors = conn.execute(
        "SELECT COUNT(*) FROM figurative_trials WHERE model_name=? AND error IS NOT NULL",
        (model_name,),
    ).fetchone()[0]

    click.echo(f"Figurative trials ({model_name}): {total}")
    if k0_total:
        click.echo(f"  k=0 (figurative only): {k0_correct}/{k0_total} ({k0_correct/k0_total*100:.1f}%)")
    if kn_total:
        click.echo(f"  k>0 (figurative+examples): {kn_correct}/{kn_total} ({kn_correct/kn_total*100:.1f}%)")
    click.echo(f"  Errors: {errors}")

    conn.close()


if __name__ == "__main__":
    cli()
