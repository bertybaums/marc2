"""Phase 4: Baseline 3-condition testing on subject models via MindRouter.

Tests each validated task under three conditions:
  - examples_only: training I/O pairs + test input
  - language_only: validated description + test input
  - both: description + training examples + test input

Uses a two-pass protocol:
  Pass 1: Subject model reasons freely
  Pass 2: Extraction model (gpt-oss-120b) extracts structured JSON

Usage:
    python collect.py --model gpt-oss-120b --concurrency 8
    python collect.py --model gpt-oss-20b --concurrency 8
    python collect.py --model gpt-oss-120b --condition examples_only --concurrency 8
    python collect.py --model gpt-oss-120b --dry-run
    python collect.py report --model gpt-oss-120b
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import click

from db import (
    init_db, get_task, get_description, get_validated_descriptions,
    insert_baseline_trial, update_baseline_response, update_baseline_evaluation,
)
from grids import parse_response_grid, compare_grids
from models import call_llm_two_pass
from prompts import build_prompt, build_extraction_messages
from tasks import load_config
from utils import find_model_config, get_extraction_model_config, load_arc, serialize_prompt


def _run_single_trial(model_config, extraction_config, config, db_path,
                      task_id, cond, num_ex, rep, model_name):
    """Run a single baseline trial. Thread-safe: opens its own DB connection."""
    conn = init_db(db_path)
    try:
        task = get_task(conn, task_id)
        desc = get_description(conn, task_id)
        arc_task = load_arc(config, task["arc_name"])

        messages = build_prompt(cond, desc, arc_task, num_examples=num_ex)
        prompt_text = serialize_prompt(messages)

        trial_id = insert_baseline_trial(
            conn, task_id, model_name, cond, num_ex, rep, prompt_text
        )

        # Skip if already done
        row = conn.execute(
            "SELECT response_text, error FROM baseline_trials WHERE trial_id=?",
            (trial_id,)
        ).fetchone()
        if row["response_text"] or row["error"]:
            return ("skip", task_id, cond, rep, "")

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
                if predicted_grid is not None:
                    parse_error = None

            if predicted_grid is not None or attempt == max_attempts - 1:
                break

        update_baseline_response(conn, trial_id, combined_raw, extracted, total_ms)

        if predicted_grid is not None:
            expected = arc_task["test"][0]["output"]
            correct, cell_acc = compare_grids(predicted_grid, expected)
            update_baseline_evaluation(
                conn, trial_id, json.dumps(predicted_grid), reasoning,
                1 if correct else 0, cell_acc,
            )
            status = "CORRECT" if correct else f"wrong (cell_acc={cell_acc:.2f})"
        else:
            update_baseline_evaluation(conn, trial_id, None, reasoning, 0, 0.0)
            status = f"parse_error: {parse_error}"

        retries = f" (retries={attempt})" if attempt > 0 else ""
        return ("ok", task_id, cond, rep, f"{status} ({total_ms}ms){retries}")

    except Exception as e:
        try:
            update_baseline_response(conn, trial_id, None, None, 0, error=str(e))
        except Exception:
            pass
        return ("error", task_id, cond, rep, str(e))
    finally:
        conn.close()


@click.group()
def cli():
    """Phase 4: Baseline 3-condition testing on subject models."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True, help="Subject model name from config")
@click.option("--condition", "cond_filter", default=None,
              type=click.Choice(["examples_only", "language_only", "both"]))
@click.option("--limit", default=None, type=int, help="Limit to first N tasks")
@click.option("--concurrency", default=8, type=int, help="Parallel API calls")
@click.option("--dry-run", is_flag=True, help="Print trial plan without calling APIs")
def run(config_path, db_path, model_name, cond_filter, limit, concurrency, dry_run):
    """Run baseline collection for a subject model."""
    config = load_config(config_path)
    model_config = find_model_config(config, model_name)
    extraction_config = get_extraction_model_config(config, model_name)

    conditions = [cond_filter] if cond_filter else ["examples_only", "language_only", "both"]

    conn = init_db(db_path)

    # Only test tasks with validated descriptions
    validated = get_validated_descriptions(conn)
    if limit:
        validated = validated[:limit]

    # Build trial plan
    trials = []
    for desc in validated:
        task = get_task(conn, desc["task_id"])
        for cond in conditions:
            num_ex = task["num_train"] if cond != "language_only" else 0
            trials.append((desc["task_id"], cond, num_ex, 1))

    click.echo(f"Model: {model_name} | Conditions: {conditions} | "
               f"Tasks: {len(validated)} | Trials: {len(trials)} | Concurrency: {concurrency}")

    if dry_run:
        for task_id, cond, num_ex, rep in trials[:20]:
            click.echo(f"  task={task_id} cond={cond} examples={num_ex}")
        if len(trials) > 20:
            click.echo(f"  ... and {len(trials) - 20} more")
        conn.close()
        return

    conn.close()

    completed = 0
    skipped = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _run_single_trial, model_config, extraction_config,
                config, db_path, task_id, cond, num_ex, rep, model_name
            ): (task_id, cond, rep)
            for task_id, cond, num_ex, rep in trials
        }
        total = len(futures)
        for i, future in enumerate(as_completed(futures), 1):
            result_type, task_id, cond, rep, msg = future.result()
            if result_type == "skip":
                skipped += 1
            elif result_type == "ok":
                completed += 1
                click.echo(f"[{i}/{total}] task={task_id} {cond} ... {msg}")
            else:
                errors += 1
                click.echo(f"[{i}/{total}] task={task_id} {cond} ... ERROR: {msg}")
            if i % 50 == 0:
                click.echo(f"  --- progress: {completed} done, {skipped} skipped, {errors} errors ---")

    click.echo(f"\nDone. completed={completed} skipped={skipped} errors={errors}")


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True)
def report(db_path, model_name):
    """Show baseline results for a model."""
    conn = init_db(db_path)

    for cond in ["examples_only", "language_only", "both"]:
        total = conn.execute(
            "SELECT COUNT(*) FROM baseline_trials WHERE model_name=? AND condition=?",
            (model_name, cond),
        ).fetchone()[0]
        correct = conn.execute(
            "SELECT COUNT(*) FROM baseline_trials WHERE model_name=? AND condition=? AND correct=1",
            (model_name, cond),
        ).fetchone()[0]
        errors = conn.execute(
            "SELECT COUNT(*) FROM baseline_trials WHERE model_name=? AND condition=? AND error IS NOT NULL",
            (model_name, cond),
        ).fetchone()[0]

        if total:
            pct = correct / total * 100
            click.echo(f"  {cond:20s}: {correct}/{total} correct ({pct:.1f}%) | {errors} errors")
        else:
            click.echo(f"  {cond:20s}: no trials")

    # Overall
    total = conn.execute(
        "SELECT COUNT(DISTINCT task_id) FROM baseline_trials WHERE model_name=?",
        (model_name,),
    ).fetchone()[0]
    click.echo(f"\n  Tasks tested: {total}")

    conn.close()


if __name__ == "__main__":
    cli()
