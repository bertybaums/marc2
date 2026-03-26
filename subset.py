"""Phase 5: Task classification based on baseline results.

Classifies each task (per model) into:
  - examples_sufficient: solved by examples alone
  - language_sufficient: not solved by examples, but solved by description alone
  - both_required: requires both description and examples
  - unsolvable: none of the conditions work

Usage:
    python subset.py classify --model gpt-oss-120b
    python subset.py report --model gpt-oss-120b
"""

import click

from db import init_db, get_task, upsert_subset, get_subsets, get_validated_descriptions


def _get_baseline_result(conn, task_id, model_name, condition):
    """Get whether the baseline condition was correct."""
    row = conn.execute(
        """SELECT MAX(correct) as best_correct FROM baseline_trials
           WHERE task_id=? AND model_name=? AND condition=?""",
        (task_id, model_name, condition),
    ).fetchone()
    if row is None or row["best_correct"] is None:
        return None
    return row["best_correct"] == 1


@click.group()
def cli():
    """Phase 5: Task classification."""
    pass


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", required=True)
def classify(db_path, model_name):
    """Classify tasks into subsets based on baseline results."""
    conn = init_db(db_path)

    # Only classify tasks with validated descriptions (those tested in Phase 4)
    validated = get_validated_descriptions(conn)

    counts = {"examples_sufficient": 0, "language_sufficient": 0,
              "both_required": 0, "unsolvable": 0, "incomplete": 0}

    for desc in validated:
        tid = desc["task_id"]
        ex_only = _get_baseline_result(conn, tid, model_name, "examples_only")
        lang_only = _get_baseline_result(conn, tid, model_name, "language_only")
        both = _get_baseline_result(conn, tid, model_name, "both")

        if ex_only is None or lang_only is None or both is None:
            counts["incomplete"] += 1
            continue

        if ex_only:
            subset = "examples_sufficient"
        elif lang_only:
            subset = "language_sufficient"
        elif both:
            subset = "both_required"
        else:
            subset = "unsolvable"

        upsert_subset(conn, tid, model_name, subset)
        counts[subset] += 1

    click.echo(f"Subset classification for {model_name}:")
    total = sum(counts.values())
    for name, count in counts.items():
        pct = count / total * 100 if total else 0
        click.echo(f"  {name:25s}: {count:4d} ({pct:.1f}%)")

    marc_eligible = counts["language_sufficient"] + counts["both_required"]
    click.echo(f"\nMARC-eligible (lang_suff + both_req): {marc_eligible}")

    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--model", "model_name", default=None)
def report(db_path, model_name):
    """Print subset summary."""
    conn = init_db(db_path)
    subsets = get_subsets(conn, model_name=model_name)

    by_model = {}
    for s in subsets:
        mn = s["model_name"]
        if mn not in by_model:
            by_model[mn] = {}
        sub = s["subset"]
        if sub not in by_model[mn]:
            by_model[mn][sub] = []
        by_model[mn][sub].append(s)

    for mn, subs in sorted(by_model.items()):
        click.echo(f"\n{mn}:")
        total = sum(len(rows) for rows in subs.values())
        for sub in ["examples_sufficient", "language_sufficient", "both_required", "unsolvable"]:
            rows = subs.get(sub, [])
            pct = len(rows) / total * 100 if total else 0
            click.echo(f"  {sub:25s}: {len(rows):4d} ({pct:.1f}%)")

        marc = len(subs.get("language_sufficient", [])) + len(subs.get("both_required", []))
        click.echo(f"  MARC-eligible:            {marc}")

    conn.close()


if __name__ == "__main__":
    cli()
