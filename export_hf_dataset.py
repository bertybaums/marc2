"""Export marc2.db to HuggingFace Datasets format (Parquet).

Creates a multi-config dataset:
  - tasks:              1,000 task metadata records
  - solve_trials:       solving results with reasoning traces
  - descriptions:       791 validated language-complete descriptions
  - task_subsets:        subset classifications per model
  - figurative:         1,910 figurative descriptions (original + alternatives)
  - baseline_trials:    2,373 baseline trial results
  - figurative_trials:  8,645 figurative trial results

Prompt text and raw API responses are excluded to keep the dataset compact.
Reasoning traces from solve_trials ARE included as they are a core artifact.

Usage:
    python export_hf_dataset.py [--db marc2.db] [--output hf_dataset/]
"""

import sqlite3
from pathlib import Path

import click
import pyarrow as pa
import pyarrow.parquet as pq


def _safe_int(val):
    return val if val is not None else 0

def _safe_float(val):
    return val if val is not None else float("nan")


def export_tasks(conn, out_dir):
    """Export tasks table."""
    rows = conn.execute("""
        SELECT task_id, arc_name, source, num_train
        FROM tasks ORDER BY task_id
    """).fetchall()

    table = pa.table({
        "task_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "arc_name": pa.array([r[1] for r in rows]),
        "source": pa.array([r[2] for r in rows]),
        "num_train": pa.array([r[3] for r in rows], type=pa.int32()),
    })

    path = out_dir / "tasks" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  tasks: {len(rows)} rows -> {path}")


def export_solve_trials(conn, out_dir):
    """Export solve_trials table (reasoning traces are the gold)."""
    rows = conn.execute("""
        SELECT trial_id, task_id, attempt, reasoning,
               predicted_grid, correct, cell_accuracy, error
        FROM solve_trials ORDER BY task_id, attempt
    """).fetchall()

    table = pa.table({
        "trial_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "attempt": pa.array([r[2] for r in rows], type=pa.int32()),
        "reasoning": pa.array([r[3] for r in rows]),
        "predicted_grid": pa.array([r[4] for r in rows]),
        "correct": pa.array([_safe_int(r[5]) for r in rows], type=pa.int32()),
        "cell_accuracy": pa.array([_safe_float(r[6]) for r in rows], type=pa.float32()),
        "error": pa.array([r[7] for r in rows]),
    })

    path = out_dir / "solve_trials" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  solve_trials: {len(rows)} rows -> {path}")


def export_descriptions(conn, out_dir):
    """Export validated language-complete descriptions."""
    rows = conn.execute("""
        SELECT desc_id, task_id, see_description, do_description,
               grid_description, validated, revision_count
        FROM descriptions ORDER BY task_id
    """).fetchall()

    table = pa.table({
        "desc_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "see_description": pa.array([r[2] for r in rows]),
        "do_description": pa.array([r[3] for r in rows]),
        "grid_description": pa.array([r[4] for r in rows]),
        "validated": pa.array([_safe_int(r[5]) for r in rows], type=pa.int32()),
        "revision_count": pa.array([_safe_int(r[6]) for r in rows], type=pa.int32()),
    })

    path = out_dir / "descriptions" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  descriptions: {len(rows)} rows -> {path}")


def export_task_subsets(conn, out_dir):
    """Export task_subsets table."""
    rows = conn.execute("""
        SELECT task_id, model_name, subset, min_examples_with_language
        FROM task_subsets ORDER BY task_id, model_name
    """).fetchall()

    table = pa.table({
        "task_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "model_name": pa.array([r[1] for r in rows]),
        "subset": pa.array([r[2] for r in rows]),
        "min_examples_with_language": pa.array(
            [r[3] if r[3] is not None else -1 for r in rows], type=pa.int32()
        ),
    })

    path = out_dir / "task_subsets" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  task_subsets: {len(rows)} rows -> {path}")


def export_figurative_descriptions(conn, out_dir):
    """Export figurative_descriptions table."""
    rows = conn.execute("""
        SELECT fig_id, task_id, generator_model, variant,
               source_domain, metaphor,
               figurative_see, figurative_do, figurative_grid
        FROM figurative_descriptions ORDER BY task_id, variant
    """).fetchall()

    table = pa.table({
        "fig_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "generator_model": pa.array([r[2] for r in rows]),
        "variant": pa.array([r[3] for r in rows]),
        "source_domain": pa.array([r[4] for r in rows]),
        "metaphor": pa.array([r[5] for r in rows]),
        "figurative_see": pa.array([r[6] for r in rows]),
        "figurative_do": pa.array([r[7] for r in rows]),
        "figurative_grid": pa.array([r[8] for r in rows]),
    })

    path = out_dir / "figurative_descriptions" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  figurative_descriptions: {len(rows)} rows -> {path}")


def export_baseline_trials(conn, out_dir):
    """Export baseline_trials table (behavioral results only)."""
    rows = conn.execute("""
        SELECT trial_id, task_id, model_name, condition,
               num_examples, correct, cell_accuracy
        FROM baseline_trials
        ORDER BY task_id, model_name, condition, num_examples
    """).fetchall()

    table = pa.table({
        "trial_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "model_name": pa.array([r[2] for r in rows]),
        "condition": pa.array([r[3] for r in rows]),
        "num_examples": pa.array([r[4] for r in rows], type=pa.int32()),
        "correct": pa.array([_safe_int(r[5]) for r in rows], type=pa.int32()),
        "cell_accuracy": pa.array([_safe_float(r[6]) for r in rows], type=pa.float32()),
    })

    path = out_dir / "baseline_trials" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  baseline_trials: {len(rows)} rows -> {path}")


def export_figurative_trials(conn, out_dir):
    """Export figurative_trials table (behavioral results only)."""
    rows = conn.execute("""
        SELECT ft.trial_id, ft.fig_id, ft.task_id, ft.model_name,
               ft.num_examples, ft.correct, ft.cell_accuracy,
               fd.variant, fd.source_domain
        FROM figurative_trials ft
        JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
        ORDER BY ft.task_id, fd.variant, ft.num_examples
    """).fetchall()

    table = pa.table({
        "trial_id": pa.array([r[0] for r in rows], type=pa.int32()),
        "fig_id": pa.array([r[1] for r in rows], type=pa.int32()),
        "task_id": pa.array([r[2] for r in rows], type=pa.int32()),
        "model_name": pa.array([r[3] for r in rows]),
        "num_examples": pa.array([r[4] for r in rows], type=pa.int32()),
        "correct": pa.array([_safe_int(r[5]) for r in rows], type=pa.int32()),
        "cell_accuracy": pa.array([_safe_float(r[6]) for r in rows], type=pa.float32()),
        "variant": pa.array([r[7] for r in rows]),
        "source_domain": pa.array([r[8] for r in rows]),
    })

    path = out_dir / "figurative_trials" / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    print(f"  figurative_trials: {len(rows)} rows -> {path}")


@click.command()
@click.option("--db", "db_path", default="marc2.db")
@click.option("--output", "output_dir", default="hf_dataset")
def export(db_path, output_dir):
    """Export marc2.db to HuggingFace Datasets format (Parquet)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    print(f"Exporting {db_path} -> {out_dir}/\n")

    export_tasks(conn, out_dir)
    export_solve_trials(conn, out_dir)
    export_descriptions(conn, out_dir)
    export_task_subsets(conn, out_dir)
    export_figurative_descriptions(conn, out_dir)
    export_baseline_trials(conn, out_dir)
    export_figurative_trials(conn, out_dir)

    conn.close()
    print(f"\nDone. Dataset ready at {out_dir}/")


if __name__ == "__main__":
    export()
