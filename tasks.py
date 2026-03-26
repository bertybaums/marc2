"""ARC-AGI2 task loading and database initialization."""

import json
from pathlib import Path

import click
import yaml

from db import init_db, upsert_task, get_all_tasks


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_arc_dir(config):
    """Return the ARC-AGI2 data directory as a Path."""
    base = Path(__file__).parent
    return base / config["data"]["arc_agi2_dir"] / "data"


def load_arc_task(arc_dir, arc_name, source='training'):
    """Load an ARC-AGI2 task JSON. Returns {"train": [...], "test": [...]}."""
    path = arc_dir / source / f"{arc_name}.json"
    with open(path) as f:
        return json.load(f)


def resolve_arc_path(config, arc_name):
    """Find the ARC-AGI2 task JSON file, checking both training and evaluation."""
    arc_dir = get_arc_dir(config)
    for subdir in ["training", "evaluation"]:
        path = arc_dir / subdir / f"{arc_name}.json"
        if path.exists():
            return path
    return None


def iter_tasks(arc_dir, source='training'):
    """Iterate over (arc_name, task_data) pairs for a given split.

    Yields (arc_name, task_dict) sorted by filename.
    """
    subdir = arc_dir / source
    for path in sorted(subdir.glob("*.json")):
        arc_name = path.stem
        with open(path) as f:
            yield arc_name, json.load(f)


@click.group()
def cli():
    """MARC2 task management."""
    pass


@cli.command()
@click.option("--config", "config_path", default="config.yaml")
@click.option("--db", "db_path", default="marc2.db")
def init(config_path, db_path):
    """Initialize the database with ARC-AGI2 tasks."""
    config = load_config(config_path)
    arc_dir = get_arc_dir(config)

    if not arc_dir.exists():
        click.echo(f"ERROR: ARC-AGI2 data not found at {arc_dir}")
        click.echo("Run: bash setup.sh")
        return

    conn = init_db(db_path)

    # Training tasks: task_id 0-999
    training_dir = arc_dir / "training"
    training_files = sorted(training_dir.glob("*.json"))
    for idx, path in enumerate(training_files):
        arc_name = path.stem
        with open(path) as f:
            task = json.load(f)
        num_train = len(task["train"])
        upsert_task(conn, idx, arc_name, num_train, source='training')

    # Evaluation tasks: task_id 2000-2119
    eval_dir = arc_dir / "evaluation"
    eval_files = sorted(eval_dir.glob("*.json"))
    for idx, path in enumerate(eval_files):
        arc_name = path.stem
        with open(path) as f:
            task = json.load(f)
        num_train = len(task["train"])
        upsert_task(conn, 2000 + idx, arc_name, num_train, source='evaluation')

    click.echo(f"Loaded {len(training_files)} training + {len(eval_files)} evaluation tasks")
    conn.close()


@cli.command("list")
@click.option("--db", "db_path", default="marc2.db")
@click.option("--source", type=click.Choice(["training", "evaluation"]), default=None)
def list_tasks(db_path, source):
    """List all tasks in the database."""
    conn = init_db(db_path)
    tasks = get_all_tasks(conn, source=source)
    for t in tasks:
        click.echo(f"  {t['task_id']:5d}  {t['arc_name']}  train={t['num_train']}  {t['source']}")
    click.echo(f"Total: {len(tasks)} tasks")
    conn.close()


@cli.command()
@click.option("--db", "db_path", default="marc2.db")
@click.argument("task_id", type=int)
def show(db_path, task_id):
    """Show details of a specific task."""
    conn = init_db(db_path)
    from db import get_task, get_description, get_solve_trial
    t = get_task(conn, task_id)
    if not t:
        click.echo(f"Task {task_id} not found")
        return
    click.echo(f"Task {t['task_id']}: {t['arc_name']} ({t['source']})")
    click.echo(f"Training examples: {t['num_train']}")

    # Show solve status
    trials = get_solve_trial(conn, task_id)
    if trials:
        for trial in trials:
            status = "CORRECT" if trial['correct'] else "WRONG"
            acc = f"{trial['cell_accuracy']:.1%}" if trial['cell_accuracy'] is not None else "?"
            click.echo(f"  Solve attempt {trial['attempt']}: {status} ({acc})")

    # Show description status
    desc = get_description(conn, task_id)
    if desc:
        val = {0: "untested", 1: "VALIDATED", -1: "FAILED"}[desc['validated']]
        click.echo(f"  Description: {val} (rev {desc['revision_count']})")
        click.echo(f"    See: {desc['see_description'][:80]}...")
        click.echo(f"    Do:  {desc['do_description'][:80]}...")

    conn.close()


if __name__ == "__main__":
    cli()
