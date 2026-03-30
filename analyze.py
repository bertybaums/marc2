#!/usr/bin/env python3
"""Phase 9: Analysis and reporting — HTML report with Plotly charts."""

import json
from pathlib import Path

import click
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from db import init_db


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _pipeline_funnel(conn, model_name):
    """Compute pipeline stage counts for the waterfall chart."""
    total_tasks = conn.execute(
        "SELECT COUNT(*) as n FROM tasks WHERE source='training'"
    ).fetchone()["n"]

    solved = conn.execute(
        "SELECT COUNT(DISTINCT task_id) as n FROM solve_trials WHERE correct=1"
    ).fetchone()["n"]

    validated = conn.execute(
        "SELECT COUNT(*) as n FROM descriptions WHERE validated=1"
    ).fetchone()["n"]

    # MARC-eligible = language_sufficient + both_required
    marc_eligible = conn.execute(
        "SELECT COUNT(*) as n FROM task_subsets WHERE model_name=? AND subset IN ('language_sufficient', 'both_required')",
        (model_name,),
    ).fetchone()["n"]

    # MARC-verified = tasks with at least one figurative trial that is correct
    # with examples > 0, where examples_only failed and fig_only failed
    marc_verified = conn.execute(
        """SELECT COUNT(DISTINCT ft.task_id) as n
           FROM figurative_trials ft
           JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
           WHERE ft.model_name=? AND ft.correct=1 AND ft.num_examples > 0
             AND ft.task_id NOT IN (
               SELECT task_id FROM baseline_trials
               WHERE model_name=? AND condition='examples_only' AND correct=1
             )
             AND ft.fig_id NOT IN (
               SELECT fig_id FROM figurative_trials
               WHERE model_name=? AND num_examples=0 AND correct=1
             )""",
        (model_name, model_name, model_name),
    ).fetchone()["n"]

    # Total MARC clues = distinct fig_ids that satisfy MARC property
    total_clues = conn.execute(
        """SELECT COUNT(DISTINCT ft.fig_id) as n
           FROM figurative_trials ft
           JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
           WHERE ft.model_name=? AND ft.correct=1 AND ft.num_examples > 0
             AND ft.task_id NOT IN (
               SELECT task_id FROM baseline_trials
               WHERE model_name=? AND condition='examples_only' AND correct=1
             )
             AND ft.fig_id NOT IN (
               SELECT fig_id FROM figurative_trials
               WHERE model_name=? AND num_examples=0 AND correct=1
             )""",
        (model_name, model_name, model_name),
    ).fetchone()["n"]

    return {
        "total_tasks": total_tasks,
        "solved": solved,
        "validated": validated,
        "marc_eligible": marc_eligible,
        "marc_verified": marc_verified,
        "total_clues": total_clues,
    }


def _baseline_stats(conn, model_name):
    """Success rates per condition."""
    rows = conn.execute(
        """SELECT condition,
                  COUNT(*) as total,
                  SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as correct_count,
                  AVG(cell_accuracy) as avg_cell_acc
           FROM baseline_trials
           WHERE model_name=? AND correct IS NOT NULL
           GROUP BY condition""",
        (model_name,),
    ).fetchall()
    return {r["condition"]: dict(r) for r in rows}


def _subset_counts(conn, model_name):
    """Subset sizes."""
    rows = conn.execute(
        """SELECT subset, COUNT(*) as count
           FROM task_subsets WHERE model_name=?
           GROUP BY subset""",
        (model_name,),
    ).fetchall()
    return {r["subset"]: r["count"] for r in rows}


def _marc_tasks(conn, model_name):
    """Tasks satisfying the MARC property with min_k and metaphor."""
    return conn.execute(
        """SELECT ft.task_id, MIN(ft.num_examples) as min_k, fd.metaphor,
                  fd.source_domain, fd.variant
           FROM figurative_trials ft
           JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
           WHERE ft.model_name=? AND ft.correct=1 AND ft.num_examples > 0
             AND ft.task_id NOT IN (
               SELECT task_id FROM baseline_trials
               WHERE model_name=? AND condition='examples_only' AND correct=1
             )
             AND ft.fig_id NOT IN (
               SELECT fig_id FROM figurative_trials
               WHERE model_name=? AND num_examples=0 AND correct=1
             )
           GROUP BY ft.task_id, fd.fig_id
           ORDER BY ft.task_id, fd.variant""",
        (model_name, model_name, model_name),
    ).fetchall()


def _marc_yield_by_domain(conn, model_name):
    """Count of MARC-valid clues per source domain."""
    rows = conn.execute(
        """SELECT fd.source_domain, COUNT(DISTINCT fd.fig_id) as count
           FROM figurative_trials ft
           JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
           WHERE ft.model_name=? AND ft.correct=1 AND ft.num_examples > 0
             AND fd.source_domain IS NOT NULL
             AND ft.task_id NOT IN (
               SELECT task_id FROM baseline_trials
               WHERE model_name=? AND condition='examples_only' AND correct=1
             )
             AND ft.fig_id NOT IN (
               SELECT fig_id FROM figurative_trials
               WHERE model_name=? AND num_examples=0 AND correct=1
             )
           GROUP BY fd.source_domain
           ORDER BY count DESC""",
        (model_name, model_name, model_name),
    ).fetchall()
    return rows


def _min_k_distribution(conn, model_name):
    """Distribution of min_k across MARC-verified tasks."""
    rows = conn.execute(
        """SELECT MIN(ft.num_examples) as min_k
           FROM figurative_trials ft
           JOIN figurative_descriptions fd ON ft.fig_id = fd.fig_id
           WHERE ft.model_name=? AND ft.correct=1 AND ft.num_examples > 0
             AND ft.task_id NOT IN (
               SELECT task_id FROM baseline_trials
               WHERE model_name=? AND condition='examples_only' AND correct=1
             )
             AND ft.fig_id NOT IN (
               SELECT fig_id FROM figurative_trials
               WHERE model_name=? AND num_examples=0 AND correct=1
             )
           GROUP BY ft.task_id""",
        (model_name, model_name, model_name),
    ).fetchall()
    return [r["min_k"] for r in rows]


def _cell_accuracy_by_condition(conn, model_name):
    """Cell accuracy values grouped by condition."""
    result = {}
    for cond in ["examples_only", "language_only", "both"]:
        rows = conn.execute(
            """SELECT cell_accuracy FROM baseline_trials
               WHERE model_name=? AND condition=? AND cell_accuracy IS NOT NULL""",
            (model_name, cond),
        ).fetchall()
        result[cond] = [r["cell_accuracy"] for r in rows]
    return result


def _opacity_by_domain(conn, model_name):
    """Per-domain opacity analysis: too_transparent / marc_valid / too_opaque.

    too_transparent: figurative_only succeeds (not opaque enough)
    marc_valid: examples_only fails, fig_only fails, fig+examples succeeds
    too_opaque: fig+examples also fails (too opaque to help)
    """
    domains = conn.execute(
        """SELECT DISTINCT source_domain FROM figurative_descriptions
           WHERE source_domain IS NOT NULL ORDER BY source_domain"""
    ).fetchall()

    result = {}
    for row in domains:
        domain = row["source_domain"]

        # All fig_ids for this domain that have been tested
        fig_ids = conn.execute(
            """SELECT DISTINCT fd.fig_id, fd.task_id
               FROM figurative_descriptions fd
               JOIN figurative_trials ft ON fd.fig_id = ft.fig_id
               WHERE fd.source_domain=? AND ft.model_name=?""",
            (domain, model_name),
        ).fetchall()

        transparent = 0
        valid = 0
        opaque = 0

        for frow in fig_ids:
            fid = frow["fig_id"]
            tid = frow["task_id"]

            # Check if examples_only succeeds for this task
            ex_ok = conn.execute(
                """SELECT COUNT(*) as n FROM baseline_trials
                   WHERE task_id=? AND model_name=? AND condition='examples_only'
                         AND correct=1""",
                (tid, model_name),
            ).fetchone()["n"] > 0

            if ex_ok:
                # Task already solvable from examples — skip
                continue

            # Check if fig_only succeeds
            fig_only_ok = conn.execute(
                """SELECT COUNT(*) as n FROM figurative_trials
                   WHERE fig_id=? AND model_name=? AND num_examples=0 AND correct=1""",
                (fid, model_name),
            ).fetchone()["n"] > 0

            if fig_only_ok:
                transparent += 1
                continue

            # Check if fig+examples succeeds
            fig_ex_ok = conn.execute(
                """SELECT COUNT(*) as n FROM figurative_trials
                   WHERE fig_id=? AND model_name=? AND num_examples > 0 AND correct=1""",
                (fid, model_name),
            ).fetchone()["n"] > 0

            if fig_ex_ok:
                valid += 1
            else:
                opaque += 1

        result[domain] = {"too_transparent": transparent, "marc_valid": valid, "too_opaque": opaque}

    return result


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _chart_funnel(funnel):
    """Pipeline waterfall chart."""
    stages = ["Training Tasks", "Solved", "Validated", "MARC-Eligible",
              "MARC-Verified", "Total Clues"]
    values = [funnel["total_tasks"], funnel["solved"], funnel["validated"],
              funnel["marc_eligible"], funnel["marc_verified"], funnel["total_clues"]]

    # Waterfall: show successive decreases, then the clue count as a total
    measures = ["absolute", "absolute", "absolute", "absolute", "absolute", "absolute"]

    fig = go.Figure(go.Waterfall(
        x=stages,
        y=values,
        measure=measures,
        text=[str(v) for v in values],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(
        title="Pipeline Funnel: ARC-AGI2 to MARC Clues",
        yaxis_title="Count",
        showlegend=False,
    )
    return fig


def _chart_baseline(stats):
    """Grouped bar chart of baseline success rates."""
    conditions = ["examples_only", "language_only", "both"]
    labels = ["Examples Only", "Language Only", "Both"]
    rates = []
    for cond in conditions:
        if cond in stats:
            s = stats[cond]
            rates.append(s["correct_count"] / s["total"] * 100 if s["total"] > 0 else 0)
        else:
            rates.append(0)

    fig = go.Figure(go.Bar(
        x=labels,
        y=rates,
        text=[f"{r:.1f}%" for r in rates],
        textposition="auto",
        marker_color=["#636EFA", "#EF553B", "#00CC96"],
    ))
    fig.update_layout(
        title="Baseline Success Rates by Condition",
        yaxis_title="Success Rate (%)",
        yaxis_range=[0, 100],
    )
    return fig


def _chart_subsets(counts):
    """Bar chart of subset distribution."""
    subset_names = ["examples_sufficient", "language_sufficient", "both_required", "unsolvable"]
    labels = ["Examples\nSufficient", "Language\nSufficient", "Both\nRequired", "Unsolvable"]
    values = [counts.get(s, 0) for s in subset_names]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        text=[str(v) for v in values],
        textposition="auto",
        marker_color=colors,
    ))
    fig.update_layout(
        title="Task Subset Distribution",
        yaxis_title="Number of Tasks",
    )
    return fig


def _chart_yield_by_domain(domain_rows):
    """Bar chart of MARC yield per source domain."""
    domains = [r["source_domain"] for r in domain_rows]
    counts = [r["count"] for r in domain_rows]

    fig = go.Figure(go.Bar(
        x=domains,
        y=counts,
        text=[str(c) for c in counts],
        textposition="auto",
    ))
    fig.update_layout(
        title="MARC Yield by Source Domain",
        xaxis_title="Source Domain",
        yaxis_title="MARC-Valid Clues",
        xaxis_tickangle=-45,
    )
    return fig


def _chart_min_k(min_ks):
    """Histogram of min_k distribution."""
    fig = go.Figure(go.Histogram(
        x=min_ks,
        nbinsx=max(min_ks) if min_ks else 5,
        marker_color="#636EFA",
    ))
    fig.update_layout(
        title="Distribution of Minimum Examples Needed (min_k)",
        xaxis_title="min_k (examples needed with figurative description)",
        yaxis_title="Number of Tasks",
    )
    return fig


def _chart_cell_accuracy(acc_data):
    """Box plots of cell accuracy per condition."""
    fig = go.Figure()
    colors = {"examples_only": "#636EFA", "language_only": "#EF553B", "both": "#00CC96"}
    labels = {"examples_only": "Examples Only", "language_only": "Language Only", "both": "Both"}
    for cond in ["examples_only", "language_only", "both"]:
        if acc_data.get(cond):
            fig.add_trace(go.Box(
                y=acc_data[cond],
                name=labels[cond],
                marker_color=colors[cond],
            ))
    fig.update_layout(
        title="Cell Accuracy Distributions by Condition",
        yaxis_title="Cell Accuracy",
        yaxis_range=[0, 1.05],
    )
    return fig


def _chart_opacity(opacity_data):
    """Stacked bar: per domain showing too_transparent / marc_valid / too_opaque."""
    domains = sorted(opacity_data.keys())
    transparent = [opacity_data[d]["too_transparent"] for d in domains]
    valid = [opacity_data[d]["marc_valid"] for d in domains]
    opaque = [opacity_data[d]["too_opaque"] for d in domains]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Too Transparent", x=domains, y=transparent,
                         marker_color="#FFA15A"))
    fig.add_trace(go.Bar(name="MARC Valid", x=domains, y=valid,
                         marker_color="#00CC96"))
    fig.add_trace(go.Bar(name="Too Opaque", x=domains, y=opaque,
                         marker_color="#EF553B"))
    fig.update_layout(
        title="Figurative Opacity Analysis by Domain",
        xaxis_title="Source Domain",
        yaxis_title="Number of Clues",
        barmode="stack",
        xaxis_tickangle=-45,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

def _render_chart(fig, div_id):
    """Return HTML snippet to render a Plotly chart."""
    fig_json = fig.to_json()
    return (
        f'<div class="chart" id="{div_id}"></div>\n'
        f"<script>(function() {{ var d = {fig_json}; "
        f"Plotly.newPlot('{div_id}', d.data, d.layout, {{responsive: true}}); }})();</script>\n"
    )


def _build_html(funnel, stats, counts, domain_yield, min_ks, acc_data, opacity_data,
                marc_rows, model_name):
    """Assemble the full HTML report."""
    parts = []
    parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MARC2 Analysis Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 1200px;
         margin: auto; padding: 20px; background: #fafafa; color: #222; }
  h1 { border-bottom: 3px solid #636EFA; padding-bottom: 8px; }
  h2 { color: #444; margin-top: 40px; }
  .chart { margin: 30px 0; background: white; padding: 10px;
           border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.1);
           min-height: 450px; position: relative; overflow: hidden; }
  table { border-collapse: collapse; margin: 15px 0; width: 100%; }
  th, td { border: 1px solid #ccc; padding: 8px 12px; text-align: left; }
  th { background: #636EFA; color: white; }
  tr:nth-child(even) { background: #f5f5f5; }
  .date { color: #888; font-size: 0.9em; }
  .metric { font-size: 1.4em; font-weight: bold; color: #636EFA; }
  .summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin: 20px 0; }
  .summary-card { background: white; padding: 20px; border-radius: 8px;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.1); text-align: center; }
  .summary-card .label { color: #666; font-size: 0.9em; margin-top: 5px; }
</style>
</head>
<body>
""")

    parts.append(f'<h1>MARC2 Analysis Report</h1>\n')
    parts.append(f'<p class="date">March 26, 2026 &mdash; Subject model: {model_name}</p>\n')

    # --- Summary cards ---
    baseline_total = sum(s["total"] for s in stats.values()) if stats else 0
    baseline_tasks = len(set(
        cond_stats.get("total", 0) for cond_stats in stats.values()
    ))
    ex_rate = (stats["examples_only"]["correct_count"] / stats["examples_only"]["total"] * 100
               if stats.get("examples_only") and stats["examples_only"]["total"] > 0 else 0)
    lang_rate = (stats["language_only"]["correct_count"] / stats["language_only"]["total"] * 100
                 if stats.get("language_only") and stats["language_only"]["total"] > 0 else 0)

    parts.append('<div class="summary-grid">\n')
    cards = [
        (str(funnel["total_tasks"]), "Training Tasks"),
        (str(funnel["solved"]), "Solved by Claude"),
        (str(funnel["validated"]), "Validated Descriptions"),
        (str(funnel["marc_eligible"]), "MARC-Eligible"),
        (str(funnel["marc_verified"]), "MARC-Verified Tasks"),
        (str(funnel["total_clues"]), "Total MARC Clues"),
    ]
    for val, label in cards:
        parts.append(f'<div class="summary-card"><div class="metric">{val}</div>'
                     f'<div class="label">{label}</div></div>\n')
    parts.append('</div>\n')

    # --- Summary table ---
    parts.append("<h2>Key Metrics</h2>\n<table>\n")
    parts.append("<tr><th>Metric</th><th>Value</th></tr>\n")
    metrics = [
        ("Training tasks", funnel["total_tasks"]),
        ("Solved by Claude", f"{funnel['solved']} ({funnel['solved']/funnel['total_tasks']*100:.1f}%)"),
        ("Validated descriptions", f"{funnel['validated']} ({funnel['validated']/max(funnel['solved'],1)*100:.1f}% of solved)"),
        ("MARC-eligible (lang_suff + both_req)", funnel["marc_eligible"]),
        ("MARC-verified tasks", funnel["marc_verified"]),
        ("Total MARC clues (all variants)", funnel["total_clues"]),
        ("Examples-only success rate", f"{ex_rate:.1f}%"),
        ("Language-only success rate", f"{lang_rate:.1f}%"),
    ]
    for label, val in metrics:
        parts.append(f"<tr><td>{label}</td><td>{val}</td></tr>\n")
    parts.append("</table>\n")

    # --- Charts ---
    charts = [
        ("funnel", _chart_funnel(funnel)),
        ("baseline", _chart_baseline(stats)),
        ("subsets", _chart_subsets(counts)),
        ("domain_yield", _chart_yield_by_domain(domain_yield)),
        ("min_k", _chart_min_k(min_ks)),
        ("cell_acc", _chart_cell_accuracy(acc_data)),
        ("opacity", _chart_opacity(opacity_data)),
    ]
    for div_id, fig in charts:
        parts.append(_render_chart(fig, div_id))

    # --- MARC puzzles table ---
    # De-duplicate to one row per task (best variant)
    seen_tasks = {}
    for row in marc_rows:
        tid = row["task_id"]
        if tid not in seen_tasks:
            seen_tasks[tid] = row

    parts.append(f"<h2>Valid MARC Puzzles ({len(seen_tasks)} tasks)</h2>\n")
    parts.append("<table>\n")
    parts.append("<tr><th>Task ID</th><th>min_k</th><th>Domain</th><th>Metaphor</th></tr>\n")
    for tid in sorted(seen_tasks.keys()):
        row = seen_tasks[tid]
        domain = row["source_domain"] or ""
        metaphor = row["metaphor"] or ""
        # Truncate long metaphors for the table
        if len(metaphor) > 120:
            metaphor = metaphor[:117] + "..."
        parts.append(
            f"<tr><td>{row['task_id']}</td><td>{row['min_k']}</td>"
            f"<td>{domain}</td><td>{metaphor}</td></tr>\n"
        )
    parts.append("</table>\n")

    parts.append("</body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--db", "db_path", default="marc2.db", help="Path to SQLite database.")
@click.option("--model", "model_name", default="gpt-oss-120b", help="Subject model name.")
@click.option("--output", "output_path", default="report.html", help="Output HTML file.")
def analyze(db_path, model_name, output_path):
    """Generate MARC2 HTML analysis report with Plotly charts."""
    conn = init_db(db_path)

    click.echo(f"Querying database: {db_path}")
    click.echo(f"Subject model: {model_name}")

    funnel = _pipeline_funnel(conn, model_name)
    stats = _baseline_stats(conn, model_name)
    counts = _subset_counts(conn, model_name)
    domain_yield = _marc_yield_by_domain(conn, model_name)
    min_ks = _min_k_distribution(conn, model_name)
    acc_data = _cell_accuracy_by_condition(conn, model_name)
    opacity_data = _opacity_by_domain(conn, model_name)
    marc_rows = _marc_tasks(conn, model_name)

    click.echo(f"Pipeline: {funnel['total_tasks']} tasks -> {funnel['solved']} solved "
               f"-> {funnel['validated']} validated -> {funnel['marc_eligible']} eligible "
               f"-> {funnel['marc_verified']} verified -> {funnel['total_clues']} clues")

    html = _build_html(funnel, stats, counts, domain_yield, min_ks, acc_data,
                       opacity_data, marc_rows, model_name)

    Path(output_path).write_text(html)
    click.echo(f"Report written to {output_path}")
    conn.close()


if __name__ == "__main__":
    analyze()
