# MARC2: Metaphor Abstraction and Reasoning Corpus v2

## Project Overview

MARC2 extends the MARC-from-LARC methodology to the ARC-AGI2 dataset. The MARC property is **model-relative** — what one model solves from examples alone, another may need figurative language + examples to crack.

MARC2 leverages the **capability gap between models** to systematically produce MARC puzzles:

1. **Upper bound:** Claude (Opus 4.6, 82% on ARC-AGI2 training) solves tasks and distills its reasoning into language-complete descriptions. These are gold-standard because they come from a model that demonstrably understands each task.
2. **Lower bound:** Smaller subject models (gpt-oss-120b, 20b, etc.) are tested on the same tasks. The language-complete descriptions become source material for generating figurative alternatives, which are tested on the subject models for the MARC property.

The bigger the gap between Claude and the subject model, the more tasks enter the MARC-eligible window.

### Pipeline Summary

```
═══ UPPER BOUND: Claude (Opus 4.6) establishes what's solvable ═══

ARC-AGI2 tasks (1,120 public)
    ↓
Phase 1: Claude solves tasks via subagents, capturing reasoning traces (82% on training → 821/1000)
    ↓
Phase 2: Distill reasoning into LARC-style language-complete descriptions (see/do/grid)
    ↓
Phase 3: Validate descriptions (fresh subagent solves from description + test input alone → 791 validated)

═══ LOWER BOUND: Subject models reveal where figurative language helps ═══

Phase 4: Baseline 3-condition testing on subject models (examples_only, language_only, both)
    ↓
Phase 5: Task classification (examples_sufficient / language_sufficient / both_required / unsolvable)
    ↓
Phase 6: Generate figurative descriptions from validated literal ones (Claude subagents)
    ↓
Phase 7: Test figurative descriptions on subject models, verify MARC property
    ↓
Phase 8: Generate domain-diverse alternative figurative clues (Claude subagents)
    ↓
Phase 9: Analysis, inspection, export
```

## Project Structure

```
marc2/
├── .claude/CLAUDE.md       # This file
├── PLAN.md                 # Detailed implementation plan
├── DESIGN.md               # Architecture and design decisions
│
├── config.yaml             # Model configs (MindRouter endpoints for subject models)
├── schema.sql              # SQLite schema (extended from marc-from-larc)
├── marc2.db                # SQLite database (WAL mode)
│
├── data/
│   └── arc-agi2/           # git clone of https://github.com/arcprize/ARC-AGI-2
│
├── db.py                   # Database helpers
├── models.py               # LLM API abstraction (OpenAI-compat for MindRouter only)
├── grids.py                # Grid rendering (text ↔ 2D array ↔ PNG)
├── prompts.py              # Prompt templates for all conditions
├── tasks.py                # ARC-AGI2 task loading
├── utils.py                # Shared utilities
│
├── solve.py                # Phase 1: Orchestrates Claude Code subagents to solve ARC-AGI2
├── distill.py              # Phase 2: Subagents distill reasoning → language-complete descriptions
├── validate.py             # Phase 3: Subagents validate descriptions (fresh context, no examples)
├── collect.py              # Phase 4: Baseline 3-condition testing on subject models
├── subset.py               # Phase 5: Task classification
├── generate_figurative.py  # Phase 6: Generate figurative descriptions
├── test_figurative.py      # Phase 7: Test figurative descriptions, verify MARC
├── generate_alternatives.py # Phase 8: Domain-diverse alternative clues
├── analyze.py              # Phase 9: Reports, charts, export
├── inspector.py            # HTML inspector views
│
├── inspect.html            # Generated: main inspector
├── compare.html            # Generated: variant comparison
└── hf_dataset/             # Generated: HuggingFace Parquet export
```

## Data Source

### ARC-AGI2 (https://github.com/arcprize/ARC-AGI-2)
- **1,000 training tasks** + **120 public evaluation tasks** = 1,120 public tasks
- Same JSON format as ARC-AGI1: `{"train": [...], "test": [...]}`
- Each pair: `{"input": [[int]], "output": [[int]]}`, values 0-9, grids up to 30x30
- Task files: 8-character hex IDs (e.g., `00576224.json`)
- Training tasks include ARC-AGI1 tasks (superset); evaluation tasks are new and harder

### Task ID Scheme
- Training tasks: task_id 0-999 (mapped from sorted filenames)
- Evaluation tasks: task_id 2000-2119 (offset to distinguish from training)
- The `source` column distinguishes: 'training' or 'evaluation'

## Database Schema

Extended from marc-from-larc with two new tables for the solve/distill pipeline:

### New Tables
- **solve_trials**: Claude's solving attempts (prompt, reasoning, predicted_grid, correct)
- **descriptions**: Distilled language-complete descriptions (see/do/grid) with validation status

### Inherited Tables (same schema as marc-from-larc)
- **tasks**: task metadata
- **baseline_trials**: subject model trials under 3 conditions
- **task_subsets**: classification per model
- **figurative_descriptions**: multi-variant figurative clues
- **figurative_trials**: per-variant, per-model, per-k results

## Grid Representation

Same as marc-from-larc:
- Colors 0-9 → single-char codes: `.` (black), `B` (blue), `R` (red), `G` (green), `Y` (yellow), `X` (grey), `M` (magenta), `O` (orange), `A` (azure), `W` (maroon)
- Rows: space-separated codes, one per line

## Key Design Decisions

### Execution Model: Claude Code CLI Subagents
- All Claude-powered work (Phases 1-3, 6, 8) runs via Claude Code subagents — **no API calls**
- Bert has a Max plan, so all Claude usage is covered
- CLI invocation: `claude -p <prompt> --tools "" --model opus --output-format text --no-session-persistence`
- **Subagents use no tools** (`--tools ""`) — all reasoning is pure Opus 4.6 chain-of-thought
- `--bare` flag cannot be used (it skips keychain auth and fails)
- Orchestrator scripts (`solve.py`, `distill.py`, `validate.py`, `generate_figurative.py`, `generate_alternatives.py`) format prompts, spawn subagents in batches, parse responses, write to DB
- Each subagent is a fresh instance with no shared state — this is a feature for validation (Phase 3)

### Subject Models Are Pluggable (Phases 4-8)
- The MARC property is model-relative, so Phases 4-8 are designed to run independently per model
- To add a new subject model: add an entry to `config.yaml`, then run the Phase 4-8 commands with `--model <name>`
- All result tables (`baseline_trials`, `task_subsets`, `figurative_trials`) are keyed by `model_name` — results for different models coexist cleanly
- This means the same set of validated descriptions + figurative variants can be tested against an arbitrary number of subject models over time

### Claude as Solver + Description Generator
- Claude Code subagents solve ARC-AGI2 tasks, producing detailed reasoning traces
- These traces are then distilled into LARC-format see/do/grid descriptions by further subagents
- This replaces LARC's crowdsourcing approach with a scalable, high-quality alternative
- Validation step (fresh subagent, no prior context) ensures descriptions are genuinely language-complete

### Two-Pass LLM Protocol (for subject models only)
- Pass 1: Subject model reasons freely → ANSWER block
- Pass 2: Extraction model (gpt-oss-120b) converts to structured JSON
- Critical for smaller subject models (reduces parse errors significantly)
- NOT used for Claude subagent work — subagents return structured output directly

### Conceptual Digestion (inherited)
Before predicting, models identify:
1. CAST: What roles do grid elements play?
2. AFFORDANCES: What can each element do?
3. TRANSFORMATION RULE: What maps input to output?

### MARC Property (inherited)
A task has the MARC property (for a given model) when:
1. Examples alone FAIL
2. Figurative description alone FAILS
3. Figurative + examples SUCCEEDS (for some minimum k)

## LLM Infrastructure

### Claude Code CLI (Phases 1-3, 6, 8)
- All Claude work runs as subagents inside Claude Code — no API keys needed
- Covered by Bert's Max plan
- Subagents are spawned by orchestrator scripts in parallel batches

### MindRouter (University of Idaho HPC) (Phases 4-5, 7)
- Endpoint: `https://mindrouter.uidaho.edu/v1` (OpenAI-compatible)
- API key env var: `MINDROUTER_API_KEY`
- Subject models: gpt-oss-120b, gpt-oss-20b, qwen3.5-400b
- Always use `--concurrency 8` for batch jobs

## Reference: marc-from-larc

The parent project lives at `/Users/bbaum/Documents/claude-projects/marc-from-larc/`.
Key files to consult for implementation patterns:
- `prompts.py` — prompt templates (Conceptual Digestion, conditions, two-pass)
- `models.py` — LLM API abstraction
- `grids.py` — grid rendering
- `schema.sql` — database schema
- `collect.py` — baseline collection pattern (ThreadPoolExecutor)
- `test_figurative.py` — figurative testing + MARC verification
- `methodology.md` — detailed writeup of the full approach
