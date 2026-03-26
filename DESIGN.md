# MARC2 Design Document

## Core Innovation: Establishing the Spectrum Endpoints

The MARC property is **model-relative**: a task has the MARC property for a given model when
that model can't solve it from examples alone, can't solve it from a figurative clue alone,
but *can* solve it when given both together. What counts as "solvable" depends entirely on
which model you ask.

MARC2 exploits this by using one of the strongest models available (Claude Opus 4.6, ~70% on
ARC-AGI2) to establish **both ends of the spectrum**:

- **Upper bound (Phases 1-3):** Claude solves ARC-AGI2 tasks and distills its reasoning into
  language-complete descriptions. These descriptions capture the transformation rule in a way
  that Claude demonstrably understands — they're gold-standard *because* they come from a
  model that can actually do the task.

- **Lower bound (Phases 4-8):** Smaller subject models (gpt-oss-120b, 20b, etc.) are tested
  on the same tasks. Many tasks Claude solved easily will stump these models when given only
  examples. The language-complete descriptions become the source material for generating
  figurative alternatives, which are then tested on the subject models for the MARC property.

This framing means the MARC2 pipeline isn't just replacing crowdsourcing with Claude — it's
*leveraging the capability gap between models* to systematically produce MARC puzzles. The
bigger the gap between Claude and the subject model, the more tasks enter the MARC-eligible
window.

Additional benefits over marc-from-larc:
1. **~3x more tasks** (1,120 vs 400 LARC tasks)
2. **Higher description quality** (Claude's reasoning traces are detailed and precise vs. crowdsourced descriptions of variable quality)
3. **Harder tasks** (ARC-AGI2 was calibrated for 66% human accuracy, vs ~85% for ARC-AGI1)
4. **Self-contained pipeline** (no dependency on external crowdsourcing)

## Architecture

### Execution Model: Claude Code CLI Subagents

All Claude-powered work runs inside Claude Code via subagents — **no Anthropic API calls**.
This is made possible by Bert's Max plan. The pattern is:

1. **Orchestrator script** (e.g., `solve.py`) reads tasks from DB, formats prompts
2. **Spawns Claude Code subagents** in parallel batches with those prompts
3. **Parses subagent responses** and writes results back to DB

Subject model testing (Phases 4-8) still uses MindRouter API calls — those models aren't
available through Claude Code.

**Subagent interface:** Each subagent receives a self-contained prompt (task data + instructions)
and returns structured output. The orchestrator scripts handle batching, DB writes, and
error recovery. Subagents have no persistent state between tasks.

### What to Port from marc-from-larc

These files can be ported with minimal changes:
- **`db.py`** — SQLite helpers are generic. Just update default DB name to `marc2.db`.
- **`grids.py`** — Grid rendering is identical (same 0-9 color space, same text codes, same PNG rendering).
- **`utils.py`** — Model config lookup, serialization. Port as-is.
- **`models.py`** — Only needs OpenAI-compat client for MindRouter subject models. No Anthropic SDK needed (Claude work is via subagents, not API). The two-pass protocol applies only to subject models.

These need significant adaptation:
- **`tasks.py`** — ARC-AGI2 has a different directory layout (`data/arc-agi2/training/` and `data/arc-agi2/evaluation/` instead of LARC's nested structure). No LARC description files to load.
- **`prompts.py`** — Need new prompts for solving and distilling (used as subagent instructions). Condition prompts (examples_only, language_only, both, figurative) inherited for MindRouter phases.
- **`schema.sql`** — Add `solve_trials` and `descriptions` tables. Inherit the rest.

### New Components

#### `solve.py` — The Solver (Subagent Orchestrator)

**Design principles:**
- Each subagent receives one task and solves it independently
- Claude sees the same information a human would: training I/O pairs + test input
- Reasoning is free-form (no forced structure beyond Conceptual Digestion)
- No two-pass extraction needed — subagent instructions request structured output directly
- Store everything: prompt, full reasoning, predicted grid, correctness

**Prompt structure for solving (sent to subagent):**
```
You are solving a grid transformation puzzle.
[COLOR_KEY]
[Training examples formatted as text grids]
[Test input formatted as text grid]

[CONCEPTUAL_DIGESTION]
[REASONING_OUTPUT instructions]
```

**Retry strategy:**
- Attempt 1: Standard prompt
- Attempt 2 (if attempt 1 fails): Include hint about grid dimensions
  (many failures come from wrong output size)
- Both attempts stored; only need one correct for downstream phases

#### `distill.py` — Reasoning → Descriptions

**The key challenge:** Converting free-form reasoning into structured, general descriptions.

**Prompt structure for distilling:**
```
You solved the following grid transformation puzzle correctly.

[Task with all training pairs and test pair (including correct output)]

Your reasoning was:
[Full reasoning trace from solve_trials]

Now write a language-complete description of this task. A description is
"language-complete" if someone could solve ANY new instance of this task
given only your description and a new input grid.

Write three components:

SEE: What structures, patterns, or objects are visible in the input grids?
Describe the visual vocabulary — what kinds of elements appear and how they
are arranged. Be specific about colors, shapes, positions, and relationships.

DO: What is the precise transformation rule that maps input to output?
This must be procedural and complete — a step-by-step recipe that works
for any valid input, not just the examples you saw. Do not reference
specific training examples.

GRID: How do the output grid dimensions relate to the input? (same size,
cropped, expanded, etc.)
```

**Quality heuristics:**
- `do_description` should be >50 words (too short = too vague)
- `do_description` should not reference "example 1" or "the blue square in the corner" (too specific)
- `see_description` should mention colors and spatial relationships

#### `validate.py` — Verification

**Core idea:** If a description is truly language-complete, a model should solve the task from the description + test input alone.

**Important:** The validation subagent has NO context from the solving or distilling subagents —
it's a fresh Claude instance. This provides natural separation: if a description is only
understandable because of implicit shared context from the solving step, the validation
subagent will fail on it, which is exactly what we want.

**Revision loop:**
```
Your previous description was tested. Given only the description and a new
test input, the model predicted:
[wrong output grid]

The correct output was:
[correct output grid]

The description was not precise enough. Please revise it to be more explicit
about the transformation rule. Common issues:
- Ambiguous directional language (which "left"?)
- Missing edge cases (what happens at grid boundaries?)
- Implicit assumptions about grid structure
```

## Database Schema Extensions

```sql
-- New: Claude's solving attempts
CREATE TABLE solve_trials (
    trial_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id       INTEGER NOT NULL REFERENCES tasks(task_id),
    model         TEXT NOT NULL,
    attempt       INTEGER NOT NULL DEFAULT 1,
    prompt_text   TEXT,
    reasoning     TEXT,
    predicted_grid TEXT,
    correct       INTEGER,
    cell_accuracy REAL,
    error         TEXT,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(task_id, model, attempt)
);

-- New: Distilled language-complete descriptions
CREATE TABLE descriptions (
    desc_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          INTEGER NOT NULL REFERENCES tasks(task_id) UNIQUE,
    source_trial_id  INTEGER REFERENCES solve_trials(trial_id),
    see_description  TEXT NOT NULL,
    do_description   TEXT NOT NULL,
    grid_description TEXT NOT NULL,
    validated        INTEGER DEFAULT 0,
    validation_model TEXT,
    validation_correct INTEGER,
    revision_count   INTEGER DEFAULT 0,
    generation_model TEXT NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inherited from marc-from-larc (unchanged)
CREATE TABLE tasks ( ... );
CREATE TABLE baseline_trials ( ... );
CREATE TABLE task_subsets ( ... );
CREATE TABLE figurative_descriptions ( ... );
CREATE TABLE figurative_trials ( ... );
```

## Scaling Considerations

### ARC-AGI2 is Harder
- Human accuracy: 66% (vs ~85% for ARC-AGI1)
- Subject models (gpt-oss-120b) will likely solve fewer tasks → more MARC-eligible candidates
- More unsolvable tasks expected → larger "both_required" subset

### Expected Yields

Projections updated with Phase 1 training results (March 25, 2026):

| Stage | marc-from-larc | MARC2 (projected) | MARC2 (actual) |
|-------|:-:|:-:|:-:|
| Public tasks | 400 (LARC) | 1,120 (ARC-AGI2) | 1,120 |
| Claude solves correctly | n/a | ~784 (70%) | **865 training (86.5%)**, eval skipped |
| Validated descriptions | 400 (crowdsourced) | ~650-700 | **791 (91.4% of 865)** |
| MARC-eligible (lang_suff + both_req) | ~95 | ~200-350 | **350 (44.3% of 791)** |
| Figurative generated | 93 | ~180-300 | **350** |
| MARC-verified (120b) | 45 (LARC) | ~80-150 | **104** |
| Alternative clues | 250 | ~400-800 | **720 MARC-valid** (of 1,560 generated) |

Note: The initial "LARC-equivalent" database is the validated descriptions — that's
the new artifact this project produces. With 82.1% solve rate (vs 70% projected), the
downstream pipeline has ~37 more tasks entering the funnel than expected.

The harder ARC-AGI2 tasks should produce a higher MARC yield ratio because subject models
will struggle more with examples_only, creating more room for figurative descriptions to
help. The key insight: Claude (Opus 4.6) solving 82% while gpt-oss-120b solves ~32% of
ARC-AGI1 means a large gap where tasks are "solvable in principle" (Claude proved it) but
"not solvable from examples alone" for the subject model — exactly the MARC-eligible window.

## Relationship to Complementary Inquiry

MARC2 strengthens the broader research program:
- **More MARC puzzles** → more statistical power for mechanistic interpretability
- **ARC-AGI2 difficulty** → puzzles where figurative language genuinely adds information
- **Claude-generated descriptions** → demonstrates the Time Capsule method can scale
- **Domain-diverse alternatives** → richer data for probing how metaphor activates abstract reasoning circuits
