# MARC2 Implementation Plan

## Execution Model: Claude Code CLI + Subagents

All Claude-powered work (solving, distilling, validating, figurative generation) runs inside
Claude Code via subagents — **no Anthropic API calls**. This is possible because Bert has a
Max plan. Each phase script is an orchestrator that:

1. Reads tasks from the DB and formats prompts
2. Spawns Claude Code subagents (parallel where possible) with those prompts
3. Parses subagent responses and writes results back to the DB

Subject model testing (Phases 4-8) still uses MindRouter API calls — those models aren't
available through Claude Code. Critically, subject models are **pluggable**: add a new model
to `config.yaml` and run Phases 4-8 with `--model <name>`. All result tables are keyed by
`model_name`, so results for different subject models coexist cleanly. This means the same
corpus of validated descriptions + figurative variants can be tested against any number of
subject models over time.

---

## Yield Funnel

| Stage | Estimated | Actual | What filters out |
|-------|----------:|-------:|------------------|
| ARC-AGI2 public tasks | 1,120 | 1,120 | — |
| Claude solves correctly (training) | ~700 (70%) | **865 (86.5%)** | Accuracy far exceeded estimate |
| Evaluation split | ~84 (70%) | *skipped* | Not worth token cost given large training yield |
| Validated language-complete descriptions | ~650–700 | **791 (91.4%)** | 72 failed validation, 2 errors |
| Baseline tested (gpt-oss-120b) | 791×3 conditions | **2,373 trials, 0 errors** | ex_only 25.8%, lang_only 58.2%, both 51.5% |
| MARC-eligible (gpt-oss-120b) | ~200–350 | **350 (44.3%)** | 296 lang_suff + 54 both_req; 204 ex_suff, 237 unsolvable |
| Figurative generated | ~180–300 | **350 (100% of eligible)** | Zero errors |
| MARC-verified (figurative works) | ~80–150 | **104** | 172 too transparent, 74 never succeed |
| Alternative figurative clues | ~400–800 | **720 MARC-valid** (of 1,560 generated) | 47.6% MARC yield across 15 domains |

The initial "LARC-equivalent" database is the **validated descriptions** — that's the new
thing this project produces. With 82.1% solve rate on training (vs 70% estimated), the
downstream yield should be significantly larger than initially projected.

---

## Phase 1: Solve ARC-AGI2 with Claude

### Goal
Have Claude solve all 1,120 public ARC-AGI2 tasks via subagents, capturing full reasoning
traces for every attempt.

### Implementation: `solve.py`

```bash
python solve.py solve --split training --batch-size 10
python solve.py solve --split evaluation --batch-size 10
python solve.py retry --batch-size 10              # second attempts on failures
python solve.py report
```

**How it works:**
- `solve.py` reads task JSON files and formats prompts (training I/O pairs + test input)
- Spawns Claude Code CLI subagents via `claude -p <prompt> --tools "" --model opus --output-format text --no-session-persistence`
- Subagents use no tools (pure CoT reasoning) — all of Opus 4.6's chain-of-thought
- Each subagent receives one task, reasons through it, returns predicted grid in ANSWER block
- Script parses response, checks correctness, stores everything in `solve_trials`

**Prompt design:**
- Present training I/O pairs + test input (text grid format with single-char color codes)
- Use Conceptual Digestion framing: identify CAST, AFFORDANCES, TRANSFORMATION RULE
- Request step-by-step reasoning → ANSWER block with grid
- Provide expected output dimensions as a hint
- Allow up to 2 attempts per task (retry uses different approach prompt)

**Storage:** `solve_trials` table
```sql
solve_trials (
    trial_id     INTEGER PRIMARY KEY,
    task_id      INTEGER REFERENCES tasks(task_id),
    attempt      INTEGER,       -- 1 or 2
    prompt_text  TEXT,
    reasoning    TEXT,           -- full reasoning trace (the gold we're mining)
    predicted_grid TEXT,        -- JSON 2D array
    correct      INTEGER,       -- 0 or 1
    cell_accuracy REAL,
    error        TEXT
)
```

**Key decisions:**
- Subagents run in parallel batches of 10 (balances throughput vs resource usage)
- Store ALL reasoning traces, even for incorrect answers (useful for error analysis)
- Retry prompt includes previous wrong answer so the model tries a different approach
- No two-pass extraction needed: subagent instructions request structured output directly
- `--bare` flag cannot be used (skips keychain auth); `--no-session-persistence` used instead

### Training Split Results (March 25, 2026)

| Metric | Count | Pct |
|--------|------:|----:|
| Tasks attempted | 1,000 | — |
| **Correct (attempt 1)** | **865** | **86.5%** |
| Wrong | 68 | 6.8% |
| Errors (timeouts) | 67 | 6.7% |

**Timing:**
- Average ~20 seconds per task
- ~5 tasks/minute throughput at batch_size=10
- Total wall clock: ~3.5 hours (split across two token sessions)

**Key finding:** 86.5% solve rate significantly exceeds the 70% estimate. 865 solved
tasks gives the downstream pipeline substantially more material than the ~784 projected.

**Decisions:**
- Retries (attempt 2) were tested but yielded only 1/11 — not worth the token cost
- Evaluation split (120 harder tasks) skipped — 865 training tasks provide ample
  material, and eval tasks showed 0% in early trials

---

## Phase 2: Distill Reasoning into Language-Complete Descriptions

### Goal
Convert Claude's successful reasoning traces into LARC-format `see/do/grid` descriptions.

### Implementation: `distill.py`

```bash
python distill.py distill --batch-size 20
python distill.py report
```

**How it works:**
- For each correctly solved task, spawn a subagent with:
  1. The original task (all training pairs + test pair with correct output)
  2. Claude's reasoning trace from Phase 1
  3. Instructions to produce see/do/grid description
- Subagent returns structured description; script parses and stores

**Quality criteria (from LARC):**
- "Language-complete" = a competent person could solve ANY instance given only the
  description and a new input
- `do_description` must be procedural and precise, not vague
- No references to specific training examples ("in example 2, the blue square...")

### Results (March 25, 2026)

| Metric | Result |
|--------|--------|
| Tasks distilled | 865/865 |
| Errors | 0 |
| Avg see_description | 391 chars |
| Avg do_description | 678 chars |
| Avg grid_description | 299 chars |

100% success rate. Average do_description of 678 chars indicates rich, procedural descriptions.

---

## Phase 3: Validate Descriptions

### Goal
Verify descriptions are genuinely language-complete: a *fresh* subagent can solve the task
from description + test input alone (no training examples, no prior context).

### Implementation: `validate.py`

```bash
python validate.py validate --batch-size 10
python validate.py revise --max-revisions 2 --batch-size 10  # retry failures
python validate.py report
```

**Protocol:**
1. Spawn subagent with only: see + do + grid description + test input (NO training examples)
2. If correct output → validated
3. If wrong → flag for revision

**Revision loop:**
- For failed validations, spawn a new distill subagent with feedback:
  "Your description produced [wrong output]. Correct is [correct output]. Revise."
- Cap at 2 revision attempts, then discard

### Results (March 25, 2026)

| Metric | Result |
|--------|--------|
| Descriptions tested | 863/865 |
| **Validated (passed)** | **791 (91.4%)** |
| Failed | 72 |
| Errors | 2 |

91.4% validation rate far exceeded the projected ~85%. **791 language-complete descriptions**
form the "LARC equivalent" corpus — the core artifact of MARC2.

**Decision:** Revision loop not yet run on the 72 failures. Could recover ~30-50 additional
descriptions but may not be worth the token cost given the already-large corpus.

---

## Phase 4: Baseline 3-Condition Testing (Subject Models)

### Goal
Test subject models (gpt-oss-120b, gpt-oss-20b, etc.) under three conditions — this is
where we switch from Claude Code subagents to MindRouter API calls.

### Implementation: `collect.py`

```bash
python collect.py --model gpt-oss-120b --concurrency 8
python collect.py --model gpt-oss-20b --concurrency 8
```

**Conditions** (for tasks with validated descriptions):
- **examples_only**: Training grid pairs + test input
- **language_only**: Validated see/do/grid description + test input
- **both**: Description + all training examples + test input

**Two-pass protocol** (inherited from marc-from-larc):
- Pass 1: Subject model reasons freely
- Pass 2: gpt-oss-120b extracts structured JSON

Only runs on tasks that have validated descriptions from Phase 3.

### Results: gpt-oss-120b (March 25, 2026)

| Condition | Correct | Accuracy |
|-----------|---------|----------|
| examples_only | 204/791 | 25.8% |
| language_only | 460/791 | 58.2% |
| both | 407/791 | 51.5% |
| Errors | 0 | — |

**Key findings:**
- 2,373 trials, zero errors (concurrency=24, two-pass extraction)
- Language descriptions dramatically outperform examples alone (+32.4pp)
- Surprisingly, both < language_only — adding examples hurts on some tasks,
  possibly because examples confuse gpt-oss-120b or compete with the description

---

## Phase 5: Task Classification

### Implementation: `subset.py`

Classify each task (per model) into:
- **examples_sufficient**: Solved by examples alone
- **language_sufficient**: Not solved by examples, but solved by description alone
- **both_required**: Requires both description and examples
- **unsolvable**: None of the conditions work

### Results: gpt-oss-120b (March 25, 2026)

| Subset | Count | % |
|--------|------:|---:|
| examples_sufficient | 204 | 25.8% |
| language_sufficient | 296 | 37.4% |
| both_required | 54 | 6.8% |
| unsolvable | 237 | 30.0% |
| **MARC-eligible** | **350** | **44.3%** |

350 MARC-eligible tasks — at the top end of the 200-350 estimate.

---

## Phase 6: Generate Figurative Descriptions

### Implementation: `generate_figurative.py`

For tasks classified as language_sufficient or both_required:
- Spawn Claude Code subagents to generate metaphorical see/do/grid descriptions
- Input: literal description + task examples
- Quality check: Jaccard similarity to literal `do_description` must be < 0.6

### Results (March 25, 2026)

| Metric | Result |
|--------|--------|
| Figurative descriptions generated | **350/350 (100%)** |
| Errors | 0 |
| Avg metaphor length | 81 chars |
| Avg figurative_do length | 435 chars |

---

## Phase 7: Test Figurative Descriptions + Verify MARC

### Implementation: `test_figurative.py`

```bash
python test_figurative.py test --model gpt-oss-120b --concurrency 24
python test_figurative.py verify-marc --model gpt-oss-120b
```

Test each figurative description at k=0, k=1, ..., k=num_train (MindRouter API calls).
Verify MARC property: (1) examples fail, (2) figurative alone fails, (3) figurative + k examples succeeds.

### Results: gpt-oss-120b (March 26, 2026)

| Metric | Result |
|--------|--------|
| Figurative trials | 1,505 |
| k=0 (figurative only) | 172/350 (49.1%) |
| k>0 (figurative+examples) | 561/1,155 (48.6%) |
| **Valid MARC puzzles** | **104** |
| Non-MARC: figurative alone succeeds | 172 |
| Non-MARC: figurative never succeeds | 74 |
| Errors | 4 |

**Key finding:** 172/350 tasks where the figurative description alone succeeds means those
metaphors are too transparent — they don't need examples to decode. The 104 MARC-verified
puzzles are the sweet spot: evocative enough to trigger insight when combined with examples,
but not so literal that they give away the answer independently.

---

## Phase 8: Alternative Figurative Clues

### Implementation: `generate_alternatives.py`

For each MARC-verified puzzle:
- Spawn Claude Code subagents to generate 15 alternatives from different source domains
- Prompt includes **opacity guidance** targeting the MARC sweet spot: ambiguous enough to
  need examples, evocative enough to trigger insight with them
- Test each via MindRouter API (k=0 through k=num_train)

### Results (March 26, 2026)

| Metric | Result |
|--------|--------|
| Alternatives generated | 1,560 (15 per puzzle × 104 puzzles) |
| Alternatives tested | 1,513 |
| **MARC-valid** | **720 (47.6%)** |
| Too transparent | 459 (30.3%) |
| Too opaque | 334 (22.1%) |
| Source domains | 15 |
| Avg MARC-valid per puzzle | ~6.9 |

**Key findings:**
- Opacity guidance improved MARC yield from 29.7% (Phase 7) to 47.6% — a 60% improvement
- All 15 source domains produce MARC-valid clues (41-52% per domain)
- Weather (52%) and warfare (50%) perform best; geology and ocean/sailing (41%) lowest
- **Total MARC corpus: 824 clues** (104 original + 720 alternatives) across 104 puzzles

---

## Phase 9: Analysis and Export

### Implementation: `analyze.py`, `inspector.py`

- Generate report.html with Plotly charts
- Generate inspect.html (tabbed variant browser)
- Generate compare.html (variant comparison view)
- Export HuggingFace Parquet dataset

---

## Implementation Order

### Sprint 1: Setup & Infrastructure
1. Clone ARC-AGI2 dataset (`bash setup.sh`)
2. Port and adapt from marc-from-larc:
   - `db.py`, `grids.py`, `utils.py` (minimal changes)
   - `tasks.py` (adapt for ARC-AGI2 directory layout)
   - `schema.sql` (done — adds solve_trials + descriptions tables)
   - `config.yaml` (done — no Anthropic API config needed for subagents)
   - `prompts.py` (add solve/distill/validate prompt templates; inherit condition prompts)

### Sprint 2: Solve Pipeline (Phases 1-3)
3. Implement `solve.py` — subagent orchestrator for solving
4. Implement `distill.py` — subagent orchestrator for description generation
5. Implement `validate.py` — subagent orchestrator for validation

### Sprint 3: MARC Pipeline (Phases 4-8)
6. Port `collect.py`, `subset.py` from marc-from-larc (MindRouter API calls)
7. Port `generate_figurative.py` (subagents), `test_figurative.py` (MindRouter)
8. Port `generate_alternatives.py` (subagents for generation, MindRouter for testing)

### Sprint 4: Analysis & Export (Phase 9)
9. Port `analyze.py`, `inspector.py`, export script

## Cost

- **Phases 1-3, 6, 8 (Claude work):** Covered by Max plan — all via Claude Code subagents
- **Phases 4-5, 7 (Subject model testing):** MindRouter (free on U of Idaho HPC)
- **Total out-of-pocket: $0**
