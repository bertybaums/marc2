# MARC2: Metaphor Abstraction and Reasoning Corpus v2

MARC2 extends the [MARC-from-LARC](https://github.com/bertybaums/marc-from-larc) methodology to the [ARC-AGI2](https://github.com/arcprize/ARC-AGI-2) dataset, producing a corpus of figurative language puzzles where metaphorical descriptions help AI models solve abstract reasoning tasks they cannot solve from examples alone.

**Date:** March 26, 2026

## The MARC Property

A task has the **MARC property** (for a given model) when:

1. **Examples alone fail** — the model cannot solve the task from input/output grid examples
2. **Figurative description alone fails** — the metaphor is too ambiguous without examples
3. **Figurative + examples succeeds** — the metaphor triggers an "aha" moment when combined with examples

This three-way conjunction identifies tasks where figurative language provides genuine complementary information — not redundant with examples, not sufficient on its own, but synergistic.

## Results

| Phase | Result |
|-------|--------|
| Claude Opus 4.6 solves ARC-AGI2 training tasks | 865/1,000 (86.5%) |
| Distill reasoning into language-complete descriptions | 865/865 (100%) |
| Validate descriptions (fresh solver, no examples) | 791/865 (91.4%) |
| Baseline 3-condition testing on gpt-oss-120b | 2,373 trials, 0 errors |
| Task classification | 350 MARC-eligible (44.3%) |
| Generate figurative descriptions (original + 15 alternatives) | 1,910 clues |
| **MARC-verified puzzles** | **104 puzzles, 824 valid clues** |

### Baseline Accuracy (gpt-oss-120b)

| Condition | Accuracy |
|-----------|----------|
| examples_only | 25.8% |
| language_only | 58.2% |
| both | 51.5% |

Language descriptions dramatically outperform examples alone (+32.4 percentage points).

### MARC Yield by Source Domain

15 domains tested: biology, cooking, music, sports, weather, architecture, warfare, theater, gardening, astronomy, ocean/sailing, electronics, mythology, dance, geology. All domains produce MARC-valid clues (41–52% yield per domain). Opacity-guided metaphor generation improved overall MARC yield from 29.7% to 47.6%.

## Architecture

```
═══ UPPER BOUND: Claude Opus 4.6 establishes what's solvable ═══

ARC-AGI2 tasks (1,000 training)
    ↓
Phase 1: Claude solves tasks via CLI subagents → 865 solved (86.5%)
    ↓
Phase 2: Distill reasoning → language-complete descriptions (see/do/grid)
    ↓
Phase 3: Validate descriptions (fresh subagent, no examples) → 791 validated

═══ LOWER BOUND: Subject model reveals where figurative language helps ═══

Phase 4: Baseline testing on gpt-oss-120b (examples_only / language_only / both)
    ↓
Phase 5: Task classification → 350 MARC-eligible
    ↓
Phase 6: Generate figurative descriptions (original + 15 domain alternatives)
    ↓
Phase 7: Test figurative on gpt-oss-120b → 104 MARC-verified puzzles
    ↓
Phase 8: Verify alternatives → 824 total MARC-valid clues
    ↓
Phase 9: Analysis, inspector, HuggingFace export
```

## Execution Model

All Claude-powered work (Phases 1–3, 6, 8) runs via **Claude Code CLI subagents** — no API calls needed. Covered by a Max plan.

```bash
claude -p <prompt> --tools "" --model opus --output-format text --no-session-persistence
```

Subject model testing (Phases 4–5, 7–8) uses **MindRouter** (University of Idaho HPC) via OpenAI-compatible API.

## Quickstart

```bash
# Setup
bash setup.sh                    # Clone ARC-AGI2, install deps, init DB
python tasks.py init              # Load 1,120 tasks into DB

# Phase 1-3: Solve + Distill + Validate (Claude subagents)
python solve.py solve --split training --batch-size 10
python distill.py distill --batch-size 10
python validate.py validate --batch-size 10

# Phase 4-5: Baseline testing + classification (MindRouter)
export MINDROUTER_API_KEY="..."
python collect.py run --model gpt-oss-120b --concurrency 24
python subset.py classify --model gpt-oss-120b

# Phase 6-8: Figurative generation + testing
python generate_figurative.py generate --batch-size 10
python test_figurative.py test --model gpt-oss-120b --concurrency 24
python test_figurative.py verify-marc --model gpt-oss-120b
python generate_alternatives.py generate --batch-size 10
python generate_alternatives.py verify --model gpt-oss-120b --concurrency 24

# Phase 9: Analysis + export
python analyze.py                 # → report.html
python inspector.py inspect       # → inspect.html
python inspector.py compare       # → compare.html
python export_hf_dataset.py       # → hf_dataset/
```

## Data Access

| Resource | Link |
|----------|------|
| **HuggingFace Dataset** | [bertybaums/marc2](https://huggingface.co/datasets/bertybaums/marc2) |
| **SQLite Database** (311MB) | [GitHub Release v1.0.0](https://github.com/bertybaums/marc2/releases/tag/v1.0.0) |
| **DOI** | *pending — Zenodo integration* |

## Project Structure

```
marc2/
├── PLAN.md / DESIGN.md      # Detailed plan and architecture
├── config.yaml               # MindRouter model configs
├── schema.sql                # SQLite schema
├── db.py                     # Database helpers
├── grids.py                  # Grid rendering (text ↔ PNG)
├── tasks.py                  # ARC-AGI2 task loading
├── models.py                 # MindRouter API client
├── prompts.py                # Prompt templates (all phases)
├── utils.py                  # Shared utilities
├── solve.py                  # Phase 1: Solve via Claude subagents
├── distill.py                # Phase 2: Distill reasoning → descriptions
├── validate.py               # Phase 3: Validate descriptions
├── collect.py                # Phase 4: Baseline 3-condition testing
├── subset.py                 # Phase 5: Task classification
├── generate_figurative.py    # Phase 6: Generate figurative descriptions
├── test_figurative.py        # Phase 7: Test figurative + verify MARC
├── generate_alternatives.py  # Phase 8: Domain-diverse alternatives
├── analyze.py                # Phase 9: Plotly report
├── inspector.py              # Phase 9: Interactive HTML inspector
└── export_hf_dataset.py      # Phase 9: HuggingFace Parquet export
```

## Comparison to MARC-from-LARC

| Metric | MARC-from-LARC | MARC2 |
|--------|:-:|:-:|
| Source tasks | 400 (LARC) | 1,000 (ARC-AGI2) |
| Descriptions | Crowdsourced | Claude-generated + validated |
| Validated descriptions | 400 | 791 |
| MARC-eligible | ~95 | 350 |
| MARC-verified puzzles | 45 | 104 |
| Alternative clues | 250 | 824 |
| Source domains | 7 | 15 |
| Cost | $0 | $0 |

## Citation

```bibtex
@dataset{baum2026marc2,
  title={MARC2: Metaphor Abstraction and Reasoning Corpus v2},
  author={Baum, Bert},
  year={2026},
  url={https://github.com/bertybaums/marc2}
}
```

## License

MIT
