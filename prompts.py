"""Prompt templates for all MARC2 phases.

Phases 1-3 (solve/distill/validate): Used to format prompts for Claude Code subagents.
Phases 4-8 (baseline/figurative testing): Used for MindRouter API calls to subject models.
"""

import json

from grids import (
    COLOR_KEY, format_examples_text, format_test_input_text,
    grid_to_text, grid_to_png_bytes,
)
from models import make_image_content_block


# ============================================================================
# Output format instructions (for subject model testing, Phases 4-8)
# ============================================================================

REASONING_OUTPUT = """\
Think step by step. After your reasoning, you MUST write out the complete predicted output grid using these single-character color codes:
""" + COLOR_KEY + """

At the very end of your response, write the complete output grid inside a clearly labeled block like this:
ANSWER:
. B R
G Y .
(one row per line, characters separated by spaces)"""

EXTRACTION_PROMPT = """\
Below is reasoning about a grid puzzle. It should contain an ANSWER block with the output grid.

<reasoning>
{reasoning}
</reasoning>

Extract the output grid from the ANSWER block above. Convert it to JSON format.
Do NOT re-analyze. Just extract and format the grid that was already produced.
Respond with ONLY this JSON object, nothing else:
{{"reasoning":"<one sentence summary>","output_grid":[[".","."],["B","R"]]}}

Color codes: .=black B=blue R=red G=green Y=yellow X=grey M=magenta O=orange A=azure W=maroon"""

CONCEPTUAL_DIGESTION = """\
Before predicting the output, perform a Conceptual Digestion:
1. Identify the CAST: What roles do different grid elements play?
2. Identify the AFFORDANCES: What can each element do or have done to it?
3. Identify the TRANSFORMATION RULE: What is the precise rule that maps input to output?

Then apply the rule to the test input to predict the output grid."""


# ============================================================================
# Phase 1: Solve prompts (for Claude Code subagents)
# ============================================================================

def build_solve_prompt(arc_task):
    """Build the prompt text for a Claude Code subagent to solve an ARC-AGI2 task.

    Returns a single string (not messages list) — the subagent receives this as its task.
    """
    train = arc_task["train"]
    test_input = arc_task["test"][0]["input"]
    expected_output = arc_task["test"][0]["output"]
    rows, cols = len(expected_output), len(expected_output[0])

    prompt = f"""\
You are solving an ARC-AGI2 grid transformation puzzle.

{COLOR_KEY}

Study the training examples below to discover the transformation pattern, then apply it to predict the output for the test input.

{format_examples_text(train)}

{format_test_input_text(test_input)}

The expected output grid dimensions are {rows} rows x {cols} columns.

{CONCEPTUAL_DIGESTION}

Think through this carefully step by step. After your reasoning, write the complete predicted output grid.

IMPORTANT: Your final answer MUST be in this exact format:

ANSWER:
. B R
G Y .
(one row per line, single-character color codes separated by spaces)"""

    return prompt


def build_solve_retry_prompt(arc_task, prev_reasoning, prev_grid_text):
    """Build a retry prompt for Phase 1 when the first attempt was wrong.

    Includes the previous (wrong) attempt so the subagent can try a different approach.
    """
    train = arc_task["train"]
    test_input = arc_task["test"][0]["input"]
    expected_output = arc_task["test"][0]["output"]
    rows, cols = len(expected_output), len(expected_output[0])

    prompt = f"""\
You are solving an ARC-AGI2 grid transformation puzzle. Your previous attempt was incorrect.

{COLOR_KEY}

Study the training examples below to discover the transformation pattern, then apply it to predict the output for the test input.

{format_examples_text(train)}

{format_test_input_text(test_input)}

The expected output grid dimensions are {rows} rows x {cols} columns.

## Previous (incorrect) attempt

Your previous answer was wrong. Here is what you tried:

{prev_grid_text}

Try a DIFFERENT approach this time. Look for patterns you may have missed.

{CONCEPTUAL_DIGESTION}

Think through this carefully step by step. After your reasoning, write the complete predicted output grid.

IMPORTANT: Your final answer MUST be in this exact format:

ANSWER:
. B R
G Y .
(one row per line, single-character color codes separated by spaces)"""

    return prompt


# ============================================================================
# Phase 2: Distill prompts (for Claude Code subagents)
# ============================================================================

def build_distill_prompt(arc_task, reasoning_trace):
    """Build prompt for distilling a reasoning trace into a language-complete description.

    Returns a single string for the subagent.
    """
    train = arc_task["train"]
    test_input = arc_task["test"][0]["input"]
    test_output = arc_task["test"][0]["output"]

    prompt = f"""\
You are distilling an ARC-AGI2 puzzle solution into a language-complete description.

{COLOR_KEY}

## The Puzzle

{format_examples_text(train)}

Test Input:
{grid_to_text(test_input)}

Correct Test Output:
{grid_to_text(test_output)}

## Successful Reasoning Trace

{reasoning_trace}

## Your Task

Based on the puzzle and the reasoning trace above, write a language-complete description of this transformation. A "language-complete" description means that a competent person could solve ANY instance of this puzzle given only your description and a new input grid — without seeing any examples.

Produce three components:

1. **see_description**: What does the solver see in the input grid? Describe the visual elements, patterns, and structure. Be specific about colors, shapes, positions, and relationships.

2. **do_description**: What transformation should be applied? Write precise, procedural instructions that can be followed step by step. Do NOT reference specific training examples ("in example 2..."). The instructions must generalize to any valid input.

3. **grid_description**: What are the grid-level properties? Describe dimensions, background color, boundaries, coordinate system, or any structural constraints.

Respond with ONLY this JSON:
{{"see_description": "...", "do_description": "...", "grid_description": "..."}}"""

    return prompt


# ============================================================================
# Phase 3: Validate prompts (for Claude Code subagents)
# ============================================================================

def build_validate_prompt(see, do, grid, test_input):
    """Build prompt for validation: solve from description alone (no examples).

    Returns a single string for the subagent.
    """
    prompt = f"""\
You are solving a grid transformation puzzle using only a written description.
You have NOT seen any training examples — only the description and the test input.

{COLOR_KEY}

## Transformation Description

What you see: {see}
What to do: {do}
Grid details: {grid}

{format_test_input_text(test_input)}

Apply the description to the test input and predict the output grid.
Think step by step, then write your answer.

IMPORTANT: Your final answer MUST be in this exact format:

ANSWER:
. B R
G Y .
(one row per line, single-character color codes separated by spaces)"""

    return prompt


def build_revise_prompt(arc_task, reasoning_trace, prev_see, prev_do, prev_grid,
                        wrong_output_text, correct_output_text):
    """Build prompt for revising a description that failed validation."""
    train = arc_task["train"]

    prompt = f"""\
You previously distilled a puzzle description, but a fresh solver using ONLY your description produced the wrong output. Revise your description to be more precise.

{COLOR_KEY}

## The Puzzle

{format_examples_text(train)}

## Original Reasoning Trace

{reasoning_trace}

## Your Previous Description

See: {prev_see}
Do: {prev_do}
Grid: {prev_grid}

## Validation Result

A solver using only your description produced:
{wrong_output_text}

The correct output was:
{correct_output_text}

## Your Task

Revise the description to fix whatever ambiguity or error caused the wrong output. Make the do_description more precise and procedural. Do NOT reference specific training examples.

Respond with ONLY this JSON:
{{"see_description": "...", "do_description": "...", "grid_description": "..."}}"""

    return prompt


# ============================================================================
# Phase 4: Baseline condition prompts (for subject model MindRouter calls)
# ============================================================================

def build_examples_only(arc_task, num_examples=None):
    """Build messages for the examples_only condition."""
    train = arc_task["train"]
    if num_examples is not None:
        train = train[:num_examples]
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. Study the training examples to discover the pattern, then predict the output for the test input.

{COLOR_KEY}

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_language_only(desc_row, arc_task):
    """Build messages for the language_only condition.

    Uses validated description from descriptions table (not tasks table).
    """
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a description of the transformation rule and a test input grid. Apply the rule to predict the output.

{COLOR_KEY}

## Transformation Description

What you see: {desc_row['see_description']}
What to do: {desc_row['do_description']}
Grid details: {desc_row['grid_description']}

{CONCEPTUAL_DIGESTION}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_both(desc_row, arc_task, num_examples=None):
    """Build messages for the both condition (language + examples)."""
    train = arc_task["train"]
    if num_examples is not None:
        train = train[:num_examples]
    test_input = arc_task["test"][0]["input"]

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a description of the transformation rule, training examples, and a test input grid.

{COLOR_KEY}

## Transformation Description

What you see: {desc_row['see_description']}
What to do: {desc_row['do_description']}
Grid details: {desc_row['grid_description']}

{CONCEPTUAL_DIGESTION}

## Training Examples

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


# ============================================================================
# Phase 6: Figurative description generation (for Claude Code subagents)
# ============================================================================

def build_figurative_generation(desc_row, arc_task):
    """Build prompt for generating a figurative/metaphorical description."""
    train = arc_task["train"]

    prompt = f"""\
You are transforming a literal puzzle instruction into a figurative/metaphorical version.

{COLOR_KEY}

## The Puzzle (for reference)

{format_examples_text(train)}

## Literal Description

What you see: {desc_row['see_description']}
What to do: {desc_row['do_description']}
Grid details: {desc_row['grid_description']}

## Your Task

Create a metaphorical version that:
1. Assigns imaginative roles to the grid elements (colors, shapes, positions)
2. Describes the transformation through a metaphorical scenario or narrative
3. Is evocative enough that seeing examples would make the solution click ("Aha!")
4. Is NOT precise enough to solve the puzzle without seeing any examples

Think of metaphors like: "The cat wants to play with the toy it cannot see" or "The floor is lava" or "DNA encodes the steps for replication."

The metaphorical description must still reference grid elements (colors, positions, regions) but through figurative language rather than literal procedural instructions.

Respond with ONLY this JSON:
{{
  "metaphor": "<one-line metaphor concept>",
  "see": "<figurative version of what you see>",
  "do": "<figurative version of what to do>",
  "grid": "<figurative version of grid details>"
}}"""

    return prompt


# ============================================================================
# Phase 7: Figurative testing prompts (for subject model MindRouter calls)
# ============================================================================

def build_figurative_only(fig_row, arc_task):
    """Build messages for testing figurative description alone (k=0)."""
    test_input = arc_task["test"][0]["input"]

    # Handle both split (see/do/grid) and unified (metaphor only) formats
    if fig_row['figurative_see']:
        desc_block = f"""\
What you see: {fig_row['figurative_see']}
What to do: {fig_row['figurative_do']}
Grid details: {fig_row['figurative_grid']}"""
    else:
        desc_block = fig_row['metaphor']

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation and a test input grid.

{COLOR_KEY}

## Metaphorical Description

{desc_block}

{CONCEPTUAL_DIGESTION}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


def build_figurative_with_examples(fig_row, arc_task, num_examples):
    """Build messages for figurative description + k training examples."""
    train = arc_task["train"][:num_examples]
    test_input = arc_task["test"][0]["input"]

    if fig_row['figurative_see']:
        desc_block = f"""\
What you see: {fig_row['figurative_see']}
What to do: {fig_row['figurative_do']}
Grid details: {fig_row['figurative_grid']}"""
    else:
        desc_block = fig_row['metaphor']

    prompt = f"""\
You are solving a grid transformation puzzle. You are given a metaphorical description of the transformation, training examples, and a test input grid.

{COLOR_KEY}

## Metaphorical Description

{desc_block}

{CONCEPTUAL_DIGESTION}

## Training Examples

{format_examples_text(train, num_examples)}

{format_test_input_text(test_input)}

{REASONING_OUTPUT}"""

    return [{"role": "user", "content": prompt}]


# ============================================================================
# Extraction (Pass 2 for subject models)
# ============================================================================

def build_extraction_messages(reasoning_text):
    """Build pass-2 extraction messages from pass-1 reasoning."""
    prompt = EXTRACTION_PROMPT.format(reasoning=reasoning_text)
    return [{"role": "user", "content": prompt}]


# ============================================================================
# Dispatcher
# ============================================================================

def build_prompt(condition, desc_row, arc_task, num_examples=None, fig_row=None):
    """Build prompt messages for any condition.

    Args:
        condition: 'examples_only', 'language_only', 'both',
                   'figurative_only', 'figurative_with_examples'
        desc_row: sqlite3.Row from descriptions table (for language conditions)
        arc_task: dict from ARC JSON
        num_examples: number of training examples to include (None = all)
        fig_row: sqlite3.Row from figurative_descriptions table
    """
    if condition == "examples_only":
        return build_examples_only(arc_task, num_examples)
    elif condition == "language_only":
        return build_language_only(desc_row, arc_task)
    elif condition == "both":
        return build_both(desc_row, arc_task, num_examples)
    elif condition == "figurative_only":
        return build_figurative_only(fig_row, arc_task)
    elif condition == "figurative_with_examples":
        return build_figurative_with_examples(fig_row, arc_task, num_examples)
    else:
        raise ValueError(f"Unknown condition: {condition}")
