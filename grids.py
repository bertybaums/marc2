"""Grid utilities: text <-> grid conversion, PNG rendering, response parsing."""

import base64
import io
import json
import re
from typing import List

from PIL import Image, ImageDraw, ImageFont

# ARC color index -> single-character code
COLOR_CODES = {
    0: ".",  # black
    1: "B",  # blue
    2: "R",  # red
    3: "G",  # green
    4: "Y",  # yellow
    5: "X",  # grey
    6: "M",  # magenta
    7: "O",  # orange
    8: "A",  # azure
    9: "W",  # maroon
}

CODE_TO_INDEX = {v: k for k, v in COLOR_CODES.items()}

COLOR_KEY = "Color key: .=black B=blue R=red G=green Y=yellow X=grey M=magenta O=orange A=azure W=maroon"

# ARC color index -> RGB for PNG rendering
COLOR_RGB = {
    0: (0, 0, 0),         # black
    1: (0, 116, 217),     # blue
    2: (255, 65, 54),     # red
    3: (46, 204, 64),     # green
    4: (255, 220, 0),     # yellow
    5: (170, 170, 170),   # grey
    6: (240, 18, 190),    # magenta
    7: (255, 133, 27),    # orange
    8: (127, 219, 255),   # azure
    9: (128, 0, 0),       # maroon
}

CELL_SIZE = 20
BORDER_WIDTH = 1
BORDER_COLOR = (128, 128, 128)


def grid_to_text(grid: List[List[int]]) -> str:
    """Convert a 2D integer grid to a text representation with single-char color codes."""
    return "\n".join(" ".join(COLOR_CODES[cell] for cell in row) for row in grid)


def text_to_grid(text: str) -> List[List[int]]:
    """Convert text representation back to a 2D integer grid."""
    grid = []
    for line in text.strip().split("\n"):
        row = [CODE_TO_INDEX[c] for c in line.split()]
        grid.append(row)
    return grid


def grid_to_png_bytes(grid: List[List[int]]) -> bytes:
    """Render a 2D integer grid as a PNG image (20x20px cells, 1px grey border)."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    width = cols * CELL_SIZE + (cols + 1) * BORDER_WIDTH
    height = rows * CELL_SIZE + (rows + 1) * BORDER_WIDTH

    img = Image.new("RGB", (width, height), BORDER_COLOR)
    draw = ImageDraw.Draw(img)

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            x0 = BORDER_WIDTH + c * (CELL_SIZE + BORDER_WIDTH)
            y0 = BORDER_WIDTH + r * (CELL_SIZE + BORDER_WIDTH)
            x1 = x0 + CELL_SIZE - 1
            y1 = y0 + CELL_SIZE - 1
            draw.rectangle([x0, y0, x1, y1], fill=COLOR_RGB[cell])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def grid_to_base64_png(grid: List[List[int]]) -> str:
    """Render grid as base64-encoded PNG string."""
    return base64.b64encode(grid_to_png_bytes(grid)).decode("ascii")


def format_example_text(example: dict, idx: int) -> str:
    """Format a single training example as text."""
    lines = [f"Example {idx}:"]
    lines.append("Input:")
    lines.append(grid_to_text(example["input"]))
    lines.append("Output:")
    lines.append(grid_to_text(example["output"]))
    return "\n".join(lines)


def format_examples_text(examples: list, num_examples: int = None) -> str:
    """Format multiple training examples as text."""
    if num_examples is not None:
        examples = examples[:num_examples]
    return "\n\n".join(format_example_text(ex, i + 1) for i, ex in enumerate(examples))


def format_test_input_text(test_input: List[List[int]]) -> str:
    """Format test input grid as text."""
    return f"Test Input:\n{grid_to_text(test_input)}"


def compare_grids(predicted: List[List[int]], expected: List[List[int]]):
    """Compare two grids. Returns (correct: bool, cell_accuracy: float)."""
    if predicted == expected:
        return True, 1.0

    total = 0
    matching = 0
    max_rows = max(len(predicted), len(expected))
    for r in range(max_rows):
        pred_row = predicted[r] if r < len(predicted) else []
        exp_row = expected[r] if r < len(expected) else []
        max_cols = max(len(pred_row), len(exp_row))
        for c in range(max_cols):
            total += 1
            pred_val = pred_row[c] if c < len(pred_row) else -1
            exp_val = exp_row[c] if c < len(exp_row) else -2
            if pred_val == exp_val:
                matching += 1

    return False, matching / total if total > 0 else 0.0


def parse_response_grid(response_text: str):
    """Parse model response to extract predicted grid and reasoning.

    Tries multiple strategies:
    1. Parse as JSON with "reasoning" and "output_grid" fields
    2. Find JSON embedded in the response text
    3. Extract grid from an ANSWER: block in freeform text

    Returns (grid: list|None, reasoning: str|None, error: str|None).
    """
    if not response_text:
        return None, None, "Empty response"

    text = response_text.strip()

    # Strategy 1: try JSON parsing (possibly inside markdown code blocks)
    json_text = text
    if "```json" in json_text:
        try:
            start = json_text.index("```json") + 7
            end = json_text.index("```", start)
            json_text = json_text[start:end].strip()
        except ValueError:
            pass
    elif "```" in json_text:
        try:
            start = json_text.index("```") + 3
            end = json_text.index("```", start)
            json_text = json_text[start:end].strip()
        except ValueError:
            pass

    data = _try_parse_json(json_text)
    if data is None:
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            data = _try_parse_json(text[brace_start:brace_end + 1])

    if data is not None:
        reasoning = data.get("reasoning", "")
        output_grid = data.get("output_grid")
        if output_grid is not None:
            grid, err = _convert_grid(output_grid)
            if grid is not None:
                return grid, reasoning, None
            return None, reasoning, err
        return None, reasoning, "No output_grid in response JSON"

    # Strategy 2: extract grid from ANSWER: block
    grid, err = _extract_answer_block(text)
    if grid is not None:
        return grid, text, None

    return None, text, "Could not parse grid from response"


def _try_parse_json(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _convert_grid(output_grid):
    if not output_grid or not output_grid[0]:
        return None, "Empty output_grid"
    if isinstance(output_grid[0][0], str):
        try:
            return [[CODE_TO_INDEX[c] for c in row] for row in output_grid], None
        except KeyError as e:
            return None, f"Unknown color code in grid: {e}"
    return output_grid, None


def _extract_answer_block(text):
    """Try to extract a grid from an ANSWER: block in freeform text."""
    # Look for ANSWER: followed by grid lines
    answer_match = re.search(r'ANSWER\s*:\s*\n((?:[.\w ]+\n?)+)', text, re.IGNORECASE)
    if not answer_match:
        answer_match = re.search(
            r'(?:Output grid|Output|Final grid|Final answer)\s*:\s*\n((?:[.\w ]+\n?)+)',
            text, re.IGNORECASE
        )

    if answer_match:
        grid_text = answer_match.group(1).strip()
        try:
            grid = text_to_grid(grid_text)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                return grid, None
        except (KeyError, ValueError, IndexError):
            pass

    # Try numbered rows: 0: ". B R G"
    numbered_rows = re.findall(
        r'^\s*\d+\s*:\s*"([.BRGXYMAOW ]+)"',
        text, re.MULTILINE
    )
    if len(numbered_rows) >= 2:
        grid_text = "\n".join(numbered_rows)
        try:
            grid = text_to_grid(grid_text)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                return grid, None
        except (KeyError, ValueError, IndexError):
            pass

    # Try last block of consecutive grid-like lines
    grid_line_pattern = re.compile(r'^[.BRGXYMAOW](?:\s+[.BRGXYMAOW])+$')
    lines = text.split('\n')
    last_block_start = None
    last_block_end = None
    i = len(lines) - 1
    while i >= 0:
        if grid_line_pattern.match(lines[i].strip()):
            end = i
            while i >= 0 and grid_line_pattern.match(lines[i].strip()):
                i -= 1
            start = i + 1
            if end - start >= 1:
                last_block_start = start
                last_block_end = end
                break
        i -= 1

    if last_block_start is not None:
        grid_text = "\n".join(lines[last_block_start:last_block_end + 1])
        try:
            grid = text_to_grid(grid_text)
            if grid and len(grid) > 0 and len(grid[0]) > 0:
                return grid, None
        except (KeyError, ValueError, IndexError):
            pass

    return None, "No grid found in response"
