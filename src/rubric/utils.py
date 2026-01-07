import json
import os
import re
import warnings

from google import genai
from google.genai import types

from rubric.types import LengthPenalty, ThinkingOutputDict, ToGradeInput


def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in text.

    This is the default counting function used by LengthPenalty.
    For more accurate token counting with a specific model, provide a custom
    count_fn that uses a tokenizer.
    """
    return len(text.split())


def parse_thinking_output(text: str) -> ThinkingOutputDict:
    """Parse thinking and output sections from text with XML-style markers.

    Looks for <thinking>...</thinking> and <output>...</output> markers.
    If markers are not found, treats the entire text as output.

    Args:
        text: Text potentially containing thinking/output markers.

    Returns:
        Dict with 'thinking' and 'output' keys. Empty strings if sections not found.

    Examples:
        >>> parse_thinking_output("<thinking>ABC</thinking><output>DEF</output>")
        {'thinking': 'ABC', 'output': 'DEF'}

        >>> parse_thinking_output("Just output text")
        {'thinking': '', 'output': 'Just output text'}

        >>> parse_thinking_output("<thinking>Think</thinking>Rest")
        {'thinking': 'Think', 'output': 'Rest'}
    """
    # Try to extract thinking section
    thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL | re.IGNORECASE)
    thinking = thinking_match.group(1).strip() if thinking_match else ""

    # Try to extract output section
    output_match = re.search(r"<output>(.*?)</output>", text, re.DOTALL | re.IGNORECASE)

    if output_match:
        # Explicit output markers found
        output = output_match.group(1).strip()
    elif thinking_match:
        # Has thinking but no output markers - treat rest as output
        # Remove the thinking section and use remainder
        output = re.sub(
            r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE
        ).strip()
    else:
        # No markers at all - treat entire text as output
        output = text

    return ThinkingOutputDict(thinking=thinking, output=output)


def normalize_to_grade_input(to_grade: ToGradeInput) -> ThinkingOutputDict:
    """Normalize to_grade input to dict format.

    Args:
        to_grade: Either a string (with optional markers) or a dict.

    Returns:
        Dict with 'thinking' and 'output' keys.

    Raises:
        ValueError: If dict format is invalid (missing keys, wrong types).
    """
    if isinstance(to_grade, str):
        return parse_thinking_output(to_grade)

    # Handle dict input
    if not isinstance(to_grade, dict):
        raise ValueError(f"to_grade must be a string or dict, got {type(to_grade).__name__}")

    # Validate dict has correct keys
    thinking = to_grade.get("thinking", "")
    output = to_grade.get("output", "")

    # Validate types
    if not isinstance(thinking, str):
        raise ValueError(f"'thinking' must be a string, got {type(thinking).__name__}")
    if not isinstance(output, str):
        raise ValueError(f"'output' must be a string, got {type(output).__name__}")

    # Warn if dict has unexpected keys
    expected_keys = {"thinking", "output"}
    extra_keys = set(to_grade.keys()) - expected_keys
    if extra_keys:
        warnings.warn(
            f"Unexpected keys in to_grade dict: {extra_keys}. "
            f"Only 'thinking' and 'output' are used.",
            UserWarning,
        )

    return ThinkingOutputDict(thinking=thinking, output=output)


def compute_length_penalty(text: str | ThinkingOutputDict, config: LengthPenalty) -> float:
    """Compute the length penalty for the given text based on the config.

    The penalty follows an exponential curve:
    - Returns 0 if word/token count is at or below free_budget
    - Returns penalty_at_cap if count is at or above max_cap
    - Returns an interpolated value between those bounds using the exponent

    Args:
        text: Either a string (backwards compatible) or a dict with 'thinking'
            and 'output' keys. When a string is provided, it's treated as
            all output (no thinking section).
        config: LengthPenalty configuration specifying thresholds, penalty,
            and which sections to count based on penalty_type.

    Returns:
        A penalty value between 0 and penalty_at_cap to subtract from the score.
    """
    # Normalize input to dict format
    if isinstance(text, str):
        # Backwards compatibility: treat string as output only
        text_dict = ThinkingOutputDict(thinking="", output=text)
    else:
        text_dict = text

    # Select which text to count based on penalty_type
    if config.penalty_type == "ALL":
        # Concatenate both sections (with space to avoid word merging)
        text_to_count = text_dict.get("thinking", "") + " " + text_dict.get("output", "")
    elif config.penalty_type == "OUTPUT_ONLY":
        text_to_count = text_dict.get("output", "")
    elif config.penalty_type == "THINKING_ONLY":
        text_to_count = text_dict.get("thinking", "")
    else:
        raise ValueError(
            f"Invalid penalty_type: {config.penalty_type}. "
            f"Must be 'ALL', 'OUTPUT_ONLY', or 'THINKING_ONLY'."
        )

    # Count tokens/words
    count_fn = config.count_fn if config.count_fn is not None else word_count
    count = count_fn(text_to_count)

    # Apply penalty curve
    if count <= config.free_budget:
        return 0.0
    if count >= config.max_cap:
        return config.penalty_at_cap

    frac = (count - config.free_budget) / float(config.max_cap - config.free_budget)
    return config.penalty_at_cap * (frac**config.exponent)


def _find_matching_brace(text: str, start: int) -> int:
    """Find the index of the closing brace matching the opening brace at start.

    Properly handles nested braces and braces inside JSON strings.

    Args:
        text: The text to search in.
        start: Index of the opening brace '{'.

    Returns:
        Index of the matching closing brace, or -1 if not found.
    """
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        if escape:
            escape = False
            continue

        if char == "\\":
            escape = True
            continue

        if char == '"' and not escape:
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return i

    return -1


def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON string with various formats (including markdown fences).

    Handles common LLM output patterns:
    - Markdown code fences: ```json {...} ```
    - Leading text before JSON: "Here is the result: {...}"
    - Trailing text after JSON: "{...} I hope this helps!"
    """
    cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip(), flags=re.IGNORECASE)

    cleaned = re.sub(r"^\s*json\s*", "", cleaned, flags=re.IGNORECASE)

    # Find opening brace
    start = cleaned.find("{")
    if start == -1:
        # No JSON object found, let json.loads raise appropriate error
        return json.loads(cleaned)

    # Find matching closing brace
    end = _find_matching_brace(cleaned, start)
    if end != -1:
        cleaned = cleaned[start : end + 1]
    else:
        # No matching brace found, try from start and let json.loads handle it
        cleaned = cleaned[start:]

    return json.loads(cleaned)


async def default_generate_fn(system_prompt: str, user_prompt: str) -> str:
    """Generate a response from the Gemini API."""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-3-pro-preview",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
        ),
    )
    return response.text or ""
