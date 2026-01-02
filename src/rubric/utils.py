import json
import os
import re

from google import genai
from google.genai import types

from rubric.types import LengthPenalty


def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in text.

    This is the default counting function used by LengthPenalty.
    For more accurate token counting with a specific model, provide a custom
    count_fn that uses a tokenizer.
    """
    return len(text.split())


def compute_length_penalty(text: str, config: LengthPenalty) -> float:
    """Compute the length penalty for the given text based on the config.

    The penalty follows an exponential curve:
    - Returns 0 if word/token count is at or below free_budget
    - Returns penalty_at_cap if count is at or above max_cap
    - Returns an interpolated value between those bounds using the exponent

    Args:
        text: The text to measure.
        config: LengthPenalty configuration specifying thresholds and penalty.

    Returns:
        A penalty value between 0 and penalty_at_cap to subtract from the score.
    """
    count_fn = config.count_fn if config.count_fn is not None else word_count
    count = count_fn(text)

    if count <= config.free_budget:
        return 0.0
    if count >= config.max_cap:
        return config.penalty_at_cap

    frac = (count - config.free_budget) / float(config.max_cap - config.free_budget)
    return config.penalty_at_cap * (frac**config.exponent)


def parse_json_to_dict(json_string: str) -> dict:
    """Parse JSON string with various formats (including markdown fences)."""
    cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip(), flags=re.IGNORECASE)

    cleaned = re.sub(r"^\s*json\s*", "", cleaned, flags=re.IGNORECASE)

    if cleaned and cleaned[0] != "{":
        brace = cleaned.find("{")
        if brace != -1:
            cleaned = cleaned[brace:]

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
