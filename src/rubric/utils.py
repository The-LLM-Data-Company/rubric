import json
import os
import re

from google import genai
from google.genai import types


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
        ),
    )
    return response.text or ""
