import os

from google import genai
from google.genai import types

from rubric.autograders.schemas import (
    OneShotOutput,
    PerCriterionOutput,
    RubricAsJudgeOutput,
)


async def default_per_criterion_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs
) -> PerCriterionOutput:
    """Default generate function for PerCriterionGrader using Gemini API.

    Calls Gemini with JSON schema for structured output and validates the response.
    Users should implement their own generate functions with proper retry logic
    and error handling tailored to their LLM client.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-3-pro-preview",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=PerCriterionOutput,
        ),
    )
    return response.parsed


async def default_oneshot_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs
) -> OneShotOutput:
    """Default generate function for PerCriterionOneShotGrader using Gemini API.

    Calls Gemini with JSON schema for structured output and validates the response.
    Users should implement their own generate functions with proper retry logic
    and error handling tailored to their LLM client.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-3-pro-preview",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=OneShotOutput,
        ),
    )
    return response.parsed


async def default_rubric_as_judge_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs
) -> RubricAsJudgeOutput:
    """Default generate function for RubricAsJudgeGrader using Gemini API.

    Calls Gemini with JSON schema for structured output and validates the response.
    Users should implement their own generate functions with proper retry logic
    and error handling tailored to their LLM client.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = await client.aio.models.generate_content(
        model="gemini-3-pro-preview",
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0,
            response_mime_type="application/json",
            response_schema=RubricAsJudgeOutput,
        ),
    )
    return response.parsed
