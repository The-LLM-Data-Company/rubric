"""
Example generate functions for OpenAI

These functions demonstrate how to implement typed generate functions
that return validated Pydantic models for use with rubric autograders.

Usage:
    from generate_fns import openai_per_criterion_generate_fn
    from rubric.autograders import PerCriterionGrader

    grader = PerCriterionGrader(generate_fn=openai_per_criterion_generate_fn)
"""

import os
from typing import Any

from openai import AsyncOpenAI

from rubric import OneShotOutput, PerCriterionOutput


async def openai_per_criterion_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs: Any
) -> PerCriterionOutput:
    """Generate function for PerCriterionGrader using OpenAI API with structured outputs."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "schema": PerCriterionOutput.model_json_schema(),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    return PerCriterionOutput.model_validate_json(content)


async def openai_oneshot_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs: Any
) -> OneShotOutput:
    """Generate function for PerCriterionOneShotGrader using OpenAI API with structured outputs."""
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "output",
                "schema": OneShotOutput.model_json_schema(),
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content or "{}"
    return OneShotOutput.model_validate_json(content)
