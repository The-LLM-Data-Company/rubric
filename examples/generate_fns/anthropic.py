"""
Example generate functions for Anthropic

These functions demonstrate how to implement typed generate functions
that return validated Pydantic models for use with rubric autograders.

Usage:
    from generate_fns import openai_per_criterion_generate_fn
    from rubric.autograders import PerCriterionGrader

    grader = PerCriterionGrader(generate_fn=openai_per_criterion_generate_fn)
"""

import os
from typing import Any

from anthropic import AsyncAnthropic

from rubric import OneShotOutput, PerCriterionOutput


async def anthropic_per_criterion_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs: Any
) -> PerCriterionOutput:
    """Generate function for PerCriterionGrader using Anthropic API with structured outputs."""
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.beta.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        betas=["structured-outputs-2025-11-13"],
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format={
            "type": "json_schema",
            "schema": PerCriterionOutput.model_json_schema(),
        },
    )
    content = response.content[0].text
    return PerCriterionOutput.model_validate_json(content)


async def anthropic_oneshot_generate_fn(
    system_prompt: str, user_prompt: str, **kwargs: Any
) -> OneShotOutput:
    """Generate function for PerCriterionOneShotGrader using Anthropic API with structured outputs."""
    client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await client.beta.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        betas=["structured-outputs-2025-11-13"],
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_format={
            "type": "json_schema",
            "schema": OneShotOutput.model_json_schema(),
        },
    )
    content = response.content[0].text
    return OneShotOutput.model_validate_json(content)
