"""
Basic usage example with default Gemini generate functions.

Run with: python examples/basic_usage.py
"""

import asyncio
import os

from rubric import Criterion, Rubric, default_per_criterion_generate_fn
from rubric.autograders import (
    PerCriterionGrader,
    PerCriterionOneShotGrader,
    RubricAsJudgeGrader,
)
from rubric.utils import (
    default_oneshot_generate_fn,
    default_rubric_as_judge_generate_fn,
)


async def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not found")
        print("Set it with: export GEMINI_API_KEY='your-api-key'")
        return

    rubric = Rubric(
        rubric=[
            Criterion(weight=2.0, requirement="Response mentions Paris"),
            Criterion(weight=1.0, requirement="Response mentions France"),
            Criterion(weight=1.0, requirement="Response mentions Paris has a rich history"),
            Criterion(weight=-0.5, requirement="Response contains profanity"),
        ]
    )

    response = "Paris is the capital of France. It is a beautiful city with rich history."

    print("\n📝 Response to grade:\n", response)
    print("\n📋 Rubric:")
    print(rubric.rubric)

    graders = [
        ("PerCriterionGrader", PerCriterionGrader(generate_fn=default_per_criterion_generate_fn)),
        (
            "PerCriterionOneShotGrader",
            PerCriterionOneShotGrader(generate_fn=default_oneshot_generate_fn),
        ),
        (
            "RubricAsJudgeGrader",
            RubricAsJudgeGrader(generate_fn=default_rubric_as_judge_generate_fn),
        ),
    ]

    for name, grader in graders:
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)

        result = await rubric.grade(response, autograder=grader)
        print(result)

    print("\n✨ All graders completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
