import re
from collections.abc import Awaitable, Callable

import pytest
from dotenv import load_dotenv

from rubric import (
    Criterion,
    OneShotOutput,
    PerCriterionOutput,
    Rubric,
    RubricAsJudgeOutput,
)
from rubric.autograders.schemas import CriterionEvaluation

CriterionList = list[Criterion]
PerCriterionGenerateFn = Callable[[str, str], Awaitable[PerCriterionOutput]]
OneShotGenerateFn = Callable[[str, str], Awaitable[OneShotOutput]]
RubricAsJudgeGenerateFn = Callable[[str, str], Awaitable[RubricAsJudgeOutput]]

load_dotenv()


@pytest.fixture
def sample_output() -> str:
    return "Paris is the capital of France. It is a beautiful city with rich history."


@pytest.fixture
def sample_criteria() -> CriterionList:
    return [
        Criterion(
            weight=2.0,
            requirement="Output mentions Paris",
        ),
        Criterion(
            weight=1.0,
            requirement="Output mentions France",
        ),
        Criterion(
            weight=1.0,
            requirement="Output is written in complete sentences",
        ),
        Criterion(
            weight=-0.5,
            requirement="Output contains profanity or offensive language",
        ),
    ]


@pytest.fixture
def sample_rubric(sample_criteria: CriterionList) -> Rubric:
    return Rubric(sample_criteria)


def _extract_field(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    if not match:
        raise ValueError("Expected field not found in prompt")
    return match.group(1).strip()


@pytest.fixture
def per_criterion_generate_fn() -> PerCriterionGenerateFn:
    criterion_pattern = re.compile(r"<criterion>(.*?)</criterion>", re.DOTALL)
    type_pattern = re.compile(r"<criterion_type>(.*?)</criterion_type>", re.DOTALL)
    positive_requirements_met = {
        "Output mentions Paris",
        "Output mentions France",
        "Output is written in complete sentences",
    }
    negative_errors_present = {
        "Output contains profanity or offensive language": False,  # Error NOT present
    }

    async def _generate(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
        criterion_text = _extract_field(criterion_pattern, user_prompt)
        criterion_type = _extract_field(type_pattern, user_prompt).lower()

        if criterion_type == "negative":
            # For negative criteria: criterion_status="MET" means error IS present (bad)
            # criterion_status="UNMET" means error is NOT present (good)
            error_present = negative_errors_present.get(criterion_text, False)
            explanation = (
                "Error detected in the output."
                if error_present
                else "Error not present in the output."
            )
            return PerCriterionOutput(
                criterion_status="MET" if error_present else "UNMET",
                explanation=explanation,
            )

        criteria_met = criterion_text in positive_requirements_met
        explanation = (
            "Requirement satisfied by the submission."
            if criteria_met
            else "Requirement not satisfied by the submission."
        )
        return PerCriterionOutput(
            criterion_status="MET" if criteria_met else "UNMET",
            explanation=explanation,
        )

    return _generate


@pytest.fixture
def one_shot_generate_fn(sample_criteria: CriterionList) -> OneShotGenerateFn:
    async def _generate(system_prompt: str, user_prompt: str) -> OneShotOutput:
        evaluations = []
        for index, criterion in enumerate(sample_criteria, start=1):
            if criterion.weight < 0:
                # For negative criteria: UNMET means error is NOT present (good)
                criterion_status = "UNMET"
                explanation = "Error not present in the submission."
            else:
                criterion_status = "MET"
                explanation = "Requirement satisfied by the submission."
            evaluations.append(
                CriterionEvaluation(
                    criterion_number=index,
                    criterion_status=criterion_status,
                    explanation=explanation,
                )
            )

        return OneShotOutput(criteria_evaluations=evaluations)

    return _generate


@pytest.fixture
def rubric_as_judge_generate_fn() -> RubricAsJudgeGenerateFn:
    async def _generate(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=135.0, explanation="Perfect score")

    return _generate
