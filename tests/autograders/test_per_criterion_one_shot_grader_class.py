import pytest

from rubric import Criterion, CriterionEvaluation, OneShotOutput, Rubric
from rubric.autograders import PerCriterionOneShotGrader


@pytest.mark.asyncio
async def test_per_criterion_one_shot_grader_class_integration(
    sample_rubric, sample_output, one_shot_generate_fn
):
    grader = PerCriterionOneShotGrader(generate_fn=one_shot_generate_fn)

    report = await sample_rubric.grade(sample_output, autograder=grader)

    assert report.score == pytest.approx(1.0)
    assert report.report is not None
    assert len(report.report) == len(sample_rubric.rubric)
    assert [criterion.verdict for criterion in report.report] == [
        "MET",
        "MET",
        "MET",
        "UNMET",
    ]


@pytest.mark.asyncio
async def test_per_criterion_one_shot_grader_with_negative_criterion_unmet(sample_rubric):
    async def generate_with_issue(system_prompt: str, user_prompt: str) -> OneShotOutput:
        return OneShotOutput(
            criteria_evaluations=[
                CriterionEvaluation(criterion_number=1, criterion_status="MET", explanation="Test"),
                CriterionEvaluation(criterion_number=2, criterion_status="MET", explanation="Test"),
                CriterionEvaluation(criterion_number=3, criterion_status="MET", explanation="Test"),
                CriterionEvaluation(
                    criterion_number=4,
                    criterion_status="UNMET",
                    explanation="Error not present",
                ),
            ]
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_with_issue)

    report = await sample_rubric.grade("Test", autograder=grader)

    assert report.score == pytest.approx(1.0)
    assert report.report is not None
    verdicts = [criterion.verdict for criterion in report.report]
    assert verdicts == ["MET", "MET", "MET", "UNMET"]


@pytest.mark.asyncio
async def test_all_negative_criteria_all_unmet_returns_perfect_score():
    """All-negative rubric with no errors present should return 1.0."""
    rubric = Rubric(
        [
            Criterion(weight=-1.0, requirement="Contains factual errors"),
            Criterion(weight=-1.0, requirement="Contains profanity"),
            Criterion(weight=-1.0, requirement="Contains harmful content"),
        ]
    )

    async def generate_no_errors(system_prompt: str, user_prompt: str) -> OneShotOutput:
        return OneShotOutput(
            criteria_evaluations=[
                CriterionEvaluation(
                    criterion_number=1,
                    criterion_status="UNMET",
                    explanation="No errors",
                ),
                CriterionEvaluation(
                    criterion_number=2,
                    criterion_status="UNMET",
                    explanation="No profanity",
                ),
                CriterionEvaluation(
                    criterion_number=3,
                    criterion_status="UNMET",
                    explanation="No harmful content",
                ),
            ]
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_no_errors)
    result = await rubric.grade("Clean, accurate text", autograder=grader)

    assert result.score == pytest.approx(1.0)
    assert result.raw_score == pytest.approx(0.0)
    assert all(r.verdict == "UNMET" for r in result.report)


@pytest.mark.asyncio
async def test_all_negative_criteria_all_met_returns_zero_score():
    """All-negative rubric with all errors present should return 0.0."""
    rubric = Rubric(
        [
            Criterion(weight=-1.0, requirement="Contains factual errors"),
            Criterion(weight=-1.0, requirement="Contains profanity"),
        ]
    )

    async def generate_all_errors(system_prompt: str, user_prompt: str) -> OneShotOutput:
        return OneShotOutput(
            criteria_evaluations=[
                CriterionEvaluation(
                    criterion_number=1, criterion_status="MET", explanation="Has errors"
                ),
                CriterionEvaluation(
                    criterion_number=2,
                    criterion_status="MET",
                    explanation="Has profanity",
                ),
            ]
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_all_errors)
    result = await rubric.grade("Bad text", autograder=grader)

    assert result.score == pytest.approx(0.0)
    assert result.raw_score == pytest.approx(-2.0)
    assert all(r.verdict == "MET" for r in result.report)
