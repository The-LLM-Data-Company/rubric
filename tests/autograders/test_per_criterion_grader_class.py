import json

import pytest

from rubric import Criterion, Rubric
from rubric.autograders import PerCriterionGrader
from rubric.types import EvaluationReport


@pytest.mark.asyncio
async def test_per_criterion_grader_class_integration(
    sample_rubric, sample_output, per_criterion_generate_fn
):
    grader = PerCriterionGrader(generate_fn=per_criterion_generate_fn)

    report: EvaluationReport = await sample_rubric.grade(sample_output, autograder=grader)

    print(report.report)

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
async def test_per_criterion_grader_handles_invalid_json(sample_rubric):
    async def bad_generate(system_prompt: str, user_prompt: str) -> str:
        return "not-json"

    grader = PerCriterionGrader(generate_fn=bad_generate)

    report = await grader.grade(
        to_grade="Example submission",
        rubric=sample_rubric.rubric,
    )

    assert report.score == 0.0
    assert report.report is not None

    for criterion_report in report.report:
        assert criterion_report.verdict == "UNMET"
        assert "Error parsing judge response" in criterion_report.reason


@pytest.mark.asyncio
async def test_per_criterion_grader_with_negative_criterion_unmet(sample_rubric):
    async def generate_with_issue(system_prompt: str, user_prompt: str) -> str:
        import json
        import re

        criterion_type_match = re.search(
            r"<criterion_type>(.*?)</criterion_type>", user_prompt, re.DOTALL
        )
        criterion_type = (
            criterion_type_match.group(1).strip().lower() if criterion_type_match else "positive"
        )

        if criterion_type == "negative":
            # For negative criteria: criterion_status="UNMET" means the error is NOT present (good!)
            return json.dumps({"criterion_status": "UNMET", "explanation": "Error not present"})
        else:
            return json.dumps({"criterion_status": "MET", "explanation": "Requirement met"})

    grader = PerCriterionGrader(generate_fn=generate_with_issue)

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

    async def generate_no_errors(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"criterion_status": "UNMET", "explanation": "Error not present"})

    grader = PerCriterionGrader(generate_fn=generate_no_errors)
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
            Criterion(weight=-1.0, requirement="Contains harmful content"),
        ]
    )

    async def generate_all_errors(system_prompt: str, user_prompt: str) -> str:
        return json.dumps({"criterion_status": "MET", "explanation": "Error is present"})

    grader = PerCriterionGrader(generate_fn=generate_all_errors)
    result = await rubric.grade("Bad text with errors", autograder=grader)

    assert result.score == pytest.approx(0.0)
    assert result.raw_score == pytest.approx(-3.0)
    assert all(r.verdict == "MET" for r in result.report)


@pytest.mark.asyncio
async def test_all_negative_criteria_partial_errors_returns_partial_score():
    """All-negative rubric with some errors should return partial score."""
    rubric = Rubric(
        [
            Criterion(weight=-1.0, requirement="Contains factual errors"),
            Criterion(weight=-1.0, requirement="Contains profanity"),
            Criterion(weight=-1.0, requirement="Contains harmful content"),
        ]
    )

    call_count = 0

    async def generate_one_error(system_prompt: str, user_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # First criterion has error, others don't
        if call_count == 1:
            return json.dumps({"criterion_status": "MET", "explanation": "Error is present"})
        return json.dumps({"criterion_status": "UNMET", "explanation": "Error not present"})

    grader = PerCriterionGrader(generate_fn=generate_one_error)
    result = await rubric.grade("Text with one error", autograder=grader)

    # 1 error out of 3: score = 1.0 + (-1.0 / 3.0) = 2/3 ≈ 0.667
    assert result.score == pytest.approx(2.0 / 3.0)
    assert result.raw_score == pytest.approx(-1.0)


@pytest.mark.asyncio
async def test_all_negative_criteria_with_different_weights():
    """All-negative rubric with varying weights should weight errors appropriately."""
    rubric = Rubric(
        [
            Criterion(weight=-2.0, requirement="Contains major factual errors"),
            Criterion(weight=-1.0, requirement="Contains minor typos"),
        ]
    )

    call_count = 0

    async def generate_minor_error_only(system_prompt: str, user_prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        # Only the minor error (second criterion) is present
        if call_count == 2:
            return json.dumps({"criterion_status": "MET", "explanation": "Minor error present"})
        return json.dumps({"criterion_status": "UNMET", "explanation": "Error not present"})

    grader = PerCriterionGrader(generate_fn=generate_minor_error_only)
    result = await rubric.grade("Text with minor error", autograder=grader)

    # total_negative_weight = 3.0, weighted_score_sum = -1.0
    # score = 1.0 + (-1.0 / 3.0) = 2/3 ≈ 0.667
    assert result.score == pytest.approx(2.0 / 3.0)
    assert result.raw_score == pytest.approx(-1.0)
