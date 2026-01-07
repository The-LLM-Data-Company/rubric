import json
import warnings

import pytest

from rubric import Criterion, Rubric
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
async def test_per_criterion_one_shot_grader_handles_invalid_json(sample_rubric):
    async def bad_generate(system_prompt: str, user_prompt: str) -> str:
        return "not-json"

    grader = PerCriterionOneShotGrader(generate_fn=bad_generate)

    judge_results = await grader.judge(
        to_grade="Example submission",
        rubric=sample_rubric.rubric,
    )
    report = await grader.aggregate(judge_results)

    assert report.score == 0.0
    assert report.report is not None

    for criterion_report in report.report:
        assert criterion_report.verdict == "UNMET"
        assert "Error parsing judge response" in criterion_report.reason


@pytest.mark.asyncio
async def test_per_criterion_one_shot_grader_with_negative_criterion_unmet(sample_rubric):
    async def generate_with_issue(system_prompt: str, user_prompt: str) -> str:
        import json

        return json.dumps(
            {
                "criteria_evaluations": [
                    {"criterion_number": 1, "criterion_status": "MET", "explanation": "Test"},
                    {"criterion_number": 2, "criterion_status": "MET", "explanation": "Test"},
                    {"criterion_number": 3, "criterion_status": "MET", "explanation": "Test"},
                    {
                        "criterion_number": 4,
                        "criterion_status": "UNMET",
                        "explanation": "Error not present",
                    },
                ]
            }
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

    async def generate_no_errors(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "criteria_evaluations": [
                    {
                        "criterion_number": 1,
                        "criterion_status": "UNMET",
                        "explanation": "No errors",
                    },
                    {
                        "criterion_number": 2,
                        "criterion_status": "UNMET",
                        "explanation": "No profanity",
                    },
                    {
                        "criterion_number": 3,
                        "criterion_status": "UNMET",
                        "explanation": "No harmful content",
                    },
                ]
            }
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

    async def generate_all_errors(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "criteria_evaluations": [
                    {"criterion_number": 1, "criterion_status": "MET", "explanation": "Has errors"},
                    {
                        "criterion_number": 2,
                        "criterion_status": "MET",
                        "explanation": "Has profanity",
                    },
                ]
            }
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_all_errors)
    result = await rubric.grade("Bad text", autograder=grader)

    assert result.score == pytest.approx(0.0)
    assert result.raw_score == pytest.approx(-2.0)
    assert all(r.verdict == "MET" for r in result.report)


@pytest.mark.asyncio
async def test_string_criterion_numbers_are_matched():
    """String criterion numbers should be coerced to int for matching."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Is accurate"),
            Criterion(weight=1.0, requirement="Is helpful"),
        ]
    )

    async def generate_with_strings(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "criteria_evaluations": [
                    {"criterion_number": "1", "criterion_status": "MET", "explanation": "Good"},
                    {"criterion_number": "2", "criterion_status": "MET", "explanation": "Good"},
                ]
            }
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_with_strings)
    result = await rubric.grade("Test", autograder=grader)

    assert result.score == pytest.approx(1.0)
    assert all(r.verdict == "MET" for r in result.report)


@pytest.mark.asyncio
async def test_float_criterion_numbers_are_matched():
    """Float criterion numbers should be coerced to int for matching."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Is accurate"),
            Criterion(weight=1.0, requirement="Is helpful"),
        ]
    )

    async def generate_with_floats(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "criteria_evaluations": [
                    {"criterion_number": 1.0, "criterion_status": "MET", "explanation": "Good"},
                    {"criterion_number": 2.0, "criterion_status": "MET", "explanation": "Good"},
                ]
            }
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_with_floats)
    result = await rubric.grade("Test", autograder=grader)

    assert result.score == pytest.approx(1.0)
    assert all(r.verdict == "MET" for r in result.report)


@pytest.mark.asyncio
async def test_alternative_key_names_matched_with_warning():
    """Alternative key names like 'id' should match with a warning."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Is accurate"),
            Criterion(weight=1.0, requirement="Is helpful"),
        ]
    )

    async def generate_with_alt_keys(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "criteria_evaluations": [
                    {"id": 1, "criterion_status": "MET", "explanation": "Good"},
                    {"id": 2, "criterion_status": "MET", "explanation": "Good"},
                ]
            }
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_with_alt_keys)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = await rubric.grade("Test", autograder=grader)

        # Should have warnings about alternative key usage
        assert len(w) >= 1
        assert any("'id' instead of 'criterion_number'" in str(warning.message) for warning in w)

    assert result.score == pytest.approx(1.0)
    assert all(r.verdict == "MET" for r in result.report)


@pytest.mark.asyncio
async def test_mixed_criterion_number_types():
    """Mixed types (int, string, float) should all be matched correctly."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="First"),
            Criterion(weight=1.0, requirement="Second"),
            Criterion(weight=1.0, requirement="Third"),
        ]
    )

    async def generate_mixed(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "criteria_evaluations": [
                    {"criterion_number": 1, "criterion_status": "MET", "explanation": "Int"},
                    {"criterion_number": "2", "criterion_status": "MET", "explanation": "String"},
                    {"criterion_number": 3.0, "criterion_status": "MET", "explanation": "Float"},
                ]
            }
        )

    grader = PerCriterionOneShotGrader(generate_fn=generate_mixed)
    result = await rubric.grade("Test", autograder=grader)

    assert result.score == pytest.approx(1.0)
    assert all(r.verdict == "MET" for r in result.report)
