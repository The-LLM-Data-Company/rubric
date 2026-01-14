import pytest

from rubric import Criterion, Rubric, RubricAsJudgeOutput
from rubric.autograders import RubricAsJudgeGrader


@pytest.mark.asyncio
async def test_rubric_as_judge_grader_class_integration(
    sample_rubric, sample_output, rubric_as_judge_generate_fn
):
    grader = RubricAsJudgeGrader(generate_fn=rubric_as_judge_generate_fn)

    report = await sample_rubric.grade(sample_output, autograder=grader)

    assert report.score == pytest.approx(1.0)
    assert report.report is None


@pytest.mark.asyncio
async def test_rubric_as_judge_grader_raw_score_semantics():
    """Test that raw_score uses weighted-sum semantics consistent with other graders."""
    rubric = Rubric(
        [
            Criterion(weight=10.0, requirement="Is accurate"),
            Criterion(weight=5.0, requirement="Is helpful"),
        ]
    )

    async def generate_85(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=85, explanation="Good quality")

    grader = RubricAsJudgeGrader(generate_fn=generate_85)
    result = await rubric.grade("Test output", autograder=grader)

    # total_positive_weight = 15.0
    # LLM score = 85/100 = 0.85
    # Synthetic raw_score = 0.85 * 15.0 = 12.75
    assert result.raw_score == pytest.approx(12.75)
    assert result.llm_raw_score == pytest.approx(85.0)
    assert result.score == pytest.approx(0.85)


@pytest.mark.asyncio
async def test_rubric_as_judge_grader_llm_raw_score_preserved():
    """Test that llm_raw_score preserves the original 0-100 LLM output."""
    rubric = Rubric(
        [
            Criterion(weight=20.0, requirement="Is complete"),
        ]
    )

    async def generate_70(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=70, explanation="Decent")

    grader = RubricAsJudgeGrader(generate_fn=generate_70)
    result = await rubric.grade("Test output", autograder=grader)

    # Original LLM score should be preserved
    assert result.llm_raw_score == pytest.approx(70.0)
    # Synthetic raw_score = 0.70 * 20.0 = 14.0
    assert result.raw_score == pytest.approx(14.0)


@pytest.mark.asyncio
async def test_rubric_as_judge_grader_normalize_false():
    """Test that normalize=False returns synthetic raw_score, not LLM score."""
    rubric = Rubric(
        [
            Criterion(weight=10.0, requirement="Is accurate"),
            Criterion(weight=5.0, requirement="Is helpful"),
        ]
    )

    async def generate_80(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=80, explanation="Good")

    grader = RubricAsJudgeGrader(generate_fn=generate_80, normalize=False)
    result = await rubric.grade("Test output", autograder=grader)

    # With normalize=False, score should be the synthetic raw_score
    # raw_score = 0.80 * 15.0 = 12.0
    assert result.score == pytest.approx(12.0)
    assert result.raw_score == pytest.approx(12.0)
    assert result.llm_raw_score == pytest.approx(80.0)


@pytest.mark.asyncio
async def test_rubric_as_judge_grader_all_negative_rubric():
    """Test raw_score semantics for all-negative rubrics."""
    rubric = Rubric(
        [
            Criterion(weight=-2.0, requirement="Contains factual errors"),
            Criterion(weight=-3.0, requirement="Contains harmful content"),
        ]
    )

    # LLM score of 100 means no errors detected
    async def generate_100(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=100, explanation="Perfect")

    grader = RubricAsJudgeGrader(generate_fn=generate_100)
    result = await rubric.grade("Perfect output", autograder=grader)

    # For all-negative rubric with score=100 (no errors):
    # raw_score should be 0 (no penalties)
    assert result.raw_score == pytest.approx(0.0)
    assert result.llm_raw_score == pytest.approx(100.0)
    assert result.score == pytest.approx(1.0)

    # LLM score of 0 means all errors detected
    async def generate_0(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=0, explanation="Many errors")

    grader_0 = RubricAsJudgeGrader(generate_fn=generate_0)
    result_0 = await rubric.grade("Bad output", autograder=grader_0)

    # raw_score should be -5.0 (total negative weight)
    assert result_0.raw_score == pytest.approx(-5.0)
    assert result_0.llm_raw_score == pytest.approx(0.0)
    assert result_0.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_rubric_as_judge_grader_mixed_rubric():
    """Test raw_score semantics for mixed positive/negative rubrics."""
    rubric = Rubric(
        [
            Criterion(weight=10.0, requirement="Is accurate"),
            Criterion(weight=-5.0, requirement="Contains errors"),
        ]
    )

    async def generate_90(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
        return RubricAsJudgeOutput(overall_score=90, explanation="Excellent")

    grader = RubricAsJudgeGrader(generate_fn=generate_90)
    result = await rubric.grade("Test output", autograder=grader)

    # For mixed rubric, raw_score is based on positive weight only
    # raw_score = 0.90 * 10.0 = 9.0
    assert result.raw_score == pytest.approx(9.0)
    assert result.llm_raw_score == pytest.approx(90.0)
    assert result.score == pytest.approx(0.90)
