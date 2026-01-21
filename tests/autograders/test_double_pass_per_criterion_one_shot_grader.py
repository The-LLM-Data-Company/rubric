import pytest

from rubric import Criterion, CriterionEvaluation, OneShotOutput, Rubric
from rubric.autograders import DoublePassPerCriterionOneShotGrader


@pytest.mark.asyncio
async def test_double_pass_grader_basic_integration(sample_rubric, sample_output):
    """Basic integration test - should work like regular one-shot when both passes agree."""

    # Custom generate function that returns consistent results regardless of criteria order
    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        # Parse the number of criteria from the prompt
        criteria_count = user_prompt.count("[POSITIVE") + user_prompt.count("[NEGATIVE")

        evaluations = []
        for i in range(1, criteria_count + 1):
            # Check if this position has a NEGATIVE criterion
            # In both passes, negative criteria get UNMET (error not present = good)
            if f"{i}. [NEGATIVE" in user_prompt:
                evaluations.append(
                    CriterionEvaluation(
                        criterion_number=i,
                        criterion_status="UNMET",
                        explanation="Error not present",
                    )
                )
            else:
                evaluations.append(
                    CriterionEvaluation(
                        criterion_number=i,
                        criterion_status="MET",
                        explanation="Requirement satisfied",
                    )
                )

        return OneShotOutput(criteria_evaluations=evaluations)

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)

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
async def test_positive_criterion_requires_both_passes_to_agree():
    """Positive criteria require BOTH passes to say MET (conservative)."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="First criterion"),
            Criterion(weight=1.0, requirement="Second criterion"),
            Criterion(weight=1.0, requirement="Third criterion"),
        ]
    )

    call_count = [0]

    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        call_count[0] += 1
        if call_count[0] == 1:
            # Pass 1: First MET, Second MET, Third UNMET
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1, criterion_status="MET", explanation="Pass 1: met"
                    ),
                    CriterionEvaluation(
                        criterion_number=2, criterion_status="MET", explanation="Pass 1: met"
                    ),
                    CriterionEvaluation(
                        criterion_number=3, criterion_status="UNMET", explanation="Pass 1: unmet"
                    ),
                ]
            )
        else:
            # Pass 2 (reversed order: Third, Second, First)
            # Position 1 = Third (UNMET), Position 2 = Second (MET), Position 3 = First (UNMET)
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1, criterion_status="UNMET", explanation="Pass 2: unmet"
                    ),
                    CriterionEvaluation(
                        criterion_number=2, criterion_status="MET", explanation="Pass 2: met"
                    ),
                    CriterionEvaluation(
                        criterion_number=3, criterion_status="UNMET", explanation="Pass 2: unmet"
                    ),
                ]
            )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)
    result = await rubric.grade("test", autograder=grader)

    # First: MET in pass 1, UNMET in pass 2 → UNMET (need both)
    # Second: MET in pass 1, MET in pass 2 → MET (both agree)
    # Third: UNMET in pass 1, UNMET in pass 2 → UNMET
    assert result.report[0].verdict == "UNMET"
    assert result.report[1].verdict == "MET"
    assert result.report[2].verdict == "UNMET"

    # Only 1 out of 3 criteria MET
    assert result.score == pytest.approx(1.0 / 3.0)


@pytest.mark.asyncio
async def test_positive_criterion_disagreement_uses_unmet_explanation():
    """When positive criterion has disagreement, use UNMET explanation."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Test criterion"),
        ]
    )

    call_count = [0]

    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        call_count[0] += 1
        if call_count[0] == 1:
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1,
                        criterion_status="MET",
                        explanation="Pass 1 says MET",
                    ),
                ]
            )
        else:
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1,
                        criterion_status="UNMET",
                        explanation="Pass 2 says UNMET",
                    ),
                ]
            )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)
    result = await rubric.grade("test", autograder=grader)

    # Conservative: disagreement means UNMET, use UNMET explanation
    assert result.report[0].verdict == "UNMET"
    assert result.report[0].reason == "Pass 2 says UNMET"


@pytest.mark.asyncio
async def test_negative_criterion_met_if_either_pass_detects():
    """Negative criteria are MET if EITHER pass detects the error (strict on errors)."""
    rubric = Rubric(
        [
            Criterion(weight=-1.0, requirement="Contains errors"),
            Criterion(weight=-1.0, requirement="Contains profanity"),
        ]
    )

    call_count = [0]

    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        call_count[0] += 1
        if call_count[0] == 1:
            # Pass 1: First error detected, second not detected
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1, criterion_status="MET", explanation="Error found"
                    ),
                    CriterionEvaluation(
                        criterion_number=2, criterion_status="UNMET", explanation="No profanity"
                    ),
                ]
            )
        else:
            # Pass 2 (reversed): Position 1 = profanity (detected),
            # Position 2 = errors (not detected)
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1, criterion_status="MET", explanation="Profanity found"
                    ),
                    CriterionEvaluation(
                        criterion_number=2, criterion_status="UNMET", explanation="No errors"
                    ),
                ]
            )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)
    result = await rubric.grade("test", autograder=grader)

    # First (errors): MET in pass 1, UNMET in pass 2 → MET (either detects)
    # Second (profanity): UNMET in pass 1, MET in pass 2 → MET (either detects)
    assert result.report[0].verdict == "MET"
    assert result.report[0].reason == "Error found"  # From pass 1 which detected it
    assert result.report[1].verdict == "MET"
    assert result.report[1].reason == "Profanity found"  # From pass 2 which detected it

    # Both errors detected, score = 0
    assert result.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_mixed_rubric_conservative_logic():
    """Test conservative logic with mixed positive and negative criteria."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Good requirement"),
            Criterion(weight=-1.0, requirement="Contains errors"),
        ]
    )

    call_count = [0]

    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        call_count[0] += 1
        if call_count[0] == 1:
            # Pass 1: positive MET, negative UNMET
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1, criterion_status="MET", explanation="Good"
                    ),
                    CriterionEvaluation(
                        criterion_number=2, criterion_status="UNMET", explanation="No errors"
                    ),
                ]
            )
        else:
            # Pass 2 (reversed): [negative, positive]
            # Position 1 = negative (UNMET), Position 2 = positive (MET)
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1, criterion_status="UNMET", explanation="No errors"
                    ),
                    CriterionEvaluation(
                        criterion_number=2, criterion_status="MET", explanation="Good"
                    ),
                ]
            )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)
    result = await rubric.grade("test", autograder=grader)

    # Positive: MET in both passes → MET
    # Negative: UNMET in both passes → UNMET (no error detected)
    assert result.report[0].verdict == "MET"
    assert result.report[1].verdict == "UNMET"
    assert result.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_all_negative_rubric_no_errors():
    """All-negative rubric with no errors should return 1.0."""
    rubric = Rubric(
        [
            Criterion(weight=-1.0, requirement="Contains factual errors"),
            Criterion(weight=-1.0, requirement="Contains profanity"),
        ]
    )

    async def generate_no_errors(system_prompt: str, user_prompt: str) -> OneShotOutput:
        return OneShotOutput(
            criteria_evaluations=[
                CriterionEvaluation(
                    criterion_number=1, criterion_status="UNMET", explanation="No errors"
                ),
                CriterionEvaluation(
                    criterion_number=2, criterion_status="UNMET", explanation="No profanity"
                ),
            ]
        )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_no_errors)
    result = await rubric.grade("Clean text", autograder=grader)

    assert result.score == pytest.approx(1.0)
    assert result.raw_score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_normalize_false():
    """Test that normalize=False returns raw weighted sums."""
    rubric = Rubric(
        [
            Criterion(weight=2.0, requirement="First"),
            Criterion(weight=1.0, requirement="Second"),
        ]
    )

    # Both passes return same results - both criteria MET
    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        return OneShotOutput(
            criteria_evaluations=[
                CriterionEvaluation(criterion_number=1, criterion_status="MET", explanation="Test"),
                CriterionEvaluation(criterion_number=2, criterion_status="MET", explanation="Test"),
            ]
        )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn, normalize=False)
    result = await rubric.grade("test", autograder=grader)

    # Both criteria MET in both passes → both MET
    assert result.score == pytest.approx(3.0)  # 2.0 + 1.0
    assert result.raw_score == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_both_passes_unmet_uses_pass1_explanation():
    """When positive criterion is UNMET in both passes, use pass 1 explanation."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Test criterion"),
        ]
    )

    call_count = [0]

    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        call_count[0] += 1
        if call_count[0] == 1:
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1,
                        criterion_status="UNMET",
                        explanation="Pass 1 explanation",
                    ),
                ]
            )
        else:
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1,
                        criterion_status="UNMET",
                        explanation="Pass 2 explanation",
                    ),
                ]
            )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)
    result = await rubric.grade("test", autograder=grader)

    assert result.report[0].verdict == "UNMET"
    assert result.report[0].reason == "Pass 1 explanation"


@pytest.mark.asyncio
async def test_both_passes_met_uses_pass1_explanation():
    """When positive criterion is MET in both passes, use pass 1 explanation."""
    rubric = Rubric(
        [
            Criterion(weight=1.0, requirement="Test criterion"),
        ]
    )

    call_count = [0]

    async def generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
        call_count[0] += 1
        if call_count[0] == 1:
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1,
                        criterion_status="MET",
                        explanation="Pass 1 explanation",
                    ),
                ]
            )
        else:
            return OneShotOutput(
                criteria_evaluations=[
                    CriterionEvaluation(
                        criterion_number=1,
                        criterion_status="MET",
                        explanation="Pass 2 explanation",
                    ),
                ]
            )

    grader = DoublePassPerCriterionOneShotGrader(generate_fn=generate_fn)
    result = await rubric.grade("test", autograder=grader)

    assert result.report[0].verdict == "MET"
    assert result.report[0].reason == "Pass 1 explanation"
