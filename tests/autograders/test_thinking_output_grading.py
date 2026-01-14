"""Integration tests for thinking/output token support in grading."""

import pytest

from rubric import Criterion, LengthPenalty, PerCriterionOutput, Rubric
from rubric.autograders import PerCriterionGrader


@pytest.fixture
def simple_criteria():
    """Simple criteria for testing."""
    return [
        Criterion(weight=10.0, requirement="Output is correct"),
        Criterion(weight=5.0, requirement="Output is well-explained"),
    ]


@pytest.fixture
def mock_generate_fn():
    """Generate function that marks all criteria as MET."""

    async def _generate(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
        return PerCriterionOutput(criterion_status="MET", explanation="Requirement satisfied.")

    return _generate


class TestInputFormats:
    """Test different input formats."""

    @pytest.mark.asyncio
    async def test_dict_input(self, simple_criteria, mock_generate_fn):
        """Test grading with dict input."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(generate_fn=mock_generate_fn)

        result = await rubric.grade(
            {"thinking": "reasoning...", "output": "answer"}, autograder=grader
        )

        assert result.score == 1.0
        assert result.raw_score == 15.0

    @pytest.mark.asyncio
    async def test_string_with_markers(self, simple_criteria, mock_generate_fn):
        """Test grading with string containing markers."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(generate_fn=mock_generate_fn)

        result = await rubric.grade(
            "<thinking>reasoning</thinking><output>answer</output>", autograder=grader
        )

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_plain_string_backwards_compatible(self, simple_criteria, mock_generate_fn):
        """Test plain string still works (backwards compatible)."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(generate_fn=mock_generate_fn)

        result = await rubric.grade("plain response", autograder=grader)

        assert result.score == 1.0


class TestLengthPenaltyWithPenaltyType:
    """Test length penalty with different penalty types."""

    @pytest.mark.asyncio
    async def test_output_only_penalty(self, simple_criteria, mock_generate_fn):
        """Test OUTPUT_ONLY penalty ignores thinking length."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(
            generate_fn=mock_generate_fn,
            length_penalty=LengthPenalty(
                free_budget=5, max_cap=10, penalty_at_cap=0.5, penalty_type="OUTPUT_ONLY"
            ),
        )

        # Long thinking, short output
        result = await rubric.grade(
            {"thinking": " ".join(["word"] * 100), "output": "short"}, autograder=grader
        )

        assert result.score == 1.0  # No penalty despite long thinking

    @pytest.mark.asyncio
    async def test_thinking_only_penalty(self, simple_criteria, mock_generate_fn):
        """Test THINKING_ONLY penalty ignores output length."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(
            generate_fn=mock_generate_fn,
            length_penalty=LengthPenalty(
                free_budget=5, max_cap=10, penalty_at_cap=0.5, penalty_type="THINKING_ONLY"
            ),
        )

        # Short thinking, long output
        result = await rubric.grade(
            {"thinking": "brief", "output": " ".join(["word"] * 100)}, autograder=grader
        )

        assert result.score == 1.0  # No penalty despite long output

    @pytest.mark.asyncio
    async def test_all_penalty_type_default(self, simple_criteria, mock_generate_fn):
        """Test that ALL penalty type counts both sections."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(
            generate_fn=mock_generate_fn,
            length_penalty=LengthPenalty(
                free_budget=5,
                max_cap=10,
                penalty_at_cap=0.5,  # penalty_type="ALL" by default
            ),
        )

        # Long response
        result = await rubric.grade(" ".join(["word"] * 20), autograder=grader)

        assert result.score < 1.0  # Should be penalized

    @pytest.mark.asyncio
    async def test_raw_scores_with_penalty(self, simple_criteria, mock_generate_fn):
        """Test raw scores (normalize=False) with length penalty."""
        rubric = Rubric(simple_criteria)
        grader = PerCriterionGrader(
            generate_fn=mock_generate_fn,
            normalize=False,
            length_penalty=LengthPenalty(
                free_budget=5, max_cap=10, penalty_at_cap=50.0, penalty_type="OUTPUT_ONLY"
            ),
        )

        result = await rubric.grade(
            {"thinking": "brief", "output": " ".join(["word"] * 20)}, autograder=grader
        )

        assert result.score < 0  # Heavily penalized in raw score mode
        assert result.raw_score == 15.0  # Base score before penalty
