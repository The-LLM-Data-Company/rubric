import pytest

from rubric import Criterion, LengthPenalty, PerCriterionOutput, Rubric
from rubric.autograders import PerCriterionGrader
from rubric.utils import compute_length_penalty, word_count


@pytest.fixture
def simple_rubric() -> Rubric:
    return Rubric(
        [
            Criterion(weight=10.0, requirement="Contains greeting"),
            Criterion(weight=5.0, requirement="Contains farewell"),
            Criterion(weight=-3.0, requirement="Contains profanity"),
        ]
    )


@pytest.fixture
def all_met_generate_fn():
    async def _generate(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
        if "negative" in user_prompt.lower():
            return PerCriterionOutput(criterion_status="UNMET", explanation="No profanity found")
        return PerCriterionOutput(criterion_status="MET", explanation="Requirement satisfied")

    return _generate


class TestLengthPenaltyComputation:
    def test_no_penalty_under_free_budget(self):
        config = LengthPenalty(free_budget=100, max_cap=200, penalty_at_cap=50.0)
        text = " ".join(["word"] * 50)
        assert compute_length_penalty(text, config) == 0.0

    def test_max_penalty_at_cap(self):
        config = LengthPenalty(free_budget=100, max_cap=200, penalty_at_cap=50.0)
        text = " ".join(["word"] * 250)
        assert compute_length_penalty(text, config) == 50.0

    def test_partial_penalty_between_budget_and_cap(self):
        config = LengthPenalty(free_budget=100, max_cap=200, penalty_at_cap=50.0, exponent=1.0)
        text = " ".join(["word"] * 150)
        penalty = compute_length_penalty(text, config)
        assert 0.0 < penalty < 50.0
        assert penalty == pytest.approx(25.0)

    def test_custom_count_fn(self):
        config = LengthPenalty(
            free_budget=10,
            max_cap=20,
            penalty_at_cap=10.0,
            count_fn=lambda text: len(text),
        )
        text = "a" * 25
        assert compute_length_penalty(text, config) == 10.0


class TestWordCount:
    def test_word_count_basic(self):
        assert word_count("hello world") == 2

    def test_word_count_empty(self):
        assert word_count("") == 0

    def test_word_count_multiple_spaces(self):
        assert word_count("hello   world") == 2


@pytest.mark.asyncio
class TestLengthPenaltyWithNormalize:
    async def test_normalized_score_with_length_penalty(self, simple_rubric, all_met_generate_fn):
        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=True,
            length_penalty=LengthPenalty(
                free_budget=10,
                max_cap=20,
                penalty_at_cap=0.5,
            ),
        )

        short_text = "Hello goodbye"
        result = await simple_rubric.grade(short_text, autograder=grader)

        assert result.raw_score == 15.0
        assert result.score == pytest.approx(1.0)

    async def test_normalized_score_with_length_penalty_applied(
        self, simple_rubric, all_met_generate_fn
    ):
        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=True,
            length_penalty=LengthPenalty(
                free_budget=5,
                max_cap=10,
                penalty_at_cap=0.5,
            ),
        )

        long_text = " ".join(["word"] * 15)
        result = await simple_rubric.grade(long_text, autograder=grader)

        assert result.raw_score == 15.0
        assert result.score == pytest.approx(0.5)


@pytest.mark.asyncio
class TestLengthPenaltyWithoutNormalize:
    async def test_raw_score_no_length_penalty(self, simple_rubric, all_met_generate_fn):
        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=False,
        )

        result = await simple_rubric.grade("Hello goodbye", autograder=grader)

        assert result.raw_score == 15.0
        assert result.score == 15.0

    async def test_raw_score_with_length_penalty_under_budget(
        self, simple_rubric, all_met_generate_fn
    ):
        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=False,
            length_penalty=LengthPenalty(
                free_budget=100,
                max_cap=200,
                penalty_at_cap=50.0,
            ),
        )

        short_text = "Hello goodbye"
        result = await simple_rubric.grade(short_text, autograder=grader)

        assert result.raw_score == 15.0
        assert result.score == 15.0

    async def test_raw_score_with_length_penalty_at_cap(self, simple_rubric, all_met_generate_fn):
        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=False,
            length_penalty=LengthPenalty(
                free_budget=5,
                max_cap=10,
                penalty_at_cap=50.0,
            ),
        )

        long_text = " ".join(["word"] * 15)
        result = await simple_rubric.grade(long_text, autograder=grader)

        assert result.raw_score == 15.0
        assert result.score == pytest.approx(-35.0)

    async def test_raw_score_with_partial_length_penalty(self, simple_rubric, all_met_generate_fn):
        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=False,
            length_penalty=LengthPenalty(
                free_budget=5,
                max_cap=15,
                penalty_at_cap=10.0,
                exponent=1.0,
            ),
        )

        text = " ".join(["word"] * 10)
        result = await simple_rubric.grade(text, autograder=grader)

        assert result.raw_score == 15.0
        expected_penalty = 5.0
        assert result.score == pytest.approx(15.0 - expected_penalty)

    async def test_raw_score_can_go_negative(self, all_met_generate_fn):
        rubric = Rubric(
            [
                Criterion(weight=5.0, requirement="Contains greeting"),
            ]
        )

        grader = PerCriterionGrader(
            generate_fn=all_met_generate_fn,
            normalize=False,
            length_penalty=LengthPenalty(
                free_budget=5,
                max_cap=10,
                penalty_at_cap=100.0,
            ),
        )

        long_text = " ".join(["word"] * 20)
        result = await rubric.grade(long_text, autograder=grader)

        assert result.raw_score == 5.0
        assert result.score == pytest.approx(-95.0)


@pytest.mark.asyncio
class TestLengthPenaltyWithNegativeCriteria:
    async def test_negative_criteria_met_reduces_raw_score(self):
        rubric = Rubric(
            [
                Criterion(weight=10.0, requirement="Contains greeting"),
                Criterion(weight=-5.0, requirement="Contains spam"),
            ]
        )

        async def generate_with_spam(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
            if "negative" in user_prompt.lower():
                return PerCriterionOutput(criterion_status="MET", explanation="Spam detected")
            return PerCriterionOutput(criterion_status="MET", explanation="Greeting found")

        grader = PerCriterionGrader(
            generate_fn=generate_with_spam,
            normalize=False,
        )

        result = await rubric.grade("Hello spam", autograder=grader)

        assert result.raw_score == 5.0
        assert result.score == 5.0

    async def test_negative_criteria_with_length_penalty(self):
        rubric = Rubric(
            [
                Criterion(weight=10.0, requirement="Contains greeting"),
                Criterion(weight=-5.0, requirement="Contains spam"),
            ]
        )

        async def generate_with_spam(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
            if "negative" in user_prompt.lower():
                return PerCriterionOutput(criterion_status="MET", explanation="Spam detected")
            return PerCriterionOutput(criterion_status="MET", explanation="Greeting found")

        grader = PerCriterionGrader(
            generate_fn=generate_with_spam,
            normalize=False,
            length_penalty=LengthPenalty(
                free_budget=5,
                max_cap=10,
                penalty_at_cap=10.0,
            ),
        )

        long_text = " ".join(["word"] * 15)
        result = await rubric.grade(long_text, autograder=grader)

        assert result.raw_score == 5.0
        assert result.score == pytest.approx(-5.0)
