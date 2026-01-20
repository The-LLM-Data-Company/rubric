"""Shared abstractions for autograder implementations."""

from abc import ABC, abstractmethod
from typing import Any

from rubric.types import Criterion, EvaluationReport


class Autograder(ABC):
    """Base class describing the LLM-backed grading workflow.

    Each concrete autograder accepts a typed generate_fn that returns validated Pydantic
    models specific to that grader's output format. Users implement their generate functions
    with parsing, validation, and retry logic tailored to their LLM client.

    Args:
        normalize: If True (default), scores are normalized to 0-1. If False, raw weighted
            sums are returned, which is useful for RL training scenarios.
    """

    def __init__(
        self,
        *,
        normalize: bool = True,
    ):
        self.normalize: bool = normalize

    @abstractmethod
    async def judge(self, to_grade: str, rubric: list[Criterion], query: str | None = None) -> Any:
        """Collect raw judge results for the provided submission."""
        pass

    @abstractmethod
    async def aggregate(self, judge_results: Any, *, normalize: bool = True) -> EvaluationReport:
        """Transform judge results into an EvaluationReport.

        Args:
            judge_results: Raw results from judge().
            normalize: If True, normalize score to 0-1. If False, return raw weighted sum.

        Returns:
            EvaluationReport with score and optional per-criterion breakdown.
        """
        pass

    async def grade(
        self,
        to_grade: str,
        rubric: list[Criterion],
        query: str | None = None,
    ) -> EvaluationReport:
        """Grade the submission against the rubric. This is the main entry point for the autograder.

        Args:
            to_grade: The text to evaluate.
            rubric: List of criteria to evaluate against.
            query: Optional input/query that prompted the response.

        Returns:
            EvaluationReport with score and optional per-criterion breakdown.
            If normalize=True (default), score is 0-1. If normalize=False, score is raw
            weighted sum. The raw_score field contains the unnormalized weighted sum.

        You can override this method to implement custom grading logic outside the judge and
        aggregate steps.
        """
        judge_results = await self.judge(to_grade, rubric, query)
        return await self.aggregate(judge_results, normalize=self.normalize)
