"""Shared abstractions for autograder implementations."""

from abc import ABC, abstractmethod
from typing import Any

from rubric.types import Criterion, EvaluationReport, GenerateFn, LengthPenalty
from rubric.utils import compute_length_penalty


class Autograder(ABC):
    """Base class describing the LLM-backed grading workflow.

    Subclasses inherit a ready-to-use `generate()` helper that delegates to the caller-supplied
    `generate_fn`. This keeps the LLM client choice outside of the core grading logic while making
    the dependency visible in constructors.

    Args:
        generate_fn: Async function for LLM generation with (system_prompt, user_prompt) signature.
        length_penalty: Optional configuration for penalizing overly long outputs.
            When provided, a penalty based on the token/word count is subtracted from the final score.
    """

    def __init__(
        self,
        generate_fn: GenerateFn | None = None,
        length_penalty: LengthPenalty | None = None,
    ):
        self.generate_fn: GenerateFn | None = generate_fn
        self.length_penalty: LengthPenalty | None = length_penalty

    async def generate(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        """Invoke the injected LLM callable with explicit system/user prompts."""
        if self.generate_fn is None:
            raise ValueError("generate_fn must be provided or override the generate method")
        return await self.generate_fn(system_prompt, user_prompt, **kwargs)

    @abstractmethod
    async def judge(self, to_grade: str, rubric: list[Criterion], query: str | None = None) -> Any:
        """Collect raw judge results for the provided submission."""
        pass

    @abstractmethod
    async def aggregate(self, judge_results: Any) -> EvaluationReport:
        """Transform judge results into an EvaluationReport."""
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
            EvaluationReport with score (0-1) and optional per-criterion breakdown.
            If length_penalty was configured, the penalty is subtracted from the score.

        You can override this method to implement custom grading logic outside the judge and
        aggregate steps.
        """

        judge_results = await self.judge(to_grade, rubric, query)
        report = await self.aggregate(judge_results)

        if self.length_penalty is not None:
            penalty = compute_length_penalty(to_grade, self.length_penalty)
            adjusted_score = max(0.0, report.score - penalty)
            return EvaluationReport(score=adjusted_score, report=report.report)

        return report
