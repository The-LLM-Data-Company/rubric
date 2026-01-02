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
        normalize: If True (default), scores are normalized to 0-1. If False, raw weighted
            sums are returned, which is useful for RL training scenarios.
    """

    def __init__(
        self,
        generate_fn: GenerateFn | None = None,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
    ):
        self.generate_fn: GenerateFn | None = generate_fn
        self.length_penalty: LengthPenalty | None = length_penalty
        self.normalize: bool = normalize

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
    async def aggregate(
        self, judge_results: Any, *, normalize: bool = True
    ) -> EvaluationReport:
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
            If normalize=True (default), score is 0-1. If normalize=False, score is raw weighted sum.
            If length_penalty was configured, the penalty is subtracted from the score.
            The raw_score field contains the unnormalized weighted sum before length penalty.

        You can override this method to implement custom grading logic outside the judge and
        aggregate steps.
        """

        judge_results = await self.judge(to_grade, rubric, query)
        report = await self.aggregate(judge_results, normalize=self.normalize)

        if self.length_penalty is not None:
            penalty = compute_length_penalty(to_grade, self.length_penalty)
            adjusted_score = report.score + penalty if penalty < 0 else report.score - penalty
            if self.normalize:
                adjusted_score = max(0.0, adjusted_score)
            return EvaluationReport(
                score=adjusted_score, raw_score=report.raw_score, report=report.report
            )

        return report
