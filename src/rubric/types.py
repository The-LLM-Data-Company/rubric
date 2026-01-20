"""Type definitions for rubrics and evaluation components."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from rubric.autograders.schemas import (
    OneShotOutput,
    PerCriterionOutput,
    RubricAsJudgeOutput,
)


class Criterion(BaseModel):
    """A single evaluation criterion with a weight and requirement description."""

    model_config = ConfigDict(frozen=True)

    weight: float
    requirement: str


class CriterionReport(Criterion):
    """A criterion with its evaluation verdict (MET/UNMET) and reasoning."""

    verdict: Literal["MET", "UNMET"]
    reason: str


class EvaluationReport(BaseModel):
    """Final evaluation result with score and optional per-criterion reports.

    For training use cases, set normalize=False in the autograder to get raw weighted sums
    instead of normalized 0-1 scores.

    Attributes:
        score: The final score (0-1 if normalized, raw weighted sum otherwise).
        raw_score: Always contains the unnormalized weighted sum, regardless of grader type.
            This provides consistent semantics across all graders for training pipelines.
        llm_raw_score: The original score returned by the LLM before any conversion.
            For PerCriterionGrader/PerCriterionOneShotGrader: same as raw_score (weighted sum).
            For RubricAsJudgeGrader: the 0-100 holistic score from the LLM.
            Useful for debugging and understanding the LLM's actual output.
        report: Optional per-criterion breakdown (None for RubricAsJudgeGrader).
    """

    score: float
    raw_score: float | None = None
    llm_raw_score: float | None = None
    report: list[CriterionReport] | None = None


class PerCriterionGenerateFn(Protocol):
    """Protocol for generate functions used by PerCriterionGrader.

    Must return a validated PerCriterionOutput with criterion_status and explanation.
    Users should handle parsing, validation, and retries within their implementation.
    """

    async def __call__(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> PerCriterionOutput: ...


class OneShotGenerateFn(Protocol):
    """Protocol for generate functions used by PerCriterionOneShotGrader.

    Must return a validated OneShotOutput with criteria_evaluations list.
    Users should handle parsing, validation, and retries within their implementation.
    """

    async def __call__(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> OneShotOutput: ...


class RubricAsJudgeGenerateFn(Protocol):
    """Protocol for generate functions used by RubricAsJudgeGrader.

    Must return a validated RubricAsJudgeOutput with overall_score (0-100).
    Users should handle parsing, validation, and retries within their implementation.
    """

    async def __call__(
        self, system_prompt: str, user_prompt: str, **kwargs: Any
    ) -> RubricAsJudgeOutput: ...
