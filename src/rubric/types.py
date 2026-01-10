"""Type definitions for rubrics and evaluation components."""

from collections.abc import Callable
from typing import Any, Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict

CountFn = Callable[[str], int]


class ThinkingOutputDict(TypedDict, total=False):
    """Dict format for submissions with separate thinking and output sections.

    Both fields are optional to allow partial submissions or gradual construction.
    When used with length penalty, missing fields are treated as empty strings.
    """

    thinking: str
    output: str


ToGradeInput = str | ThinkingOutputDict
"""Union type for to_grade parameter.

Accepts either a plain string or a dict with thinking/output keys.
"""

PenaltyType = Literal["ALL", "OUTPUT_ONLY", "THINKING_ONLY"]
"""Type for penalty_type field: specifies which sections to count for length penalty."""


class DefaultFallbackVerdicts(TypedDict, total=False):
    """Configuration for fallback verdicts when parsing fails.

    If provided to an autograder, parsing failures will use these verdicts instead of raising.
    If None is passed, parsing failures will raise a ValueError.

    Attributes:
        positive: Fallback verdict for positive criteria (weight >= 0). Defaults to "UNMET".
        negative: Fallback verdict for negative criteria (weight < 0). Defaults to "UNMET".

    Example:
        >>> # Conservative fallbacks (worst-case assumptions)
        >>> fallbacks = {"positive": "UNMET", "negative": "MET"}
        >>>
        >>> # All UNMET fallbacks
        >>> fallbacks = {"positive": "UNMET", "negative": "UNMET"}
    """

    positive: Literal["MET", "UNMET"]
    negative: Literal["MET", "UNMET"]


class LengthPenalty(BaseModel):
    """Configuration for applying length-based penalties during grading.

    The penalty is computed as:
    - 0 if count <= free_budget
    - penalty_at_cap if count >= max_cap
    - penalty_at_cap * ((count - free_budget) / (max_cap - free_budget)) ** exponent otherwise

    By default, the penalty is subtracted from the final score (which is normalized to 0-1).
    For training use cases with raw scores, use absolute penalty values (e.g., 50.0).

    Args:
        free_budget: Number of tokens/words allowed before any penalty applies.
        max_cap: Number of tokens/words at which the maximum penalty is applied.
        penalty_at_cap: Maximum penalty value (always subtracted from score). For normalized
            scores, use fractional values like 0.5 (lose up to 50% of score). For training
            with raw scores, use absolute values like 50.0 (subtract up to 50 points).
        exponent: Controls the penalty curve steepness. Higher = more lenient near free_budget.
        count_fn: Function to count tokens/words in text. If None, uses whitespace word count.
            For accurate token counting, pass a tokenizer-based function like:
            `lambda text: len(tokenizer.encode(text))`
        penalty_type: Which text to count for penalty calculation:
            - "ALL": Count both thinking and output tokens (default)
            - "OUTPUT_ONLY": Count only output tokens (useful for RL training)
            - "THINKING_ONLY": Count only thinking tokens

    Example:
        >>> # Default: word-based counting with sensible defaults for normalized scores
        >>> penalty = LengthPenalty()
        >>>
        >>> # For training with raw (unnormalized) scores - absolute penalty values
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     max_cap=10000,
        ...     penalty_at_cap=50.0,  # Subtract up to 50 points from raw score
        ...     exponent=1.6,
        ... )
        >>>
        >>> # Custom tokenizer-based counting (e.g., with HuggingFace)
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     max_cap=10000,
        ...     count_fn=lambda text: len(tokenizer.encode(text))
        ... )
        >>>
        >>> # Only penalize output tokens (allow long thinking)
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     penalty_type="OUTPUT_ONLY",
        ... )
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    free_budget: int = 6000
    max_cap: int = 8000
    penalty_at_cap: float = 0.5
    exponent: float = 1.6
    count_fn: CountFn | None = None
    penalty_type: PenaltyType = "ALL"


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
        error: Optional error message if grading failed (e.g., JSON parse error).
            When set, score defaults to 0.0 and other fields may be None.
            Training pipelines should filter out results where error is not None.
    """

    score: float
    raw_score: float | None = None
    llm_raw_score: float | None = None
    report: list[CriterionReport] | None = None
    error: str | None = None


class GenerateFn(Protocol):
    """Protocol defining the signature for generate functions."""

    async def __call__(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str: ...


class AutograderFn(Protocol):
    """Protocol defining the signature for autograder functions."""

    async def __call__(
        self,
        to_grade: str,
        rubric: list[Criterion],
        generate_fn: GenerateFn,
        **kwargs: Any,
    ) -> EvaluationReport: ...
