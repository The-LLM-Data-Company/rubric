"""Type definitions for rubrics and evaluation components."""

from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

CountFn = Callable[[str], int]


class LengthPenalty(BaseModel):
    """Configuration for applying length-based penalties during grading.

    The penalty is computed as:
    - 0 if count <= free_budget
    - penalty_at_cap if count >= max_cap
    - penalty_at_cap * ((count - free_budget) / (max_cap - free_budget)) ** exponent otherwise

    By default, the penalty is subtracted from the final score (which is normalized to 0-1).
    For training use cases with raw scores, use absolute penalty values (e.g., -50.0).

    Args:
        free_budget: Number of tokens/words allowed before any penalty applies.
        max_cap: Number of tokens/words at which the maximum penalty is applied.
        penalty_at_cap: Maximum penalty value. For normalized scores, use fractional values
            like 0.5 (lose up to 50% of score). For training with raw scores, use absolute
            values like -50.0 (will be added directly to the raw score).
        exponent: Controls the penalty curve steepness. Higher = more lenient near free_budget.
        count_fn: Function to count tokens/words in text. If None, uses whitespace word count.
            For accurate token counting, pass a tokenizer-based function like:
            `lambda text: len(tokenizer.encode(text))`

    Example:
        >>> # Default: word-based counting with sensible defaults for normalized scores
        >>> penalty = LengthPenalty()
        >>>
        >>> # For training with raw (unnormalized) scores - absolute penalty values
        >>> penalty = LengthPenalty(
        ...     free_budget=8000,
        ...     max_cap=10000,
        ...     penalty_at_cap=-50.0,  # Absolute penalty added to raw score
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
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    free_budget: int = 6000
    max_cap: int = 8000
    penalty_at_cap: float = 0.5
    exponent: float = 1.6
    count_fn: CountFn | None = None


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
    instead of normalized 0-1 scores. The raw_score field always contains the unnormalized
    weighted sum regardless of the normalize setting.
    """

    score: float
    raw_score: float | None = None
    report: list[CriterionReport] | None = None


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
