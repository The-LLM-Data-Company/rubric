"""Pydantic schemas for LLM output validation.

These models define the expected output structure for each autograder type.
Users can access `.model_json_schema()` on these models to enable constrained
decoding in their LLM clients.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PerCriterionOutput(BaseModel):
    """Expected output for PerCriterionGrader.

    Evaluates a single criterion and returns MET/UNMET verdict with explanation.

    Example:
        >>> output = PerCriterionOutput(
        ...     criterion_status="MET",
        ...     explanation="The response correctly identifies the diagnosis."
        ... )
    """

    criterion_status: Literal["MET", "UNMET"] = Field(
        description="Whether the criterion is present (MET) or absent (UNMET) in the response."
    )
    explanation: str = Field(description="Brief explanation of the verdict.")


class CriterionEvaluation(BaseModel):
    """A single criterion evaluation within a one-shot response.

    Used by OneShotOutput to represent each criterion's verdict.
    """

    criterion_number: int = Field(description="The 1-based index of the criterion being evaluated.")
    criterion_status: Literal["MET", "UNMET"] = Field(
        description="Whether the criterion is present (MET) or absent (UNMET) in the response."
    )
    explanation: str = Field(description="Brief explanation of the verdict.")


class OneShotOutput(BaseModel):
    """Expected output for PerCriterionOneShotGrader.

    Evaluates all criteria in a single LLM call.

    Example:
        >>> output = OneShotOutput(criteria_evaluations=[
        ...     CriterionEvaluation(criterion_number=1, criterion_status="MET", explanation="..."),
        ...     CriterionEvaluation(criterion_number=2, criterion_status="UNMET", explanation="...")
        ... ])
    """

    criteria_evaluations: list[CriterionEvaluation] = Field(
        description="List of evaluations for each criterion.", min_length=1
    )


class RubricAsJudgeOutput(BaseModel):
    """Expected output for RubricAsJudgeGrader.

    Returns a single holistic score from 0-100.

    Example:
        >>> output = RubricAsJudgeOutput(overall_score=85.0, explanation="...")
    """

    overall_score: float = Field(
        description="Holistic score from 0-100 representing overall rubric satisfaction.",
    )
    explanation: str = Field(description="Brief explanation of the score.")
