"""Passes the entire rubric to the LLM for holistic scoring."""

from __future__ import annotations

import json
import logging

from rubric.autograders import Autograder
from rubric.types import (
    Criterion,
    DefaultFallbackVerdicts,
    EvaluationReport,
    GenerateFn,
    LengthPenalty,
)
from rubric.utils import default_generate_fn, parse_json_to_dict

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are evaluating a response for a given query against a list of \
criteria.

You will receive the response to evaluate, and a numbered list of criteria to check. Each \
criterion is marked as POSITIVE or NEGATIVE and has an associated weight.

Your job is to MENTALLY EVALUATE each criterion using the logic below, compute a weighted \
score in your head, and return a single holistic score from 0-100.

CRITERION TYPES:
Each criterion is marked as positive or negative. Your job is THE SAME for both types: \
determine if the thing described in the criterion is actually present in the response.

POSITIVE CRITERIA:
Positive criteria describe desired traits, requirements, or content that should be present.
- MET (criterion present): The response contains/satisfies the requirement
- UNMET (criterion absent): The response does not contain/satisfy the requirement

NEGATIVE CRITERIA:
Negative criteria describe ACTIVE ERRORS or MISTAKES that the response is making.
- MET (error present): The response ADVOCATES, STATES, or RECOMMENDS the problematic thing
- UNMET (error absent): The response does NOT make this error, OR it mentions the thing only \
to WARN AGAINST it, CONTRAST with it, or explain why it's WRONG

Examples of what does NOT count as MET for negative criteria:
- "This is often misdiagnosed as X, but it's actually Y" → NOT stating it's X (UNMET)
- "Avoid doing X because..." → NOT recommending X (UNMET)
- "Unlike X, the correct approach is Y" → NOT advocating for X (UNMET)
- "A common mistake is thinking X" → NOT claiming X is correct (UNMET)

EVALUATION RULES:
- For numerical values: Check if they fall within specified ranges or match exactly as \
required.
- For factual claims: Verify the information is present and accurate, regardless of exact \
phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- For length requirements: Carefully measure the number of words, characters, items, etc.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or implications where appropriate.
- Pay careful attention to negation, warnings, and contrasts.

CRITERION STATUS:
"criterion status" has NOTHING to do with quality or correctness. It only means:
- MET: The thing described in the criterion IS present/occurring in the response
- UNMET: The thing described in the criterion IS NOT present/occurring in the response

SCORING PROCESS:
1. Mentally evaluate each criterion as MET or UNMET using the logic above
2. For positive criteria: MET criteria earn their weight in points
3. For negative criteria: MET criteria (errors present) SUBTRACT their weight; UNMET \
criteria (errors absent) contribute nothing
4. Sum the total possible positive weight
5. Sum the weighted score (positive points earned minus negative penalties)
6. Compute: (weighted score / total positive weight) × 100
7. Clamp the result to 0-100 range
8. Return this single holistic score

Think through each criterion carefully in context, apply the appropriate logic for positive \
vs negative criteria, and compute a final weighted score that reflects how well the response \
satisfies the rubric as a whole.

Respond ONLY with valid JSON in this exact format:
{
  "overall_score": <number 0-100>
}"""


class RubricAsJudgeGrader(Autograder):
    """Concrete autograder that requests a single holistic score from the model."""

    def __init__(
        self,
        generate_fn: GenerateFn = default_generate_fn,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        length_penalty: LengthPenalty | None = None,
        normalize: bool = True,
        max_retries: int = 2,
        default_fallback_verdicts: DefaultFallbackVerdicts | None = None,
    ):
        super().__init__(
            generate_fn=generate_fn,
            length_penalty=length_penalty,
            normalize=normalize,
            max_retries=max_retries,
            default_fallback_verdicts=default_fallback_verdicts,
        )
        self.system_prompt = system_prompt

    async def judge(self, to_grade: str, rubric: list[Criterion], query: str | None = None) -> dict:
        """Judge the submission and return LLM score with rubric weight info.

        Returns:
            Dict with 'llm_score' (0-100 from LLM), 'total_positive_weight', and
            'total_negative_weight' for computing consistent raw_score semantics.
        """
        criteria_lines = []
        total_positive_weight = 0.0
        total_negative_weight = 0.0

        for index, criterion in enumerate(rubric, start=1):
            if criterion.weight >= 0:
                total_positive_weight += criterion.weight
            else:
                total_negative_weight += abs(criterion.weight)

            criterion_type = (
                "NEGATIVE (error present if MET, error absent if UNMET)"
                if criterion.weight < 0
                else "POSITIVE (requirement present if MET, requirement absent if UNMET)"
            )
            criteria_lines.append(
                f"{index}. [{criterion_type}] (weight: {criterion.weight}) {criterion.requirement}"
            )

        criteria_text = "\n".join(criteria_lines)
        query_text = f"<query>{query}</query>" if query else ""
        user_prompt = f"""Mentally evaluate each criterion below, compute the weighted score \
using the logic from the system prompt, and return a single holistic score from 0-100.

<criteria>
{criteria_text}
</criteria>

{query_text}

<response>
{to_grade}
</response>

Return your evaluation as JSON only."""

        error = None
        last_error: Exception | None = None
        llm_score = 0.0
        response = ""

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.generate(self.system_prompt, user_prompt)
                result = parse_json_to_dict(response)
                overall_score_raw = result.get("overall_score", 0)
                llm_score = float(overall_score_raw)
                last_error = None
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                last_error = e
                continue

        if last_error is not None:
            error = f"Failed to parse judge response after {self.max_retries + 1} attempts: {last_error}"
            response_preview = response[:200] if response else "N/A"
            logger.warning(f"{error}. Response: {response_preview}...")
            if self.default_fallback_verdicts is not None:
                # Silently fall back to llm_score=0 when fallbacks are configured
                error = None

        return {
            "llm_score": llm_score,
            "total_positive_weight": total_positive_weight,
            "total_negative_weight": total_negative_weight,
            "error": error,
        }

    async def aggregate(self, judge_results: dict, *, normalize: bool = True) -> EvaluationReport:
        """Aggregate judge results into an EvaluationReport.

        Computes a synthetic raw_score with weighted-sum semantics for consistency
        with other graders, while preserving the original LLM score in llm_raw_score.
        """
        llm_score = judge_results["llm_score"]
        total_positive_weight = judge_results["total_positive_weight"]
        total_negative_weight = judge_results["total_negative_weight"]
        error = judge_results.get("error")
        llm_raw_score = llm_score

        # Compute synthetic raw_score with weighted-sum semantics
        # This maps 0-100 LLM score to the same scale as PerCriterionGrader
        if total_positive_weight > 0:
            # Normal case: scale LLM's 0-100 to weighted sum range
            raw_score = (llm_score / 100.0) * total_positive_weight
        elif total_negative_weight > 0:
            # All-negative rubric: 100 means no errors (score=1), 0 means all errors (score=0)
            # raw_score for all-negative should be 0 when perfect, negative when errors present
            # LLM score of 100 = no errors = raw_score of 0
            # LLM score of 0 = all errors = raw_score of -total_negative_weight
            raw_score = -total_negative_weight * (1.0 - llm_score / 100.0)
        else:
            raw_score = 0.0

        if normalize:
            score = max(0.0, min(1.0, llm_score / 100.0))
        else:
            score = raw_score

        return EvaluationReport(
            score=score,
            raw_score=raw_score,
            llm_raw_score=llm_raw_score,
            report=None,
            error=error,
        )
