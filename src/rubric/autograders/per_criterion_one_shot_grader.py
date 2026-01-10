"""Autograder that evaluates all criteria in a single LLM call."""

from __future__ import annotations

import json
import warnings

from rubric.autograders import Autograder
from rubric.types import (
    Criterion,
    CriterionReport,
    DefaultFallbackVerdicts,
    EvaluationReport,
    GenerateFn,
    LengthPenalty,
)
from rubric.utils import default_generate_fn, parse_json_to_dict

DEFAULT_SYSTEM_PROMPT = """You are evaluating a response for a given query against a list of \
criteria.

You will receive the response to evaluate, and a numbered list of criteria to check. Each criterion \
is marked as POSITIVE or NEGATIVE.

CRITERION TYPES:
Each criterion is marked as positive or negative. Your job is THE SAME for both types: determine \
if the thing described in the criterion is actually present in the response.

POSITIVE CRITERIA:
Positive criteria describe desired traits, requirements, or content that should be present.
- MET (criterion_status: "MET"): The response contains/satisfies the requirement
- UNMET (criterion_status: "UNMET"): The response does not contain/satisfy the requirement

NEGATIVE CRITERIA:
Negative criteria describe active errors or mistakes that the response is making.
- MET (criterion_status: "MET"): The response advocates, states, or recommends the problematic thing
- UNMET (criterion_status: "UNMET"): The response does NOT make this error, OR it mentions the thing \
only to warn against it or mention why it's wrong

Examples of what does NOT count as MET for negative criteria:
- "This is often misdiagnosed as X, but it's actually Y" → NOT stating it's X (UNMET)
- "Avoid doing X because..." → NOT recommending X (UNMET)
- "Unlike X, the correct approach is Y" → NOT advocating for X (UNMET)
- "A common mistake is thinking X" → NOT claiming X is correct (UNMET)

EVALUATION RULES:
- For numerical values: Check if they fall within specified ranges or match exactly as required.
- For factual claims: Verify the information is present and accurate, regardless of exact phrasing.
- For required elements: Confirm presence, counting precisely when numbers are specified.
- For exclusion requirements: Confirm that restricted content is absent.
- For length requirements: Carefully measure the number of words, characters, items, etc.
- Be strict about factual accuracy but flexible about wording.
- Accept semantically equivalent statements or implications where appropriate.
- Pay careful attention to negation, warnings, and contrasts.

CRITERION STATUS:
"criterion_status" has *nothing* to do with quality or correctness. It only means:
- "MET": The thing described in the criterion IS present/occurring in the response
- "UNMET": The thing described in the criterion IS NOT present/occurring in the response

Positive criterion: "States Q4 2023 base margin as 17.2%"
Response: "The Q4 2023 base margin was 17.2% before adjustments."
{
"criterion_status": "MET",
"explanation": "The response states Q4 2023 base margin as 17.2%, as required."
}

Negative criterion: "States that the patient has diabetes"
Response: "This patient does not have diabetes."
{
"criterion_status": "UNMET",
"explanation": "The response explicitly states the patient does NOT have diabetes, so this error is \
not present."
}

For each criterion, provide:
- A criterion_status (MET or UNMET)
- An explanation containing a brief justification

Do NOT provide an overall score - only evaluate each criterion.

Respond ONLY with valid JSON in this exact format:
{
  "criteria_evaluations": [
    {
      "criterion_number": 1,
      "criterion_status": "MET",
      "explanation": "Brief explanation"
    },
    ...
  ]
}"""

# Alternative key names that LLMs might use for criterion_number
_ALTERNATIVE_CRITERION_NUMBER_KEYS = ["criterionNumber", "criterion_num", "id", "number", "index"]


def _find_evaluation(evaluations: list, index: int) -> dict | None:
    """Find evaluation entry matching the given criterion index.

    Handles common LLM variations:
    - String numbers ("1" matches 1)
    - Float numbers (1.0 matches 1)
    - Alternative key names (criterionNumber, id, etc.)

    Args:
        evaluations: List of evaluation dicts from LLM response.
        index: The 1-based criterion index to find.

    Returns:
        The matching evaluation dict, or None if not found.
    """
    for entry in evaluations:
        # Try primary key with type coercion
        criterion_num = entry.get("criterion_number")
        if criterion_num is not None:
            try:
                if int(criterion_num) == index:
                    return entry
            except (TypeError, ValueError):
                pass

        # Try alternative keys
        for key in _ALTERNATIVE_CRITERION_NUMBER_KEYS:
            alt_num = entry.get(key)
            if alt_num is not None:
                try:
                    if int(alt_num) == index:
                        warnings.warn(
                            f"LLM used '{key}' instead of 'criterion_number'. "
                            f"Consider updating your prompt or LLM.",
                            UserWarning,
                            stacklevel=4,
                        )
                        return entry
                except (TypeError, ValueError):
                    pass

    return None


class PerCriterionOneShotGrader(Autograder):
    """Concrete autograder that judges every criterion within a single LLM response."""

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

    async def judge(
        self, to_grade: str, rubric: list[Criterion], query: str | None = None
    ) -> list[CriterionReport]:
        criteria_lines = []
        for index, criterion in enumerate(rubric, start=1):
            criterion_type = (
                "NEGATIVE (status MET if error IS present, UNMET if error is NOT present)"
                if criterion.weight < 0
                else "POSITIVE (status MET if requirement IS present, UNMET if requirement "
                "is NOT present)"
            )
            criteria_lines.append(
                f"{index}. [{criterion_type}] (weight: {criterion.weight}) {criterion.requirement}"
            )

        criteria_text = "\n".join(criteria_lines)
        query_text = f"<query>{query}</query>" if query else ""
        user_prompt = f"""Evaluate the response against the following criteria:
<criteria>
{criteria_text}
</criteria>

{query_text}

<response>
{to_grade}
</response>

Provide your evaluation as JSON only."""

        last_error: Exception | None = None
        evaluations: list = []

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.generate(self.system_prompt, user_prompt)
                result = parse_json_to_dict(response)
                evaluations = result.get("criteria_evaluations", [])
                last_error = None
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                last_error = error
                continue

        if last_error is not None:
            error_msg = f"Failed to parse judge response after {self.max_retries + 1} attempts: {last_error}"
            if self.default_fallback_verdicts is not None:
                return [
                    CriterionReport(
                        requirement=criterion.requirement,
                        verdict=self.default_fallback_verdicts.get(
                            "negative" if criterion.weight < 0 else "positive", "UNMET"
                        ),
                        reason=error_msg,
                        weight=criterion.weight,
                    )
                    for criterion in rubric
                ]
            raise ValueError(error_msg)

        criterion_reports: list[CriterionReport] = []
        for index, criterion in enumerate(rubric, start=1):
            eval_data = _find_evaluation(evaluations, index)

            if eval_data:
                criterion_status = str(eval_data.get("criterion_status", "")).strip().upper()
                verdict = "MET" if criterion_status == "MET" else "UNMET"
                explanation = str(eval_data.get("explanation", "No explanation provided"))
            else:
                verdict = "UNMET"
                explanation = "Evaluation not found in response"

            criterion_reports.append(
                CriterionReport(
                    requirement=criterion.requirement,
                    verdict=verdict,
                    reason=explanation,
                    weight=criterion.weight,
                )
            )

        return criterion_reports

    async def aggregate(
        self, judge_results: list[CriterionReport], *, normalize: bool = True
    ) -> EvaluationReport:
        parse_errors = [
            r for r in judge_results if r.reason.startswith("Failed to parse judge response")
        ]
        if parse_errors:
            error_details = "; ".join(
                f"criterion '{r.requirement[:50]}...': {r.reason}"
                if len(r.requirement) > 50
                else f"criterion '{r.requirement}': {r.reason}"
                for r in parse_errors
            )
            return EvaluationReport(
                score=0.0,
                raw_score=0.0,
                llm_raw_score=0.0,
                report=judge_results,
                error=f"Parse errors occurred: {error_details}",
            )

        total_positive_weight = sum(max(0.0, report.weight) for report in judge_results)
        total_negative_weight = sum(
            abs(report.weight) for report in judge_results if report.weight < 0
        )
        weighted_score_sum = sum(
            (1.0 if report.verdict == "MET" else 0.0) * report.weight for report in judge_results
        )

        raw_score = weighted_score_sum

        if normalize:
            if total_positive_weight > 0:
                score = max(0.0, min(1.0, weighted_score_sum / total_positive_weight))
            elif total_negative_weight > 0:
                # All-negative rubric: score starts at 1.0, errors (MET) subtract from it
                # weighted_score_sum is <= 0 for all-negative rubrics
                # Formula: 1.0 + (negative_sum / total_negative) gives 1.0 when no errors, 0.0 when all errors
                score = max(0.0, min(1.0, 1.0 + weighted_score_sum / total_negative_weight))
            else:
                score = 0.0
        else:
            score = raw_score

        return EvaluationReport(
            score=score,
            raw_score=raw_score,
            llm_raw_score=raw_score,  # Same as raw_score for per-criterion graders
            report=judge_results,
        )
