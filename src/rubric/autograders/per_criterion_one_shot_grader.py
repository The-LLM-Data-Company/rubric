"""Autograder that evaluates all criteria in a single LLM call."""

from __future__ import annotations

from rubric.autograders import Autograder
from rubric.autograders.schemas import OneShotOutput
from rubric.types import (
    Criterion,
    CriterionReport,
    EvaluationReport,
    OneShotGenerateFn,
)

DEFAULT_SYSTEM_PROMPT = """You are evaluating a response for a given query against a list of \
criteria.

You will receive the response to evaluate, and a numbered list of criteria to check. Each \
criterion is marked as POSITIVE or NEGATIVE.

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
- UNMET (criterion_status: "UNMET"): The response does NOT make this error, OR it mentions \
the thing only to warn against it or mention why it's wrong

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
"explanation": "The response explicitly states the patient does NOT have diabetes, so this \
error is not present."
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


class PerCriterionOneShotGrader(Autograder):
    """Concrete autograder that judges every criterion within a single LLM response.

    Args:
        generate_fn: Typed generate function that returns validated OneShotOutput.
            Users handle parsing, validation, and retries in their implementation.
        system_prompt: System prompt for one-shot evaluation.
        normalize: If True (default), normalize scores to 0-1.
    """

    def __init__(
        self,
        generate_fn: OneShotGenerateFn,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.generate_fn = generate_fn
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

        # Call generate_fn - user handles validation and retries
        result: OneShotOutput = await self.generate_fn(self.system_prompt, user_prompt)

        # Create a mapping from criterion_number to evaluation
        evaluation_map = {
            eval_item.criterion_number: eval_item for eval_item in result.criteria_evaluations
        }

        # Build criterion reports matching rubric order
        criterion_reports: list[CriterionReport] = []
        for index, criterion in enumerate(rubric, start=1):
            eval_item = evaluation_map.get(index)

            if eval_item:
                criterion_reports.append(
                    CriterionReport(
                        requirement=criterion.requirement,
                        verdict=eval_item.criterion_status,
                        reason=eval_item.explanation,
                        weight=criterion.weight,
                    )
                )
            else:
                # Evaluation missing for this criterion number
                criterion_reports.append(
                    CriterionReport(
                        requirement=criterion.requirement,
                        verdict="UNMET",
                        reason="Evaluation not found in response",
                        weight=criterion.weight,
                    )
                )

        return criterion_reports

    async def aggregate(
        self, judge_results: list[CriterionReport], *, normalize: bool = True
    ) -> EvaluationReport:
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
                # Formula: 1.0 + (negative_sum / total_negative)
                # gives 1.0 when no errors, 0.0 when all errors
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
