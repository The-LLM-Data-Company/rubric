"""Passes the entire rubric to the LLM for holistic scoring."""

from __future__ import annotations

import json

from rubric.autograders import Autograder
from rubric.types import Criterion, EvaluationReport, GenerateFn, LengthPenalty
from rubric.utils import default_generate_fn, parse_json_to_dict

DEFAULT_SYSTEM_PROMPT = """You are evaluating an output for a given query against a list of \
criteria.

You will receive the output to evaluate, and a numbered list of criteria to check. Each \
criterion is marked as POSITIVE or NEGATIVE and has an associated weight.

Your job is to MENTALLY EVALUATE each criterion using the logic below, compute a weighted \
score in your head, and return a single holistic score from 0-100.

CRITERION TYPES:
Each criterion is marked as positive or negative. Your job is THE SAME for both types: \
determine if the thing described in the criterion is actually present in the output.

POSITIVE CRITERIA:
Positive criteria describe desired traits, requirements, or content that should be present.
- MET (criterion present): The output contains/satisfies the requirement
- UNMET (criterion absent): The output does not contain/satisfy the requirement

NEGATIVE CRITERIA:
Negative criteria describe ACTIVE ERRORS or MISTAKES that the output is making.
- MET (error present): The output ADVOCATES, STATES, or RECOMMENDS the problematic thing
- UNMET (error absent): The output does NOT make this error, OR it mentions the thing only \
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
- MET: The thing described in the criterion IS present/occurring in the output
- UNMET: The thing described in the criterion IS NOT present/occurring in the output

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
vs negative criteria, and compute a final weighted score that reflects how well the output \
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
    ):
        super().__init__(generate_fn=generate_fn, length_penalty=length_penalty, normalize=normalize)
        self.system_prompt = system_prompt

    async def judge(
        self, to_grade: str, rubric: list[Criterion], query: str | None = None
    ) -> float:
        criteria_lines = []
        for index, criterion in enumerate(rubric, start=1):
            criterion_type = (
                "NEGATIVE (error present if MET, error absent if UNMET)"
                if criterion.weight < 0
                else "POSITIVE (requirement present if MET, requirement absent if UNMET)"
            )
            criteria_lines.append(
                f"{index}. [{criterion_type}] (weight: {criterion.weight}) {criterion.requirement}"
            )

        criteria_text = "\n".join(criteria_lines)
        query_text = f"<input>{query}</input>" if query else ""
        user_prompt = f"""Mentally evaluate each criterion below, compute the weighted score \
using the logic from the system prompt, and return a single holistic score from 0-100.

<criteria>
{criteria_text}
</criteria>

{query_text}

<output>
{to_grade}
</output>

Return your evaluation as JSON only."""

        try:
            response = await self.generate(self.system_prompt, user_prompt)
            result = parse_json_to_dict(response)
            overall_score_raw = result.get("overall_score", 0)
            overall_score = float(overall_score_raw)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return 0.0

        return overall_score

    async def aggregate(
        self, judge_results: float, *, normalize: bool = True
    ) -> EvaluationReport:
        raw_score = judge_results

        if normalize:
            score = max(0.0, min(1.0, judge_results / 100.0))
        else:
            score = raw_score

        return EvaluationReport(score=score, raw_score=raw_score, report=None)
