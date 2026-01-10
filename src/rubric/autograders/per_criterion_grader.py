"""Per Criterion grader evaluates each criterion separately in parallel LLM calls."""

import asyncio
import json

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

DEFAULT_SYSTEM_PROMPT = """You are evaluating a response for a given query against a single \
criterion.

You will receive the response to evaluate, a single criterion to check, and a <criterion_type> field \
indicating if the criterion is positive or negative.

CRITERION TYPES:
The <criterion_type> field tells you whether this criterion describes something desirable \
(positive) or undesirable (negative). Your job is THE SAME for both types: determine if the thing \
described in the criterion is actually present in the response.

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

CONDITIONAL VS UNCONDITIONAL ACTIONS (CRITICAL):
When a criterion requires an action to be done "immediately", "now", "as soon as possible", or \
unconditionally, you must distinguish:
- UNCONDITIONAL: "Give epinephrine now" or "Administer X immediately" → action IS being taken
- CONDITIONAL: "If Y occurs, give epinephrine" or "Start X if condition Z" → action is NOT being \
taken immediately; it's contingent on a future condition

If the criterion says something should happen "immediately" or without conditions, a conditional \
statement does NOT satisfy the criterion. Mark as UNMET.

Example:
- Criterion: "Administers alteplase immediately for acute ischemic stroke"
- Output: "If CT confirms no hemorrhage, consider alteplase" → UNMET (conditional, not immediate)
- Output: "Give alteplase now per acute stroke protocol" → MET (immediate, unconditional)

IMPLICIT VS EXPLICIT SATISFACTION:
Consider whether a criterion can be satisfied implicitly through context, tone, or logical \
implication, not just explicit statements:
- "States there is no location in China" can be MET by "Locations are only in United States and \
Canada"—if locations are ONLY in US and Canada, China is excluded; no need to mention China
- "Confirms the user is logged out" can be MET by "Session expired at 3:42 PM"—an expired session \
means the user is logged out, even without stating it directly

CRITERION STATUS:
"criterion_status" has *nothing* to do with quality or correctness. It only means:
- "MET": The thing described in the criterion IS present/occurring in the response
- "UNMET": The thing described in the criterion IS NOT present/occurring in the response

Your response must be valid JSON with this exact format:

{
"criterion_status": "MET",
"explanation": "Brief explanation of why the criterion is or isn't present."
}

Examples:

Positive criterion: "States Q4 2023 base margin as 17.2%"
Response: "The Q4 2023 base margin was 17.2% before adjustments."
{
"criterion_status": "MET",
"explanation": "The response states Q4 2023 base margin as 17.2%, as required."
}

Negative criterion: "States that the patient has celiac disease"
Response: "This patient does not have celiac disease."
{
"criterion_status": "UNMET",
"explanation": "The response explicitly states the patient does NOT have celiac disease, so this error is \
not present."
}

Positive criterion: "Administers epinephrine immediately for anaphylaxis"
Response: "If symptoms worsen, give epinephrine and call for help."
{
"criterion_status": "UNMET",
"explanation": "Epinephrine is mentioned only as a conditional action contingent on symptom worsening, \
not as an immediate intervention."
}

Positive criterion: "States there is no location in China"
Response: "Locations are only in United States and Canada."
{
"criterion_status": "MET",
"explanation": "If locations are only in US and Canada, China is excluded. The response logically \
entails no China location without mentioning China.""
}

Return only raw JSON starting with {, no back-ticks, no 'json' prefix."""


class PerCriterionGrader(Autograder):
    """Concrete autograder that evaluates each criterion independently."""

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

    async def _judge_single_criterion(
        self, criterion: Criterion, to_grade: str, query: str | None = None
    ) -> CriterionReport:
        criterion_type = "negative" if criterion.weight < 0 else "positive"
        query_text = f"<query>{query}</query>" if query else ""
        user_prompt = f"""<criterion_type>
{criterion_type}
</criterion_type>

<criterion>
{criterion.requirement}
</criterion>

{query_text}

<response>
{to_grade}
</response>"""

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self.generate(
                    system_prompt=self.system_prompt, user_prompt=user_prompt
                )

                result = parse_json_to_dict(response)

                explanation = result.get("explanation", "No explanation provided")

                criterion_status = result.get("criterion_status", "UNMET").upper()
                verdict = "MET" if criterion_status == "MET" else "UNMET"

                return CriterionReport(
                    requirement=criterion.requirement,
                    verdict=verdict,
                    reason=explanation,
                    weight=criterion.weight,
                )

            except (json.JSONDecodeError, KeyError) as e:
                last_error = e
                continue

        error_msg = f"Failed to parse judge response after {self.max_retries + 1} attempts: {last_error}"
        if self.default_fallback_verdicts is not None:
            fallback_verdict = self.default_fallback_verdicts.get(criterion_type, "UNMET")
            return CriterionReport(
                requirement=criterion.requirement,
                verdict=fallback_verdict,
                reason=error_msg,
                weight=criterion.weight,
            )
        raise ValueError(error_msg)

    async def judge(
        self, to_grade: str, rubric: list[Criterion], query: str | None = None
    ) -> list[CriterionReport]:
        criterion_tasks = [
            self._judge_single_criterion(criterion, to_grade, query) for criterion in rubric
        ]
        return list(await asyncio.gather(*criterion_tasks))

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
