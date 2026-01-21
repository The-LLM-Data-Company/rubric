"""Autograder that evaluates criteria twice with reversed order to reduce position bias."""

import asyncio

from rubric.autograders import Autograder
from rubric.autograders.per_criterion_one_shot_grader import DEFAULT_SYSTEM_PROMPT
from rubric.types import (
    Criterion,
    CriterionReport,
    EvaluationReport,
    OneShotGenerateFn,
)


class DoublePassPerCriterionOneShotGrader(Autograder):
    """Autograder that runs evaluation twice with reversed criteria order.

    This grader reduces position bias by evaluating criteria in both original
    and reversed order. It uses a conservative reconciliation strategy:

    - Positive criteria: MET only if BOTH passes agree (strict on requirements)
    - Negative criteria: MET if EITHER pass detects it (strict on error detection)

    This produces rigorous evaluations where high scores indicate strong confidence.

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

    def _build_user_prompt(
        self, to_grade: str, criteria: list[Criterion], query: str | None = None
    ) -> str:
        """Build user prompt for a single pass."""
        criteria_lines = []
        for index, criterion in enumerate(criteria, start=1):
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
        return f"""Evaluate the response against the following criteria:
<criteria>
{criteria_text}
</criteria>

{query_text}

<response>
{to_grade}
</response>

Provide your evaluation as JSON only."""

    async def judge(
        self, to_grade: str, rubric: list[Criterion], query: str | None = None
    ) -> list[CriterionReport]:
        # Build prompts for both passes
        reversed_rubric = list(reversed(rubric))

        user_prompt_pass1 = self._build_user_prompt(to_grade, rubric, query)
        user_prompt_pass2 = self._build_user_prompt(to_grade, reversed_rubric, query)

        # Run both passes in parallel
        result_pass1, result_pass2 = await asyncio.gather(
            self.generate_fn(self.system_prompt, user_prompt_pass1),
            self.generate_fn(self.system_prompt, user_prompt_pass2),
        )

        # Create mappings from criterion_number to evaluation for both passes
        pass1_map = {
            eval_item.criterion_number: eval_item for eval_item in result_pass1.criteria_evaluations
        }

        # For pass 2, map reversed indices back to original indices
        # Pass 2 criterion 1 corresponds to original criterion n, etc.
        n = len(rubric)
        pass2_map = {
            (n - eval_item.criterion_number + 1): eval_item
            for eval_item in result_pass2.criteria_evaluations
        }

        # Build criterion reports with merged results using conservative strategy
        criterion_reports: list[CriterionReport] = []
        for index, criterion in enumerate(rubric, start=1):
            eval_pass1 = pass1_map.get(index)
            eval_pass2 = pass2_map.get(index)

            pass1_met = eval_pass1 and eval_pass1.criterion_status == "MET"
            pass2_met = eval_pass2 and eval_pass2.criterion_status == "MET"

            # Negative criteria: MET if EITHER pass detects error (strict on error detection)
            # Positive criteria: MET only if BOTH passes agree (strict on requirements)
            is_met = (pass1_met or pass2_met) if criterion.weight < 0 else (pass1_met and pass2_met)

            if is_met:
                source = eval_pass1 if pass1_met else eval_pass2
            else:
                source = next(
                    (e for e in (eval_pass1, eval_pass2) if e and e.criterion_status == "UNMET"),
                    eval_pass1 or eval_pass2,
                )

            criterion_reports.append(
                CriterionReport(
                    requirement=criterion.requirement,
                    verdict="MET" if is_met else "UNMET",
                    reason=source.explanation if source else "Evaluation not found in response",
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
