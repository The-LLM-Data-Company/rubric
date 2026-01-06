<div align="center">
  <img alt="The LLM Data Company" src="https://raw.githubusercontent.com/The-LLM-Data-Company/rubric/main/docs/images/Logo.png" width="160" style="max-width: 100%;">
</div>

<h3 align="center">
  Rubric: A Python library for LLM-based evaluation using weighted rubrics.
</h3>

---

<p align="center">
  <a href="https://pypi.org/project/rubric/">
    <img src="https://img.shields.io/pypi/v/rubric" alt="PyPI version" />
  </a>
  <a href="https://pypi.org/project/rubric/">
    <img src="https://img.shields.io/pypi/pyversions/rubric" alt="Python versions" />
  </a>
  <a href="https://github.com/The-LLM-Data-Company/rubric/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  </a>
</p>

## Installation

```bash
uv add rubric
```

## Usage

1. **Set up environment variables:**

```bash
export OPENAI_API_KEY=your_api_key_here
# Or any other model API key used in your `generate_fn`
```

2. **Run the example below**

```python
import asyncio
import os
from openai import AsyncOpenAI
from rubric import Rubric
from rubric.autograders import PerCriterionGrader

# Declare custom generate function with any model and inference provider
async def generate_with_openai(system_prompt: str, user_prompt: str) -> str:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""

async def main():
    # Build rubric
    rubric = Rubric.from_dict([
        {"weight": 10.0, "requirement": "States Q4 2023 base margin as 17.2%"},
        {"weight": 8.0, "requirement": "Explicitly uses Shapley attribution for decomposition"},
        {"weight": -15.0, "requirement": "Uses total deliveries instead of cash-only deliveries"}
    ])

    # Select autograder strategy
    grader = PerCriterionGrader(
        generate_fn=generate_with_openai,
        system_prompt="This overrides the default grader system prompt",
    )

    # Grade output
    result = await rubric.grade(
        query="Input query...",
        to_grade="Output to evaluate...",
        autograder=grader
    )

    print(f"Score: {result.score:.2f}")  # Score is 0.0-1.0
    for criterion in result.report:
        print(f"  [{criterion.verdict}] {criterion.requirement}")
        print(f"    → {criterion.reason}")

asyncio.run(main())
```

## Autograder Strategies

### PerCriterionGrader

Evaluates each criterion in parallel inference calls.

**Scoring Formula:**

For each criterion $i$:

- If verdict = MET, contribution = $w_i$
- If verdict = UNMET, contribution = 0

Final score:

$$
\text{score} = \max\left(0, \min\left(1, \frac{\sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

Where:

- $w_i$ = weight of criterion $i$
- $\mathbb{1}[\text{verdict}_i = \text{MET}]$ = 1 if criterion is MET, 0 otherwise
- Denominator = $\sum_{i=1}^{n} \max(0, w_i)$ (positive weights only)
- Numerator = sum of weights for MET criteria
- Result clamped to [0, 1]

**All-Negative Criteria Rubrics:**

For rubrics containing only negative criteria (e.g., error detection rubrics), a different formula is used:

$$
\text{score} = \max\left(0, \min\left(1, 1 + \frac{\sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i}{\sum_{i=1}^{n} |w_i|}\right)\right)
$$

This ensures:
- Score = 1.0 when all errors are avoided (all criteria UNMET)
- Score = 0.0 when all errors are present (all criteria MET)
- Proportional scores for partial error presence

### PerCriterionOneShotGrader

PerCriterionOneShotGrader makes 1 inference call that evaluates all criteria together and returns a structured output, unlike PerCriterionGrader which makes $n$ inference calls.

**Scoring Formula:**

Same as PerCriterionGrader:

$$
\text{score} = \max\left(0, \min\left(1, \frac{\sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

### RubricAsJudgeGrader

Holistic evaluation where the model returns a final score directly.

**Scoring Formula:**

The model is instructed to mentally evaluate all criteria and return a score from 0-100:

$$
\text{score} = \frac{\text{LLM-judged score}}{100}
$$

Clamped to [0, 1]. The model is guided to use the same weighted scoring logic, but computes the result in-context rather than aggregating score post-hoc.

### Default System Prompts

Each autograder uses a specialized system prompt optimized for its evaluation approach:

**PerCriterionGrader** - Detailed criterion-by-criterion evaluation with strict JSON formatting requirements. The prompt instructs the LLM to evaluate each criterion independently, handling both positive and negative criteria with specific response formats.

**PerCriterionOneShotGrader** - Streamlined prompt for evaluating all criteria in a single response. Focuses on providing verdicts (MET/UNMET) and explanations for each criterion in a structured JSON format.

**RubricAsJudgeGrader** - Holistic evaluation prompt that asks the LLM to consider the output as a whole and provide a single overall score from 0-100, taking into account the weights of all criteria.

You can view the complete default prompts in the source files:

- [`per_criterion_grader.py`](src/rubric/autograders/per_criterion_grader.py#L10-L55)
- [`per_criterion_one_shot_grader.py`](src/rubric/autograders/per_criterion_one_shot_grader.py#L11-L31)
- [`rubric_as_judge_grader.py`](src/rubric/autograders/rubric_as_judge_grader.py#L11-L22)

**Customizing System Prompts:** You can override the default system prompt by passing a `system_prompt` parameter to any autograder:

```python
grader = PerCriterionGrader(
    generate_fn=your_function,
    system_prompt="Your custom system prompt here"
)
```

## Customization

You can customize grading at multiple levels:

**1. Custom `generate_fn` (most common)**
Pass any function that takes `(system_prompt, user_prompt)` and returns a string. Use any LLM provider (OpenAI, Anthropic, local models, etc.):

```python
grader = PerCriterionGrader(generate_fn=your_custom_function)
```

**2. Override specific methods**
Subclass any autograder and override:

- `judge()` - Orchestrates LLM calls to evaluate criteria and parse responses into structured results
- `generate()` - Wraps your `generate_fn` to customize how prompts are sent to the LLM
- `aggregate()` - Transforms individual criterion results into a final score and optional report

**3. Full control**
Override the entire `grade()` method for complete end-to-end control over the grading process.

## Error Handling

### Parse Failure Behavior

When the LLM returns invalid JSON or the response cannot be parsed, the autograders use **conservative defaults** to avoid biasing scores:

| Criterion Type | Default Verdict | Rationale |
|----------------|-----------------|-----------|
| Positive (weight > 0) | UNMET | Assume requirement not met |
| Negative (weight < 0) | MET | Assume error is present |

This ensures parse failures result in worst-case scores rather than artificially inflating results. For example, if an error-detection rubric has many negative criteria, parse failures won't incorrectly report "no errors found."

```python
# Example: Parse failure with mixed rubric
rubric = Rubric([
    Criterion(weight=1.0, requirement="Is helpful"),      # → UNMET on parse failure
    Criterion(weight=-1.0, requirement="Contains errors"), # → MET on parse failure
])

# If LLM returns invalid JSON, score = 0.0 (worst case)
# rather than being artificially inflated
```

## Loading Rubrics

```python
# Direct construction
rubric = Rubric([
    Criterion(weight=10.0, requirement="States Q4 2023 base margin as 17.2%"),
    Criterion(weight=8.0, requirement="Explicitly uses Shapley attribution for decomposition"),
    Criterion(weight=-15.0, requirement="Uses total deliveries instead of cash-only deliveries")
])

# From list of dictionaries
rubric = Rubric.from_dict([
    {"weight": 10.0, "requirement": "States Q4 2023 base margin as 17.2%"},
    {"weight": 8.0, "requirement": "Explicitly uses Shapley attribution for decomposition"},
    {"weight": -15.0, "requirement": "Uses total deliveries instead of cash-only deliveries"}
])

# From JSON string
rubric = Rubric.from_json('[{"weight": 10.0, "requirement": "Example requirement"}]')

# From YAML string
yaml_data = '''
- weight: 10.0
  requirement: "Example requirement"
'''
rubric = Rubric.from_yaml(yaml_data)

# From files
rubric = Rubric.from_file('rubric.json')
rubric = Rubric.from_file('rubric.yaml')
```

### JSON Format

```json
[
  {
    "weight": 10.0,
    "requirement": "States Q4 2023 base margin as 17.2%"
  },
  {
    "weight": 8.0,
    "requirement": "Explicitly uses Shapley attribution for decomposition"
  },
  {
    "weight": -15.0,
    "requirement": "Uses total deliveries instead of cash-only deliveries"
  }
]
```

### YAML Format

```yaml
- weight: 10.0
  requirement: "States Q4 2023 base margin as 17.2%"
- weight: 8.0
  requirement: "Explicitly uses Shapley attribution for decomposition"
- weight: -15.0
  requirement: "Uses total deliveries instead of cash-only deliveries"
```

## Requirements

- Python 3.10+
- An LLM API (e.g., OpenAI, Anthropic, OpenRouter) - set appropriate API keys as environment variables

## License

MIT License - see LICENSE file for details.
