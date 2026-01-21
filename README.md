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

### Quick Start with Default Generate Functions

For quick testing, use the built-in Gemini generate functions:

```bash
export GEMINI_API_KEY=your_api_key_here
```

```python
import asyncio
from rubric import Rubric, default_per_criterion_generate_fn
from rubric.autograders import PerCriterionGrader

async def main():
    rubric = Rubric.from_dict([
        {"weight": 10.0, "requirement": "Response mentions Paris"},
        {"weight": 5.0, "requirement": "Response is concise"}
    ])

    grader = PerCriterionGrader(generate_fn=default_per_criterion_generate_fn)
    result = await rubric.grade("Paris is the capital of France.", autograder=grader)
    print(f"Score: {result.score}")

asyncio.run(main())
```

See `examples/basic_usage.py` for more examples with all three autograder types.

### Custom Generate Function with OpenAI

For production use, implement your own `generate_fn` with structured outputs:

```python
import asyncio
import os
from openai import AsyncOpenAI
from rubric import Rubric, PerCriterionOutput
from rubric.autograders import PerCriterionGrader

# Declare custom generate function with any model and inference provider
async def generate_with_openai(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = await client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_schema", "json_schema": {
            "name": "criterion_output",
            "schema": PerCriterionOutput.model_json_schema()
        }},
        max_tokens=400,
        temperature=0.0,
    )
    content = response.choices[0].message.content or "{}"
    return PerCriterionOutput.model_validate_json(content)

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
        normalize=False,  # Raw weighted sums
        system_prompt="This overrides the default grader system prompt",
    )

    # Grade output
    result = await rubric.grade(
        query="Input query...",
        to_grade="Output to evaluate...",
        autograder=grader
    )

    print(f"Score: {result.score:.2f}")  # Raw weighted sum
    for criterion in result.report:
        print(f"  [{criterion.verdict}] {criterion.requirement}")
        print(f"    → {criterion.reason}")

asyncio.run(main())
```

## Autograder Strategies

### PerCriterionGrader

Evaluates each criterion in parallel inference calls.

**Scoring Formula:**

For each criterion $i$: MET contributes $w_i$, UNMET contributes 0.

**Raw score**:

$$
\texttt{raw\\_score} = \sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i
$$

**Normalized score** (`normalize=True`, the default):

$$
\texttt{score} = \max\left(0, \min\left(1, \frac{\texttt{raw\\_score}}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

Pass `normalize=False` to the autograder constructor for raw weighted sums.

**All-Negative Criteria Rubrics:**

For rubrics with only negative criteria (e.g., error detection):

**Raw score**: Same formula as above. Will be ≤ 0 since all weights are negative.

**Normalized score** (default):

$$
\texttt{score} = \max\left(0, \min\left(1, 1 + \frac{\texttt{raw\\_score}}{\sum_{i=1}^{n} |w_i|}\right)\right)
$$

Score = 1.0 when all errors avoided (all UNMET, `raw_score` = 0)
Score = 0.0 when all errors present (all MET, `raw_score` = -total)

### PerCriterionOneShotGrader

Makes 1 inference call for all criteria (vs. $n$ parallel calls). Same scoring as PerCriterionGrader:

**Raw score**:

$$
\texttt{raw\\_score} = \sum_{i=1}^{n} \mathbb{1}[\text{verdict}_i = \text{MET}] \cdot w_i
$$

**Normalized score** (`normalize=True`, the default):

$$
\texttt{score} = \max\left(0, \min\left(1, \frac{\texttt{raw\\_score}}{\sum_{i=1}^{n} \max(0, w_i)}\right)\right)
$$

### DoublePassPerCriterionOneShotGrader

Makes 2 parallel inference calls with criteria in original and reversed order to reduce position bias. Uses **conservative reconciliation**:

- **Positive criteria** (weight > 0): MET only if BOTH passes agree
- **Negative criteria** (weight < 0): MET if EITHER pass detects the error

This produces rigorous evaluations where high scores require consistent evidence, and errors are caught even if only one pass notices them. Same scoring formula as PerCriterionOneShotGrader after reconciliation.

### RubricAsJudgeGrader

Holistic evaluation where the model returns a single 0-100 score.

**LLM raw score**: The model's direct 0-100 output, preserved in `llm_raw_score`.

**Raw score** (converted to weighted-sum for cross-grader consistency):

$$
\texttt{raw\\_score} = \frac{\texttt{llm\\_raw\\_score}}{100} \times \sum_{i=1}^{n} \max(0, w_i)
$$

**Normalized score** (`normalize=True`, default): $\texttt{score} = \max(0, \min(1, \texttt{llm\\_raw\\_score} / 100))$

**Unnormalized score** (`normalize=False`): $\texttt{score} = \texttt{raw\\_score}$

### Default System Prompts

Each autograder uses a specialized system prompt optimized for its evaluation approach:

**PerCriterionGrader** - Detailed criterion-by-criterion evaluation with strict JSON formatting requirements. The prompt instructs the LLM to evaluate each criterion independently, handling both positive and negative criteria with specific response formats.

**PerCriterionOneShotGrader** - Streamlined prompt for evaluating all criteria in a single response. Focuses on providing verdicts (MET/UNMET) and explanations for each criterion in a structured JSON format.

**DoublePassPerCriterionOneShotGrader** - Uses the same prompt as PerCriterionOneShotGrader but runs it twice (with reversed criteria order) and applies conservative reconciliation to reduce position bias.

**RubricAsJudgeGrader** - Holistic evaluation prompt that asks the LLM to consider the output as a whole and provide a single overall score from 0-100, taking into account the weights of all criteria.

You can view the complete default prompts in the source files:

- [`per_criterion_grader.py`](src/rubric/autograders/per_criterion_grader.py#L15)
- [`per_criterion_one_shot_grader.py`](src/rubric/autograders/per_criterion_one_shot_grader.py#L15)
- [`double_pass_per_criterion_one_shot_grader.py`](src/rubric/autograders/double_pass_per_criterion_one_shot_grader.py) (uses same prompt as PerCriterionOneShotGrader)
- [`rubric_as_judge_grader.py`](src/rubric/autograders/rubric_as_judge_grader.py#L14)

**Customizing System Prompts:** You can override the default system prompt by passing a `system_prompt` parameter to any autograder:

```python
grader = PerCriterionGrader(
    generate_fn=your_function,
    system_prompt="Your custom system prompt here"
)
```

**XML Tag Structure:** The autograders wrap content in `<response>` XML tags. If a `query` is provided (optional), it's wrapped in `<query>` tags. If you provide a custom system prompt, ensure it handles the response structure you're using:

```xml
<!-- Plain string response -->
<response>
{content}
</response>

<!-- Or nested with thinking/output -->
<response>
<thinking>{thinking_content}</thinking>
<output>{output_content}</output>
</response>
```

The structure depends on what you pass to `rubric.grade()`. Customize your system prompt to handle your preferred format.

## Customization

You can customize grading at multiple levels:

**1. Custom `generate_fn` (most common)**
Pass any typed function that returns a Pydantic model. Use any LLM provider (OpenAI, Anthropic, local models, etc.):

```python
from rubric import PerCriterionOutput

async def your_custom_function(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
    # Your LLM call here with structured outputs
    ...
    return PerCriterionOutput(criterion_status="MET", explanation="...")

grader = PerCriterionGrader(generate_fn=your_custom_function)
```

Each autograder requires a specific return type:
- `PerCriterionGrader` → `PerCriterionOutput`
- `PerCriterionOneShotGrader` → `OneShotOutput`
- `DoublePassPerCriterionOneShotGrader` → `OneShotOutput`
- `RubricAsJudgeGrader` → `RubricAsJudgeOutput`

**2. Create custom autograder**
Subclass `Autograder` and implement the abstract methods:

- `judge()` - Evaluates the submission and returns raw results
- `aggregate()` - Transforms judge results into an `EvaluationReport`

The `generate_fn` pattern is optional - you can make LLM calls directly, use multiple functions, or skip LLMs entirely.

**3. Override system prompts**
Customize the default prompts for built-in autograders:

```python
grader = PerCriterionGrader(
    generate_fn=your_function,
    system_prompt="Your custom system prompt here"
)
```

## Error Handling

Since v2.0.0, validation happens at generation time via Pydantic models. Your `generate_fn` is responsible for:

1. **Structured outputs** - Use your LLM provider's structured output features (JSON schema, function calling, etc.) to ensure valid responses
2. **Retry logic** - Implement retries within your `generate_fn` if needed
3. **Validation** - Return a validated Pydantic model (`PerCriterionOutput`, `OneShotOutput`, or `RubricAsJudgeOutput`)

If your `generate_fn` returns invalid data, Pydantic will raise a `ValidationError`.

**Example with retries:**

```python
from pydantic import ValidationError
from rubric import PerCriterionOutput

async def generate_with_retries(system_prompt: str, user_prompt: str, max_retries: int = 3) -> PerCriterionOutput:
    for attempt in range(max_retries):
        try:
            response = await your_llm_call(system_prompt, user_prompt)
            return PerCriterionOutput.model_validate_json(response)
        except ValidationError as e:
            if attempt == max_retries - 1:
                raise
            continue  # Retry on validation error
```

**Best practice:** Use structured outputs (JSON schema constrained decoding) in your LLM client to avoid validation errors entirely.

## Score Fields

| Field | Description |
|-------|-------------|
| `score` | Final score. 0-1 when `normalize=True` (default), `raw_score` when `normalize=False`. |
| `raw_score` | Raw weighted sum. Consistent across all graders. |
| `llm_raw_score` | Original LLM output. For `RubricAsJudgeGrader`: 0-100 score. For others: same as `raw_score`. |
| `report` | Per-criterion breakdown (`None` for `RubricAsJudgeGrader`). |

### The `normalize` Parameter

```python
# Default: normalized 0-1 scores
grader = PerCriterionGrader(generate_fn=your_function)
result = await rubric.grade(text, autograder=grader)
print(result.score)      # 0.85 (normalized)
print(result.raw_score)  # 12.75 (raw weighted sum)

# Raw scores
grader = PerCriterionGrader(generate_fn=your_function, normalize=False)
result = await rubric.grade(text, autograder=grader)
print(result.score)      # 12.75 (raw, can be negative)
print(result.raw_score)  # 12.75 (same as score)
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
