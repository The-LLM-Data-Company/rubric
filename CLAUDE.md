# Rubric Package

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

## Package Structure

```
src/rubric/
├── __init__.py              # Public exports
├── rubric.py                # Core Rubric class
├── types.py                 # Type definitions (Criterion, protocols)
├── utils.py                 # Utility functions (default generators)
└── autograders/
    ├── __init__.py          # Autograder exports
    ├── base.py              # Abstract Autograder base class
    ├── schemas.py           # Pydantic output schemas
    ├── per_criterion_grader.py         # Parallel per-criterion evaluation
    ├── per_criterion_one_shot_grader.py # Single-call all-criteria evaluation
    └── rubric_as_judge_grader.py       # Holistic scoring
```

## Core Types

### `Criterion`
A single evaluation criterion with weight and requirement:
```python
class Criterion(BaseModel):
    weight: float      # Positive for desired traits, negative for errors
    requirement: str   # What to evaluate
```

### `CriterionReport`
A criterion with its evaluation result:
```python
class CriterionReport(Criterion):
    verdict: Literal["MET", "UNMET"]
    reason: str
```

### `EvaluationReport`
Final grading result:
```python
class EvaluationReport(BaseModel):
    score: float                              # Normalized 0-1 (or raw if normalize=False)
    raw_score: float | None                   # Always weighted-sum semantics (consistent across all graders)
    llm_raw_score: float | None               # Original LLM output before conversion
    report: list[CriterionReport] | None      # Per-criterion details (if available)
```

**Score Field Semantics:**
- `raw_score`: Always uses **weighted-sum semantics** regardless of grader type. This ensures training pipelines can use `raw_score` consistently without knowing which grader was used.
- `llm_raw_score`: The **original value** from the LLM before any conversion:
  - `PerCriterionGrader` / `PerCriterionOneShotGrader`: Same as `raw_score` (weighted sum)
  - `RubricAsJudgeGrader`: The 0-100 holistic score from the LLM (useful for debugging)

**Error Handling:**
- Validation happens at generation time via Pydantic models
- If your `generate_fn` returns invalid data, Pydantic will raise a `ValidationError`
- Users control retry logic in their `generate_fn` based on their LLM client's capabilities

## Pydantic Output Schemas

All autograders require typed `generate_fn` implementations that return validated Pydantic models. These schemas ensure strict type safety and enable constrained decoding.

### `PerCriterionOutput`
Used by `PerCriterionGrader` for single-criterion evaluation:

```python
from typing import Literal
from pydantic import BaseModel

class PerCriterionOutput(BaseModel):
    criterion_status: Literal["MET", "UNMET"]
    explanation: str
```

### `OneShotOutput`
Used by `PerCriterionOneShotGrader` for batch evaluation:

```python
from typing import Literal
from pydantic import BaseModel, Field

class CriterionEvaluation(BaseModel):
    criterion_number: int  # 1-based index
    criterion_status: Literal["MET", "UNMET"]
    explanation: str

class OneShotOutput(BaseModel):
    criteria_evaluations: list[CriterionEvaluation] = Field(min_length=1)
```

### `RubricAsJudgeOutput`
Used by `RubricAsJudgeGrader` for holistic scoring:

```python
from pydantic import BaseModel

class RubricAsJudgeOutput(BaseModel):
    overall_score: float  # 0-100 scale
    explanation: str      # Brief explanation of the score
```

### Accessing Schemas for Constrained Decoding

All schemas expose `.model_json_schema()` for constrained decoding:

```python
from rubric import PerCriterionOutput

# Get JSON schema for your LLM client
schema = PerCriterionOutput.model_json_schema()

# Example with OpenAI structured outputs
response = await openai_client.chat.completions.create(
    model="gpt-4",
    messages=[...],
    response_format={"type": "json_schema", "json_schema": {
        "name": "criterion_output",
        "schema": schema
    }}
)

# Parse response into validated Pydantic model
output = PerCriterionOutput.model_validate_json(response.choices[0].message.content)
```

## Typed GenerateFn Protocols

Each autograder requires a specific typed `generate_fn` protocol that returns a validated Pydantic model:

### `PerCriterionGenerateFn`
For `PerCriterionGrader`:
```python
from rubric import PerCriterionGenerateFn, PerCriterionOutput

async def my_generate_fn(
    system_prompt: str,
    user_prompt: str,
    **kwargs
) -> PerCriterionOutput:
    # Your LLM call here
    return PerCriterionOutput(
        criterion_status="MET",
        explanation="The criterion is satisfied."
    )
```

### `OneShotGenerateFn`
For `PerCriterionOneShotGrader`:
```python
from rubric import OneShotGenerateFn, OneShotOutput, CriterionEvaluation

async def my_generate_fn(
    system_prompt: str,
    user_prompt: str,
    **kwargs
) -> OneShotOutput:
    # Your LLM call here
    return OneShotOutput(
        criteria_evaluations=[
            CriterionEvaluation(
                criterion_number=1,
                criterion_status="MET",
                explanation="First criterion satisfied"
            ),
            # ... more evaluations
        ]
    )
```

### `RubricAsJudgeGenerateFn`
For `RubricAsJudgeGrader`:
```python
from rubric import RubricAsJudgeGenerateFn, RubricAsJudgeOutput

async def my_generate_fn(
    system_prompt: str,
    user_prompt: str,
    **kwargs
) -> RubricAsJudgeOutput:
    # Your LLM call here
    return RubricAsJudgeOutput(overall_score=85.0, explanation="Good quality overall")
```

## Autograders

All autograders inherit from `Autograder` and implement the `judge()` and `aggregate()` methods.

### Common Constructor Parameters

All autograders accept these parameters:
```python
def __init__(
    self,
    generate_fn: TypedGenerateFn,                   # Typed LLM generation function (required)
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,     # Customizable system prompt
    normalize: bool = True,                          # If False, return raw weighted sums
):
```

### `PerCriterionGrader` (Default)
Evaluates each criterion in **parallel LLM calls**. Best for accuracy when you have many criteria.

```python
from rubric import PerCriterionGenerateFn, PerCriterionOutput
from rubric.autograders import PerCriterionGrader

async def my_generate_fn(system_prompt: str, user_prompt: str) -> PerCriterionOutput:
    # Your LLM call here
    ...

grader = PerCriterionGrader(generate_fn=my_generate_fn)
```

- **judge()**: Makes one LLM call per criterion concurrently via `asyncio.gather()`
- **aggregate()**: Computes weighted score from individual verdicts
- Returns detailed `CriterionReport` for each criterion
- Requires: `PerCriterionGenerateFn` returning `PerCriterionOutput`

### `PerCriterionOneShotGrader`
Evaluates **all criteria in a single LLM call**. Best for cost efficiency with fewer criteria.

```python
from rubric import OneShotGenerateFn, OneShotOutput
from rubric.autograders import PerCriterionOneShotGrader

async def my_generate_fn(system_prompt: str, user_prompt: str) -> OneShotOutput:
    # Your LLM call here
    ...

grader = PerCriterionOneShotGrader(generate_fn=my_generate_fn)
```

- **judge()**: Single LLM call with all criteria in the prompt
- **aggregate()**: Same weighted scoring as PerCriterionGrader
- Returns detailed `CriterionReport` for each criterion
- Requires: `OneShotGenerateFn` returning `OneShotOutput`

### `RubricAsJudgeGrader`
Asks the LLM for a **single holistic score** (0-100). Fastest but no per-criterion breakdown.

```python
from rubric import RubricAsJudgeGenerateFn, RubricAsJudgeOutput
from rubric.autograders import RubricAsJudgeGrader

async def my_generate_fn(system_prompt: str, user_prompt: str) -> RubricAsJudgeOutput:
    # Your LLM call here
    ...

grader = RubricAsJudgeGrader(generate_fn=my_generate_fn)
```

- **judge()**: Single LLM call that returns overall score (0-100)
- **aggregate()**: Converts to weighted-sum `raw_score` for consistency, normalizes to 0-1 for `score`
- Returns `report=None` (no per-criterion details)
- `llm_raw_score` preserves the original 0-100 LLM score for debugging
- Requires: `RubricAsJudgeGenerateFn` returning `RubricAsJudgeOutput`

**raw_score Conversion**: The LLM's 0-100 score is converted to weighted-sum semantics:
```python
raw_score = (llm_score / 100.0) * total_positive_weight
```
This ensures `raw_score` is comparable across all grader types for training pipelines.

## Grade Calculation

The score calculation works as follows:

1. **Positive criteria** (weight > 0): MET earns the weight, UNMET earns 0
2. **Negative criteria** (weight < 0): MET subtracts the weight, UNMET contributes 0
3. **Final score** = (weighted_sum) / (total_positive_weight), clamped to [0, 1]

```python
total_positive_weight = sum(max(0.0, c.weight) for c in criteria)
weighted_sum = sum((1.0 if verdict == "MET" else 0.0) * c.weight for c in criteria)
score = max(0.0, min(1.0, weighted_sum / total_positive_weight))
```

### Raw Scores for Training (normalize=False)

For RL training scenarios, normalized 0-1 scores can make optimization artificially difficult. Pass `normalize=False` to get raw weighted sums:

```python
grader = PerCriterionGrader(generate_fn=your_generate_fn, normalize=False)
result = await rubric.grade(response, autograder=grader)
# result.score = raw weighted sum (can be negative if many negative criteria are MET)
# result.raw_score = same as score when normalize=False
```

With normalized scores (default):
```python
# A rubric with weights [10, 5, -3]
# If all positive criteria MET, no negative criteria MET:
# weighted_sum = 10 + 5 + 0 = 15
# score = 15 / 15 = 1.0 (normalized)
# raw_score = 15 (always available)
```

With raw scores:
```python
grader = PerCriterionGrader(generate_fn=your_generate_fn, normalize=False)
# Same scenario:
# score = 15 (raw weighted sum)
# raw_score = 15
```

The `raw_score` field is **always populated** regardless of the `normalize` setting, giving you access to both views.

**Cross-Grader Consistency**: `raw_score` now uses consistent weighted-sum semantics across all graders:
```python
# Same rubric, different graders - raw_score is now comparable!
result1 = await rubric.grade(text, autograder=PerCriterionGrader(generate_fn=your_generate_fn, normalize=False))
result2 = await rubric.grade(text, autograder=RubricAsJudgeGrader(generate_fn=your_generate_fn, normalize=False))

# Both use weighted-sum semantics for raw_score
# result1.raw_score and result2.raw_score are on the same scale

# For RubricAsJudgeGrader, the original 0-100 LLM score is in llm_raw_score
print(result2.llm_raw_score)  # e.g., 85.0 (original LLM output)
print(result2.raw_score)      # e.g., 12.75 (converted to weighted sum)
```

## Training / RL Use Cases

For reinforcement learning training, you typically want raw (unnormalized) scores that can be positive or negative, rather than everything squeezed into 0-1. The rubric package supports this via the `normalize` parameter.

### Basic Training Setup

```python
from rubric import Rubric
from rubric.autograders import PerCriterionGrader

# Configure for training: raw scores
grader = PerCriterionGrader(
    generate_fn=your_llm_fn,
    normalize=False,  # Return raw weighted sums
)

rubric = Rubric.from_file("rubric.yaml")
result = await rubric.grade(response, autograder=grader)

# result.score = raw weighted sum
# result.raw_score = raw weighted sum
```

### Batch Processing

For batch reward computation during training:

```python
async def compute_rewards_batch(
    responses: list[str],
    rubrics: list[Rubric],
    queries: list[str] | None = None,
) -> list[float]:
    tasks = []
    for i, (response, rubric) in enumerate(zip(responses, rubrics)):
        query = queries[i] if queries else None
        tasks.append(rubric.grade(response, autograder=grader, query=query))
    
    results = await asyncio.gather(*tasks)
    return [r.score for r in results]
```

### Key Differences from Normalized Mode

| Aspect | Normalized (default) | Training (normalize=False) |
|--------|---------------------|---------------------------|
| Score range | 0.0 to 1.0 | Can be negative or > 1 |
| Clamping | Score clamped to [0, 1] | No clamping |
| Use case | Evaluation, reporting | RL reward signals |

## Creating Custom Autograders

Subclass `Autograder` and implement `judge()` and `aggregate()`. The only requirements are:
1. Implement the abstract methods `judge()` and `aggregate()`
2. Call `super().__init__(normalize=...)`

How you implement grading logic is up to you - you can use a `generate_fn` parameter (like the built-in autograders), make LLM calls directly, use multiple functions, or even do rule-based grading without LLMs.

**Example with optional `generate_fn` pattern:**

```python
from typing import Protocol
from rubric.autograders import Autograder
from rubric.types import Criterion, EvaluationReport
from pydantic import BaseModel

# 1. Define your Pydantic output schema (if using LLMs)
class MyCustomOutput(BaseModel):
    score: float
    reasoning: str

# 2. Optionally define a typed protocol for generate_fn
class MyCustomGenerateFn(Protocol):
    async def __call__(
        self, system_prompt: str, user_prompt: str, **kwargs
    ) -> MyCustomOutput: ...

# 3. Implement your autograder
class MyAutograder(Autograder):
    def __init__(
        self,
        generate_fn: MyCustomGenerateFn,  # Optional - just an implementation choice
        *,
        system_prompt: str = "Grade this response.",
        normalize: bool = True,
    ):
        super().__init__(normalize=normalize)
        self.generate_fn = generate_fn
        self.system_prompt = system_prompt

    async def judge(
        self, to_grade: str, rubric: list[Criterion], query: str | None = None
    ) -> MyCustomOutput:
        # Build prompt with rubric and response
        user_prompt = f"Response: {to_grade}\nCriteria: {rubric}"

        # Call LLM with your chosen approach
        result = await self.generate_fn(self.system_prompt, user_prompt)
        return result

    async def aggregate(self, judge_results: MyCustomOutput) -> EvaluationReport:
        # Convert judge results to EvaluationReport
        return EvaluationReport(
            score=judge_results.score,
            raw_score=judge_results.score,
            llm_raw_score=None,
            report=None,
        )
```

**Key points:**
- The `generate_fn` pattern used by built-in autograders is optional - it's just one way to structure your code
- You could make LLM calls directly in `judge()`, use multiple functions, or skip LLMs entirely
- Call the base constructor with the `normalize` parameter
- Store any autograder-specific parameters as instance attributes

## Public Exports

### Core Types
```python
from rubric import (
    # Core classes
    Rubric,
    Criterion,
    CriterionReport,
    EvaluationReport,

    # Pydantic output schemas
    PerCriterionOutput,
    OneShotOutput,
    RubricAsJudgeOutput,
    CriterionEvaluation,

    # Typed protocols
    PerCriterionGenerateFn,
    OneShotGenerateFn,
    RubricAsJudgeGenerateFn,

    # Default generate functions
    default_per_criterion_generate_fn,
    default_oneshot_generate_fn,
    default_rubric_as_judge_generate_fn,
)
```

### Autograders
```python
from rubric.autograders import (
    Autograder,              # Base class
    PerCriterionGrader,      # Parallel per-criterion grading
    PerCriterionOneShotGrader,  # Single-call all-criteria grading
    RubricAsJudgeGrader,     # Holistic scoring
)
```



