# Rubric Package

A Python library for evaluating text outputs against weighted criteria using LLM-as-a-judge.

## Package Structure

```
src/rubric/
├── __init__.py              # Public exports
├── rubric.py                # Core Rubric class
├── types.py                 # Type definitions (Criterion, LengthPenalty, etc.)
├── utils.py                 # Utility functions (JSON parsing, length penalty)
└── autograders/
    ├── __init__.py          # Autograder exports
    ├── base.py              # Abstract Autograder base class
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
    error: str | None                         # Error message if grading failed (e.g., parse error)
```

**Score Field Semantics:**
- `raw_score`: Always uses **weighted-sum semantics** regardless of grader type. This ensures training pipelines can use `raw_score` consistently without knowing which grader was used.
- `llm_raw_score`: The **original value** from the LLM before any conversion:
  - `PerCriterionGrader` / `PerCriterionOneShotGrader`: Same as `raw_score` (weighted sum)
  - `RubricAsJudgeGrader`: The 0-100 holistic score from the LLM (useful for debugging)

**Error Handling:**
- `error`: Set when grading fails (e.g., JSON parse error). When set, `score` defaults to 0.0.
- Training pipelines should filter out results where `error is not None` to avoid corrupted data.
- A warning is logged when parse errors occur, including a preview of the unparseable response.

**Retry and Fallback Behavior:**
- By default, autograders retry up to 2 times (3 total attempts) when parsing fails.
- If all retries fail, a `ValueError` is raised (loud failure).
- Set `default_fallback_verdicts` to configure fallback verdicts per criterion type instead of raising.
- Configure `max_retries` to adjust the number of retry attempts.
- We recommend using structured outputs in your `generate_fn` when possible to avoid parse failures.

### `DefaultFallbackVerdicts`
Configuration for fallback verdicts when parsing fails:
```python
class DefaultFallbackVerdicts(TypedDict, total=False):
    positive: Literal["MET", "UNMET"]  # Fallback for positive criteria (default: "UNMET")
    negative: Literal["MET", "UNMET"]  # Fallback for negative criteria (default: "UNMET")
```

Example:
```python
# Conservative fallbacks (worst-case assumptions)
grader = PerCriterionGrader(default_fallback_verdicts={"positive": "UNMET", "negative": "MET"})

# All UNMET fallbacks
grader = PerCriterionGrader(default_fallback_verdicts={"positive": "UNMET", "negative": "UNMET"})
```

### `LengthPenalty`
Configuration for penalizing overly long outputs (see Length Penalty section below).

### `GenerateFn`
Protocol for LLM generation functions:
```python
async def __call__(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str
```

## Autograders

All autograders inherit from `Autograder` and implement the `judge()` and `aggregate()` methods.

### Base Constructor

All autograders accept these common parameters:
```python
def __init__(
    self,
    generate_fn: GenerateFn = default_generate_fn,  # LLM generation function
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,     # Customizable system prompt
    length_penalty: LengthPenalty | None = None,    # Optional length penalty config
    normalize: bool = True,                          # If False, return raw weighted sums
    max_retries: int = 2,                            # Retry attempts on parse failure (3 total)
    default_fallback_verdicts: DefaultFallbackVerdicts | None = None,  # Fallback verdicts on failure
):
```

### `PerCriterionGrader` (Default)
Evaluates each criterion in **parallel LLM calls**. Best for accuracy when you have many criteria.

- **judge()**: Makes one LLM call per criterion concurrently via `asyncio.gather()`
- **aggregate()**: Computes weighted score from individual verdicts
- Returns detailed `CriterionReport` for each criterion

### `PerCriterionOneShotGrader`
Evaluates **all criteria in a single LLM call**. Best for cost efficiency with fewer criteria.

- **judge()**: Single LLM call with all criteria in the prompt
- **aggregate()**: Same weighted scoring as PerCriterionGrader
- Returns detailed `CriterionReport` for each criterion

### `RubricAsJudgeGrader`
Asks the LLM for a **single holistic score** (0-100). Fastest but no per-criterion breakdown.

- **judge()**: Single LLM call that returns overall score (0-100)
- **aggregate()**: Converts to weighted-sum `raw_score` for consistency, normalizes to 0-1 for `score`
- Returns `report=None` (no per-criterion details)
- `llm_raw_score` preserves the original 0-100 LLM score for debugging

**raw_score Conversion**: The LLM's 0-100 score is converted to weighted-sum semantics:
```python
raw_score = (llm_score / 100.0) * total_positive_weight
```
This ensures `raw_score` is comparable across all grader types for training pipelines.

## Grade Calculation

The base score calculation (before length penalty) works as follows:

1. **Positive criteria** (weight > 0): MET earns the weight, UNMET earns 0
2. **Negative criteria** (weight < 0): MET subtracts the weight, UNMET contributes 0
3. **Final score** = (weighted_sum) / (total_positive_weight), clamped to [0, 1]

```python
total_positive_weight = sum(max(0.0, c.weight) for c in criteria)
weighted_sum = sum((1.0 if verdict == "MET" else 0.0) * c.weight for c in criteria)
score = max(0.0, min(1.0, weighted_sum / total_positive_weight))
```

**Important**: The length penalty is applied **after** this calculation.

### Raw Scores for Training (normalize=False)

For RL training scenarios, normalized 0-1 scores can make optimization artificially difficult. Pass `normalize=False` to get raw weighted sums:

```python
grader = PerCriterionGrader(normalize=False)
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
grader = PerCriterionGrader(normalize=False)
# Same scenario:
# score = 15 (raw weighted sum)
# raw_score = 15
```

The `raw_score` field is **always populated** regardless of the `normalize` setting, giving you access to both views.

**Cross-Grader Consistency**: `raw_score` now uses consistent weighted-sum semantics across all graders:
```python
# Same rubric, different graders - raw_score is now comparable!
result1 = await rubric.grade(text, autograder=PerCriterionGrader(normalize=False))
result2 = await rubric.grade(text, autograder=RubricAsJudgeGrader(normalize=False))

# Both use weighted-sum semantics for raw_score
# result1.raw_score and result2.raw_score are on the same scale

# For RubricAsJudgeGrader, the original 0-100 LLM score is in llm_raw_score
print(result2.llm_raw_score)  # e.g., 85.0 (original LLM output)
print(result2.raw_score)      # e.g., 12.75 (converted to weighted sum)
```

## Length Penalty

Length penalty discourages excessively verbose outputs during training. It is configured on the **autograder constructor** and is applied **after** the base grade calculation. The penalty is **subtracted** from the score.

### Configuration

```python
class LengthPenalty(BaseModel):
    free_budget: int = 6000        # No penalty below this count
    max_cap: int = 8000            # Maximum penalty at/above this count
    penalty_at_cap: float = 0.5    # Max penalty to subtract from score
    exponent: float = 1.6          # Curve steepness (higher = more lenient near budget)
    count_fn: CountFn | None = None  # Custom counting function
    penalty_type: PenaltyType = "ALL"  # Which sections to count: "ALL", "OUTPUT_ONLY", "THINKING_ONLY"
```

For **normalized scores** (0-1), use fractional `penalty_at_cap` values like `0.5` (lose up to 50% of score).

For **raw scores** (training), use absolute `penalty_at_cap` values like `50.0` (subtract up to 50 points from raw score).

### Penalty Formula

```
if count <= free_budget:
    penalty = 0
elif count >= max_cap:
    penalty = penalty_at_cap
else:
    frac = (count - free_budget) / (max_cap - free_budget)
    penalty = penalty_at_cap * (frac ** exponent)

final_score = max(0.0, base_score - penalty)
```

### Default Counting

By default, `LengthPenalty` uses whitespace word counting (`text.split()`). The default values (6000/8000) are calibrated for word counts.

### Custom Tokenizer Example

For accurate token-based penalties (e.g., during RL training):

```python
from transformers import AutoTokenizer
from rubric import LengthPenalty

tokenizer = AutoTokenizer.from_pretrained("gpt2")

penalty = LengthPenalty(
    free_budget=8000,   # 8000 tokens free
    max_cap=10000,      # Max penalty at 10000 tokens
    penalty_at_cap=0.5, # Lose up to 50% of score
    exponent=1.6,
    count_fn=lambda text: len(tokenizer.encode(text))
)
```

### Usage

Length penalty is configured on the autograder, not on the grade() call:

```python
from rubric import Rubric, LengthPenalty
from rubric.autograders import PerCriterionGrader

rubric = Rubric.from_file("rubric.yaml")

# Without length penalty
result = await rubric.grade(response)

# With length penalty - configure on the autograder
grader = PerCriterionGrader(length_penalty=LengthPenalty())
result = await rubric.grade(response, autograder=grader)

# With custom tokenizer
grader = PerCriterionGrader(
    length_penalty=LengthPenalty(
        free_budget=8000,
        max_cap=10000,
        count_fn=lambda t: len(tokenizer.encode(t))
    )
)
result = await rubric.grade(response, autograder=grader)
```

### Key Points

1. **Length penalty only subtracts** - it cannot increase the score
2. **Configured on autograder** - pass `length_penalty` to the autograder constructor
3. **Applied after aggregation** - the base rubric score is computed first, then penalty is subtracted
4. **Final score is clamped to 0** - `max(0.0, score - penalty)` when normalized
5. **Configurable curve** - the exponent controls how quickly penalty ramps up after free_budget

## Thinking/Output Token Support

For models that generate thinking/reasoning steps separately from final output (e.g., Claude with extended thinking), you can apply length penalties to specific sections.

### Input Formats

The `to_grade` parameter accepts three formats:

**1. Dict Format (Explicit)**
```python
await rubric.grade({
    "thinking": "Let me reason through this step by step...",
    "output": "The final answer is 42"
})
```

**2. String with Markers (Auto-parsed)**
```python
await rubric.grade(
    "<thinking>My reasoning process...</thinking><output>Final answer</output>"
)
```

**3. Plain String (Backwards Compatible)**
```python
await rubric.grade("Just a regular response")  # Treated as all output
```

### XML Tag Structure in Prompts

The autograders wrap your content in `<response>` XML tags when sending to the LLM. If a `query` is provided, it's wrapped in `<query>` tags (this is optional). If you provide a custom `system_prompt`, ensure it handles the response content appropriately. The response may contain:

**Nested structure (thinking/output):**
```xml
<response>
<thinking>{thinking_content}</thinking>
<output>{output_content}</output>
</response>
```

**Plain string:**
```xml
<response>
{content}
</response>
```

The structure depends on what you pass to `rubric.grade()`. Customize your system prompt accordingly.

### Penalty Type Selection

Use the `penalty_type` parameter in `LengthPenalty` to control which sections are counted:

```python
penalty = LengthPenalty(
    free_budget=8000,
    max_cap=10000,
    penalty_at_cap=0.5,
    penalty_type="OUTPUT_ONLY"  # Options: "ALL", "OUTPUT_ONLY", "THINKING_ONLY"
)
```

**Penalty Types:**
- `"ALL"` - Count both thinking and output tokens (default, backwards compatible)
- `"OUTPUT_ONLY"` - Only count output tokens (useful for RL training to allow long reasoning)
- `"THINKING_ONLY"` - Only count thinking tokens (penalize excessive reasoning)

### Use Cases

**1. RL Training: Allow Long Reasoning, Penalize Verbose Output**
```python
from transformers import AutoTokenizer
from rubric import Rubric, LengthPenalty
from rubric.autograders import PerCriterionGrader

tokenizer = AutoTokenizer.from_pretrained("your-model")

grader = PerCriterionGrader(
    normalize=False,  # Raw scores for training
    length_penalty=LengthPenalty(
        free_budget=8000,
        max_cap=10000,
        penalty_at_cap=50.0,  # Absolute penalty for raw scores
        penalty_type="OUTPUT_ONLY",  # Don't penalize thinking
        count_fn=lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    )
)

# Grade with separate thinking and output
result = await rubric.grade(
    {"thinking": long_reasoning, "output": final_answer},
    autograder=grader
)
# result.score = raw weighted sum - output length penalty
```

**2. Penalize Excessive Reasoning**
```python
grader = PerCriterionGrader(
    length_penalty=LengthPenalty(
        free_budget=5000,
        penalty_type="THINKING_ONLY",  # Only penalize long thinking
    )
)
```

**3. Claude API with Extended Thinking**
```python
# Claude API returns separate thinking and content
response = await claude_client.messages.create(
    model="claude-sonnet-4-5",
    extended_thinking=True,
    ...
)

# Pass to rubric
result = await rubric.grade({
    "thinking": response.thinking,
    "output": response.content[0].text
}, autograder=grader)
```

### Backwards Compatibility

All existing code continues to work without changes:
- Plain strings are treated as output (no thinking section)
- `LengthPenalty` without `penalty_type` defaults to `"ALL"`
- String with markers is automatically parsed when `LengthPenalty` is configured

## Training / RL Use Cases

For reinforcement learning training, you typically want raw (unnormalized) scores that can be positive or negative, rather than everything squeezed into 0-1. The rubric package supports this via the `normalize` parameter.

### Basic Training Setup

```python
from transformers import AutoTokenizer
from rubric import Rubric, LengthPenalty
from rubric.autograders import PerCriterionGrader

# Load tokenizer for accurate token counting
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Configure for training: raw scores + absolute length penalty
grader = PerCriterionGrader(
    generate_fn=your_llm_fn,
    normalize=False,  # Return raw weighted sums
    length_penalty=LengthPenalty(
        free_budget=8000,      # 8000 tokens free
        max_cap=10000,         # Max penalty at 10000 tokens
        penalty_at_cap=50.0,   # Subtract up to 50 points (absolute)
        exponent=1.6,
        count_fn=lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    )
)

rubric = Rubric.from_file("rubric.yaml")
result = await rubric.grade(response, autograder=grader)

# result.score = raw weighted sum - length penalty
# result.raw_score = raw weighted sum (before length penalty)
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
| Length penalty | Fractional (e.g., 0.5) | Absolute (e.g., 50.0) |
| Clamping | Score clamped to [0, 1] | No clamping |
| Use case | Evaluation, reporting | RL reward signals |

## Creating Custom Autograders

Subclass `Autograder` and implement `judge()` and `aggregate()`:

```python
from rubric.autograders import Autograder
from rubric.types import Criterion, EvaluationReport, GenerateFn, LengthPenalty

class MyAutograder(Autograder):
    def __init__(
        self,
        generate_fn: GenerateFn,
        *,
        length_penalty: LengthPenalty | None = None,
    ):
        super().__init__(generate_fn=generate_fn, length_penalty=length_penalty)
    
    async def judge(self, to_grade: str, rubric: list[Criterion], query: str | None = None) -> Any:
        # Your judging logic here
        pass
    
    async def aggregate(self, judge_results: Any) -> EvaluationReport:
        # Convert judge results to EvaluationReport
        pass
```

The base `grade()` method handles length penalty application automatically using `self.length_penalty`.

## Public Exports

```python
from rubric import (
    Rubric,
    Criterion,
    CriterionReport,
    DefaultFallbackVerdicts,
    EvaluationReport,
    LengthPenalty,
    CountFn,
    # Thinking/output support
    PenaltyType,
    ThinkingOutputDict,
    ToGradeInput,
    # Utility functions
    compute_length_penalty,
    normalize_to_grade_input,
    parse_thinking_output,
    word_count,
)
```

