# API Testing Results

!!! info "Test Configuration"
    **Model:** jev-1.13.0 (`jev-latest`) · **Base URL:** `https://api.typesafe.ai` · **Endpoint:** `POST /v1/systemone`

## Test 1: Basic Noul (Yes/No Probability)

**State:** Urgent customer support message about Stripe integration failing for 3 days.

```json
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP.",
  "model": "jev-latest",
  "questions": {
    "urgency": {
      "type": "noul",
      "instructions": "Does this message express urgency?"
    }
  }
}
```

**Result:** `urgency.noul = 0.98` — near certainty. Correct.

## Test 2: Multiple Nouls (Contrast)

**State:** Non-urgent typo report.

| Question | Noul | Interpretation |
|----------|------|----------------|
| urgency | 0.04 | Not urgent ✅ |
| sentiment (positive?) | 0.65 | Mildly positive ✅ |
| action_required | 0.12 | No immediate action needed ✅ |

Usage: 323 input tokens, 55 output tokens.

## Test 3: All Three Primitives Together

**State:** Mixed customer feedback: "Your product is amazing but the onboarding was confusing. I almost gave up on day 2. Now I love it though."

| Question | Type | Result |
|----------|------|--------|
| category | choice | `mixed_feedback` (confidence 1.0, probability 1.0) ✅ |
| churn_risk | noul | 0.21 (low risk — they're happy now) ✅ |
| onboarding_issue | noul | 0.98 (clearly mentions onboarding problems) ✅ |
| satisfaction | score | 3.49/4.0 (between "somewhat satisfied" and "very satisfied") ✅ |

**Score distribution:**

- Level 0 (very dissatisfied): 0.0
- Level 1 (somewhat dissatisfied): 0.01
- Level 2 (neutral): 0.0
- Level 3 (somewhat satisfied, minor issues): 0.49
- Level 4 (very satisfied, enthusiastic): 0.50

Usage: 519 input tokens, 114 output tokens.

## Test 4: Code Review Scenario (Structured JSON State)

**State:**
```json
{
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "context": "Production API endpoint handling 10k requests/second"
}
```

| Question | Type | Result |
|----------|------|--------|
| language | choice | `python` (confidence 1.0) ✅ |
| has_performance_issue | noul | 0.98 (exponential recursion flagged) ✅ |
| code_quality | score | 0.02/4.0 (not suitable for production, confidence 0.99) ✅ |

The model correctly identified:

- The language (trivial)
- The exponential recursion as a performance problem given the 10k req/s context
- The code as completely unsuitable for production

Usage: 514 input tokens, 87 output tokens.

## Test 5: Valid Question Types (Error-Driven Discovery)

Attempting `type: "pick"` returned an error revealing all valid types:

- `noul` — yes/no probability
- `choice` — select from options (criteria is a `dict` of `{option_name: description}`)
- `score` — rate on levels (criteria is a `list` of level descriptions)
- `bounding_box` — not yet documented in detail

## Test 6: Available Models

`GET /v1/models` returned:

| Model | Description | Release Date |
|-------|-------------|-------------|
| `jev-latest` | Latest iteration of Jev | 2026-09-10 |
| `jev-preview` | Preview version, should be better in most ways | 2026-09-10 |

## Schema Summary

### Request

```json
{
  "state": "<string | object | array>",
  "model": "jev-latest",
  "questions": {
    "<question_id>": {
      "type": "noul",
      "instructions": "<yes/no question>",
      "criteria": {"true": "...", "false": "..."}  // optional
    },
    "<question_id>": {
      "type": "choice",
      "instructions": "<which one?>",
      "criteria": {"option_a": "description", "option_b": "description"}
    },
    "<question_id>": {
      "type": "score",
      "instructions": "<rate this>",
      "criteria": ["level 0 description", "level 1 description", "level 2 description"]
    }
  }
}
```

### Response

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "<question_id>": {"type": "noul", "noul": 0.92},
    "<question_id>": {"type": "choice", "choice": "option_a", "confidence": 0.85, "probabilities": {"option_a": 0.85, "option_b": 0.15}},
    "<question_id>": {"type": "score", "score": 1.45, "confidence": 0.33, "legend": {"0": "...", "1": "...", "2": "..."}, "probabilities": {"0": 0.0, "1": 0.55, "2": 0.45}}
  },
  "usage": {"input_tokens": 519, "output_tokens": 114}
}
```

## Observations

1. **All results were sensible and well-calibrated** — high confidence when the answer is obvious, low confidence when genuinely ambiguous.
2. **Multi-question calls are efficient** — 4 questions cost only ~500 input tokens total.
3. **Structured JSON state works well** — the model understands nested JSON objects and can reason about code quality in context.
4. **Error messages are helpful** — invalid types return the full list of valid options with their enum values.
5. **Score values can be fractional** — a score of 3.49 on a 0–4 scale indicates the model splits probability between adjacent levels rather than rounding.
