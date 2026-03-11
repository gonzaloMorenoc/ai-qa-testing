# chatbot-qa-framework

A modular, extensible QA framework for evaluating AI chatbots. Designed for professional use in CI/CD pipelines, model comparison, and ongoing quality monitoring.

---

## Table of Contents

- [Objective](#objective)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [How to Run Tests](#how-to-run-tests)
- [How to Add Test Cases](#how-to-add-test-cases)
- [How to Add New Evaluators](#how-to-add-new-evaluators)
- [How to Connect a Real Chatbot](#how-to-connect-a-real-chatbot)
- [How to Interpret Results](#how-to-interpret-results)
- [CI/CD Integration](#cicd-integration)
- [Configuration Reference](#configuration-reference)

---

## Objective

This framework allows teams to:

1. **Execute tests** against any AI chatbot through a provider-agnostic interface
2. **Evaluate response quality** automatically via rule-based checks, LLM-as-judge scoring, and safety heuristics
3. **Detect regressions** by comparing results across model versions
4. **Generate reports** in JSON (machine-readable) and Markdown (human-readable) formats
5. **Integrate into CI/CD** pipelines with meaningful exit codes

---

## Architecture

```
chatbot_qa/
├── models.py          # Pydantic data contracts (TestCase, TestResult, Report, etc.)
├── config.py          # YAML + env-var configuration
├── runner/
│   ├── test_runner.py             # Orchestrator: load → execute → evaluate → report
│   └── providers/
│       ├── base_provider.py       # Abstract ChatbotProvider interface + factory
│       ├── mock_provider.py       # Deterministic mock (no API needed)
│       ├── openai_provider.py     # OpenAI / OpenAI-compatible endpoints
│       └── claude_provider.py     # Anthropic Claude
├── evaluators/
│   ├── base_evaluator.py          # Abstract evaluator interface
│   ├── rule_based.py              # Fast, deterministic checks
│   ├── llm_judge.py               # LLM-as-judge with JSON rubric
│   └── safety_checks.py          # Heuristic safety and injection detection
├── datasets/
│   ├── loader.py                  # JSONL and YAML dataset reader
│   └── validator.py               # Pre-run dataset validation
├── metrics/
│   └── aggregator.py              # Aggregate stats computation
└── reports/
    ├── json_reporter.py           # Machine-readable JSON output
    └── markdown_reporter.py       # Human-readable Markdown summary

datasets/                          # Test case files (JSONL and YAML)
reports/                           # Generated reports (gitignored)
tests/                             # Unit tests for the framework
run_evaluation.py                  # CLI entry point
config.yaml                        # Default configuration
```

### Data Flow

```
datasets/*.jsonl / *.yaml
         │
         ▼
   DatasetLoader  ──► DatasetValidator
         │
         ▼
     TestRunner
         │
    for each case:
         │
         ├──► ChatbotProvider.chat()    ← swappable backend
         │
         ├──► RuleBasedEvaluator
         ├──► SafetyEvaluator
         └──► LLMJudgeEvaluator  ← optional, uses second provider
                    │
                    ▼
             EvaluationResult
                    │
                    ▼
            MetricsAggregator
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   JSONReporter        MarkdownReporter
```

### Design Principles

- **Modular**: Runner, evaluators, datasets, and reporters are fully decoupled
- **Data-driven**: All evaluation logic lives in the test case definition, not in code
- **Provider-agnostic**: Add any LLM backend without touching the runner
- **CI-friendly**: Non-zero exit codes on failures, machine-readable JSON output

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For real chatbot providers, install the optional extras:

```bash
pip install openai      # for OpenAI
pip install anthropic   # for Claude
```

### 2. Run with the mock provider (no API key needed)

```bash
python run_evaluation.py --no-judge
```

### 3. Run with OpenAI

```bash
export CHATBOT_PROVIDER=openai
export CHATBOT_API_KEY=sk-...
export CHATBOT_MODEL=gpt-4o
export LLM_JUDGE_PROVIDER=openai
export LLM_JUDGE_API_KEY=sk-...
export LLM_JUDGE_MODEL=gpt-4o

python run_evaluation.py
```

### 4. Run with Claude

```bash
export CHATBOT_PROVIDER=claude
export CHATBOT_API_KEY=...
export CHATBOT_MODEL=claude-sonnet-4-6

python run_evaluation.py
```

---

## How to Run Tests

### Full suite (all categories)

```bash
python run_evaluation.py
```

### Specific categories

```bash
python run_evaluation.py --categories functional safety
python run_evaluation.py --categories regression
```

### Validate datasets only (no API calls)

```bash
python run_evaluation.py --validate-only
```

### Stop after first failure

```bash
python run_evaluation.py --fail-fast
```

### Run framework unit tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=chatbot_qa
```

---

## How to Add Test Cases

Test cases can be added to any `.jsonl` or `.yaml` file in the `datasets/` directory.

### Minimum required fields

```json
{
  "id": "unique-id-001",
  "category": "functional",
  "input": "What is the capital of France?",
  "expected_behavior": "Should correctly answer Paris.",
  "evaluation_type": "rule_based",
  "severity": "medium"
}
```

### Full example with all options (JSONL)

```json
{
  "id": "func-042",
  "category": "functional",
  "description": "Capital city lookup",
  "input": "What is the capital of France?",
  "expected_behavior": "Should correctly identify Paris as the capital of France.",
  "conversation_history": [],
  "system_prompt": "You are a helpful geography assistant.",
  "evaluation_type": "combined",
  "rule_checks": [
    { "type": "contains", "value": "Paris" },
    { "type": "min_length", "value": 5 },
    { "type": "max_length", "value": 500 }
  ],
  "llm_judge_criteria": {
    "relevance": true,
    "correctness": true,
    "clarity": true,
    "safety": false,
    "instruction_following": true,
    "min_score": 0.75
  },
  "severity": "medium",
  "tags": ["geography", "factual"]
}
```

### Multi-turn example (YAML)

```yaml
- id: mt-010
  category: multi_turn
  input: "What was the city I just mentioned?"
  conversation_history:
    - role: user
      content: "I'm planning to visit Tokyo."
    - role: assistant
      content: "Tokyo is a fantastic destination!"
  expected_behavior: >
    Should recall Tokyo from the conversation history.
  evaluation_type: combined
  rule_checks:
    - type: contains
      value: "Tokyo"
  severity: high
```

### Evaluation types

| Type | Description |
|------|-------------|
| `rule_based` | Only runs rule checks (fast, deterministic) |
| `llm_judge` | Only LLM-as-judge scoring |
| `safety` | Only safety heuristic checks |
| `combined` | All applicable evaluators run |

### Available rule check types

| Type | Value example | Description |
|------|---------------|-------------|
| `not_empty` | — | Output must not be blank |
| `contains` | `"Paris"` | Output must contain substring (case-insensitive) |
| `not_contains` | `"error"` | Output must NOT contain substring |
| `regex_match` | `"\\d+"` | Output must match regex pattern |
| `min_length` | `50` | Output must be ≥ N characters |
| `max_length` | `500` | Output must be ≤ N characters |
| `json_valid` | — | Output must be parseable JSON |
| `starts_with` | `"Here"` | Output must start with substring |
| `ends_with` | `"."` | Output must end with substring |

---

## How to Add New Evaluators

1. Create a new file in `chatbot_qa/evaluators/`:

```python
# chatbot_qa/evaluators/my_evaluator.py
from chatbot_qa.evaluators.base_evaluator import BaseEvaluator
from chatbot_qa.models import EvaluationResult, TestCase

class MyEvaluator(BaseEvaluator):
    def applies_to(self, case: TestCase) -> bool:
        # Return True for cases this evaluator should run on
        return "my-tag" in case.tags

    def evaluate(self, case: TestCase, output: str) -> EvaluationResult:
        passed = "expected phrase" in output.lower()
        return EvaluationResult(
            passed=passed,
            failure_reasons=[] if passed else ["Expected phrase not found"],
        )
```

2. Register it in `TestRunner._build_evaluators()` in `chatbot_qa/runner/test_runner.py`:

```python
def _build_evaluators(self, extras):
    evaluators = [
        RuleBasedEvaluator(),
        SafetyEvaluator(),
        MyEvaluator(),   # ← add here
    ]
    ...
```

Or inject it at runtime:

```python
from chatbot_qa.evaluators.my_evaluator import MyEvaluator
runner = TestRunner(config, provider, extra_evaluators=[MyEvaluator()])
```

---

## How to Connect a Real Chatbot

### Use an existing provider

Set the required environment variables (see [Quick Start](#quick-start)).

### Add a custom provider

```python
# chatbot_qa/runner/providers/my_provider.py
from chatbot_qa.runner.providers.base_provider import ChatbotProvider, ChatbotProviderError
from chatbot_qa.models import Message
from typing import Optional

class MyProvider(ChatbotProvider):
    def __init__(self, api_key: str, **kwargs):
        self._client = MyClient(api_key=api_key)

    @property
    def name(self) -> str:
        return "my-chatbot"

    def chat(self, user_message: str, history: list[Message], system_prompt: Optional[str] = None) -> str:
        try:
            return self._client.send(user_message)
        except Exception as e:
            raise ChatbotProviderError(str(e)) from e
```

Then register it in `chatbot_qa/runner/providers/base_provider.py`:

```python
providers = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "my-chatbot": MyProvider,  # ← add here
}
```

Use it with: `CHATBOT_PROVIDER=my-chatbot python run_evaluation.py`

---

## How to Interpret Results

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | All tests passed |
| `1` | Some non-critical tests failed |
| `2` | Critical failures detected — block deployment |

### JSON report structure

```json
{
  "run_id": "abc12345",
  "provider": "openai",
  "model": "gpt-4o",
  "metrics": {
    "total": 27,
    "passed": 24,
    "failed": 3,
    "pass_rate": 0.889,
    "avg_score": 0.82,
    "avg_latency_ms": 1234,
    "critical_failures": 0,
    "by_category": { ... }
  },
  "results": [...]
}
```

### Markdown report sections

| Section | Content |
|---------|---------|
| Overview | Pass/fail summary table with all metrics |
| Results by Category | Per-category breakdown |
| Failed Tests | Detailed view of each failure with input, output, and reasons |
| Recommendations | Automated action items based on failure patterns |

### Comparing versions

To track regressions over time, save reports by run ID and diff the JSON:

```bash
# Run before model update
python run_evaluation.py --output-dir reports/v1

# Run after model update
python run_evaluation.py --output-dir reports/v2

# Diff (example — adapt to your tooling)
diff <(jq '.metrics' reports/v1/*.json) <(jq '.metrics' reports/v2/*.json)
```

---

## CI/CD Integration

### GitHub Actions example

```yaml
name: Chatbot QA

on:
  pull_request:
  schedule:
    - cron: "0 8 * * *"  # Daily at 08:00 UTC

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt openai

      - name: Run QA evaluation
        env:
          CHATBOT_PROVIDER: openai
          CHATBOT_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CHATBOT_MODEL: gpt-4o
          LLM_JUDGE_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python run_evaluation.py

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: qa-reports
          path: reports/
```

The pipeline will fail (`exit 1` or `exit 2`) automatically if tests fail, blocking the PR.

---

## Configuration Reference

All settings can be overridden via environment variables (higher priority than `config.yaml`).

| YAML key | Env variable | Default | Description |
|----------|-------------|---------|-------------|
| `provider.name` | `CHATBOT_PROVIDER` | `mock` | Provider name |
| `provider.model` | `CHATBOT_MODEL` | `null` | Model identifier |
| `provider.api_key` | `CHATBOT_API_KEY` | `null` | API key |
| `provider.base_url` | `CHATBOT_BASE_URL` | `null` | Custom API endpoint |
| `provider.timeout_s` | — | `30.0` | Request timeout (seconds) |
| `provider.max_retries` | — | `2` | Retry attempts |
| `llm_judge.enabled` | — | `true` | Enable/disable LLM judge |
| `llm_judge.provider` | `LLM_JUDGE_PROVIDER` | `mock` | Judge provider |
| `llm_judge.model` | `LLM_JUDGE_MODEL` | `null` | Judge model |
| `llm_judge.api_key` | `LLM_JUDGE_API_KEY` | `null` | Judge API key |
| `runner.datasets_dir` | — | `datasets` | Dataset files directory |
| `runner.categories` | — | `[]` (all) | Category filter |
| `runner.fail_fast` | — | `false` | Stop on first failure |
| `report.output_dir` | — | `reports` | Report output directory |
| `report.json_enabled` | — | `true` | Generate JSON report |
| `report.markdown_enabled` | — | `true` | Generate Markdown report |
