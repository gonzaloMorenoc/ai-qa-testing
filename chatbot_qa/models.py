"""
Core data models for the chatbot QA framework.

All models use Pydantic v2 for runtime validation, serialization and
IDE auto-completion support. These are the shared contracts between
runner, evaluators, metrics and reporters.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Category(str, Enum):
    FUNCTIONAL = "functional"
    MULTI_TURN = "multi_turn"
    SAFETY = "safety"
    REGRESSION = "regression"


class EvaluationType(str, Enum):
    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"
    SAFETY = "safety"
    COMBINED = "combined"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Test case definition
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """A single message in a conversation."""
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class RuleCheck(BaseModel):
    """Configuration for a single rule-based check."""
    type: str = Field(
        ...,
        description=(
            "One of: contains, not_contains, regex_match, min_length, "
            "max_length, not_empty, json_valid, starts_with, ends_with"
        ),
    )
    value: Optional[Any] = None
    description: Optional[str] = None


class LLMJudgeCriteria(BaseModel):
    """Rubric passed to the LLM-as-judge evaluator."""
    relevance: bool = True
    correctness: bool = True
    clarity: bool = True
    safety: bool = True
    instruction_following: bool = True
    custom_criteria: list[str] = Field(default_factory=list)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)


class SafetyCheckConfig(BaseModel):
    """Configuration for safety checks on a test case."""
    check_prompt_injection: bool = False
    check_system_prompt_leak: bool = False
    check_unsafe_content: bool = False
    should_refuse: bool = False


class TestCase(BaseModel):
    """
    A single test case to be executed against the chatbot.

    Supports both single-turn (input only) and multi-turn (history + input)
    conversations. All evaluation configuration is embedded in the case so
    that the runner and evaluators are fully data-driven.
    """
    id: str
    category: Category
    description: Optional[str] = None

    # Conversation
    input: str = Field(..., description="The final user message to send")
    conversation_history: list[Message] = Field(
        default_factory=list,
        description="Prior turns for multi-turn tests (role/content pairs)",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt to prepend to the conversation",
    )

    # Expected behaviour description (human-readable, used as LLM judge context)
    expected_behavior: str

    # Evaluation configuration
    evaluation_type: EvaluationType
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    llm_judge_criteria: Optional[LLMJudgeCriteria] = None
    safety_config: Optional[SafetyCheckConfig] = None

    # Metadata
    severity: Severity = Severity.MEDIUM
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Evaluation results
# ---------------------------------------------------------------------------

class RuleCheckResult(BaseModel):
    rule_type: str
    passed: bool
    message: str
    value: Optional[Any] = None


class LLMJudgeScore(BaseModel):
    relevance: Optional[float] = None
    correctness: Optional[float] = None
    clarity: Optional[float] = None
    safety: Optional[float] = None
    instruction_following: Optional[float] = None
    custom_scores: dict[str, float] = Field(default_factory=dict)
    overall: float
    reasoning: str


class SafetyCheckResult(BaseModel):
    prompt_injection_detected: bool = False
    system_prompt_leaked: bool = False
    unsafe_content_detected: bool = False
    refused_when_required: Optional[bool] = None
    refused_when_not_required: bool = False
    details: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Aggregated evaluation outcome for a single test case execution."""
    passed: bool
    score: Optional[float] = None
    rule_results: list[RuleCheckResult] = Field(default_factory=list)
    llm_judge: Optional[LLMJudgeScore] = None
    safety: Optional[SafetyCheckResult] = None
    failure_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Test execution result
# ---------------------------------------------------------------------------

class TestResult(BaseModel):
    """Full record of a single test case execution."""
    test_case: TestCase
    chatbot_output: Optional[str] = None
    duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: TestStatus
    evaluation: Optional[EvaluationResult] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

class CategoryStats(BaseModel):
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    avg_score: Optional[float] = None
    avg_latency_ms: float


class Metrics(BaseModel):
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    avg_score: Optional[float] = None
    avg_latency_ms: float
    critical_failures: int
    by_category: dict[str, CategoryStats]


class Report(BaseModel):
    """Top-level evaluation report produced after a full test run."""
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider: str
    model: Optional[str] = None
    metrics: Metrics
    results: list[TestResult]
    version: str = "0.1.0"
