"""Unit tests for core data models."""

import pytest
from pydantic import ValidationError

from chatbot_qa.models import (
    Category,
    EvaluationType,
    LLMJudgeCriteria,
    Message,
    RuleCheck,
    SafetyCheckConfig,
    Severity,
    TestCase,
    TestStatus,
)


def make_minimal_case(**overrides) -> dict:
    """Return a minimal valid TestCase dict."""
    base = {
        "id": "test-001",
        "category": "functional",
        "input": "Hello",
        "expected_behavior": "Should greet the user",
        "evaluation_type": "rule_based",
        "severity": "medium",
    }
    base.update(overrides)
    return base


class TestTestCase:
    def test_minimal_valid_case(self):
        tc = TestCase.model_validate(make_minimal_case())
        assert tc.id == "test-001"
        assert tc.category == Category.FUNCTIONAL
        assert tc.severity == Severity.MEDIUM

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            TestCase.model_validate({"id": "x", "category": "functional"})

    def test_invalid_category(self):
        with pytest.raises(ValidationError):
            TestCase.model_validate(make_minimal_case(category="invalid_cat"))

    def test_conversation_history(self):
        tc = TestCase.model_validate(
            make_minimal_case(
                category="multi_turn",
                conversation_history=[
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                ],
            )
        )
        assert len(tc.conversation_history) == 2
        assert tc.conversation_history[0].role == "user"

    def test_rule_checks_populated(self):
        tc = TestCase.model_validate(
            make_minimal_case(
                rule_checks=[
                    {"type": "contains", "value": "hello"},
                    {"type": "min_length", "value": 10},
                ]
            )
        )
        assert len(tc.rule_checks) == 2
        assert tc.rule_checks[0].type == "contains"

    def test_llm_judge_criteria_defaults(self):
        criteria = LLMJudgeCriteria()
        assert criteria.relevance is True
        assert criteria.min_score == 0.6

    def test_safety_config_defaults(self):
        cfg = SafetyCheckConfig()
        assert cfg.check_prompt_injection is False
        assert cfg.should_refuse is False


class TestEnums:
    def test_category_values(self):
        assert Category.FUNCTIONAL.value == "functional"
        assert Category.MULTI_TURN.value == "multi_turn"
        assert Category.SAFETY.value == "safety"
        assert Category.REGRESSION.value == "regression"

    def test_severity_ordering(self):
        # Just verify all values exist
        assert {s.value for s in Severity} == {"low", "medium", "high", "critical"}

    def test_test_status_values(self):
        assert {s.value for s in TestStatus} == {"passed", "failed", "error", "skipped"}
