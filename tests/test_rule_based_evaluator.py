"""Unit tests for the rule-based evaluator."""

import pytest

from chatbot_qa.evaluators.rule_based import RuleBasedEvaluator
from chatbot_qa.models import EvaluationType, RuleCheck, TestCase


def make_case(rules: list[dict], evaluation_type: str = "rule_based") -> TestCase:
    return TestCase.model_validate(
        {
            "id": "rule-test",
            "category": "functional",
            "input": "Hello",
            "expected_behavior": "Should greet",
            "evaluation_type": evaluation_type,
            "rule_checks": rules,
        }
    )


class TestRuleBasedEvaluator:
    evaluator = RuleBasedEvaluator()

    # -- applies_to -----------------------------------------------------------
    def test_applies_to_rule_based(self):
        case = make_case([], evaluation_type="rule_based")
        assert self.evaluator.applies_to(case) is True

    def test_applies_to_combined(self):
        case = make_case([], evaluation_type="combined")
        assert self.evaluator.applies_to(case) is True

    def test_does_not_apply_to_llm_judge_without_rules(self):
        case = TestCase.model_validate(
            {
                "id": "x",
                "category": "functional",
                "input": "hi",
                "expected_behavior": "greet",
                "evaluation_type": "llm_judge",
                "rule_checks": [],
            }
        )
        assert self.evaluator.applies_to(case) is False

    # -- not_empty ------------------------------------------------------------
    def test_empty_output_fails(self):
        case = make_case([])
        result = self.evaluator.evaluate(case, "")
        assert result.passed is False
        assert any("empty" in r.message.lower() for r in result.rule_results)

    def test_whitespace_only_fails(self):
        case = make_case([])
        result = self.evaluator.evaluate(case, "   \n  ")
        assert result.passed is False

    def test_non_empty_passes(self):
        case = make_case([])
        result = self.evaluator.evaluate(case, "Hello there!")
        assert result.passed is True

    # -- contains -------------------------------------------------------------
    def test_contains_pass(self):
        case = make_case([{"type": "contains", "value": "hello"}])
        result = self.evaluator.evaluate(case, "Hello World")
        assert result.passed is True

    def test_contains_case_insensitive(self):
        case = make_case([{"type": "contains", "value": "HELLO"}])
        result = self.evaluator.evaluate(case, "hello world")
        assert result.passed is True

    def test_contains_fail(self):
        case = make_case([{"type": "contains", "value": "goodbye"}])
        result = self.evaluator.evaluate(case, "Hello World")
        assert result.passed is False

    # -- not_contains ---------------------------------------------------------
    def test_not_contains_pass(self):
        case = make_case([{"type": "not_contains", "value": "forbidden"}])
        result = self.evaluator.evaluate(case, "This is fine")
        assert result.passed is True

    def test_not_contains_fail(self):
        case = make_case([{"type": "not_contains", "value": "forbidden"}])
        result = self.evaluator.evaluate(case, "This contains forbidden content")
        assert result.passed is False

    # -- min_length -----------------------------------------------------------
    def test_min_length_pass(self):
        case = make_case([{"type": "min_length", "value": 5}])
        result = self.evaluator.evaluate(case, "Hello world")
        assert result.passed is True

    def test_min_length_fail(self):
        case = make_case([{"type": "min_length", "value": 100}])
        result = self.evaluator.evaluate(case, "Short")
        assert result.passed is False

    # -- max_length -----------------------------------------------------------
    def test_max_length_pass(self):
        case = make_case([{"type": "max_length", "value": 100}])
        result = self.evaluator.evaluate(case, "Short response")
        assert result.passed is True

    def test_max_length_fail(self):
        case = make_case([{"type": "max_length", "value": 5}])
        result = self.evaluator.evaluate(case, "This is a very long response")
        assert result.passed is False

    # -- json_valid -----------------------------------------------------------
    def test_json_valid_pass(self):
        case = make_case([{"type": "json_valid"}])
        result = self.evaluator.evaluate(case, '{"key": "value"}')
        assert result.passed is True

    def test_json_valid_fail(self):
        case = make_case([{"type": "json_valid"}])
        result = self.evaluator.evaluate(case, "This is not JSON")
        assert result.passed is False

    # -- regex_match ----------------------------------------------------------
    def test_regex_match_pass(self):
        case = make_case([{"type": "regex_match", "value": r"\d+"}])
        result = self.evaluator.evaluate(case, "The answer is 42")
        assert result.passed is True

    def test_regex_match_fail(self):
        case = make_case([{"type": "regex_match", "value": r"\d+"}])
        result = self.evaluator.evaluate(case, "No numbers here")
        assert result.passed is False

    def test_invalid_regex_fails_gracefully(self):
        case = make_case([{"type": "regex_match", "value": "[invalid"}])
        result = self.evaluator.evaluate(case, "anything")
        assert result.passed is False
        assert any("invalid" in r.message.lower() for r in result.rule_results)

    # -- starts_with / ends_with ----------------------------------------------
    def test_starts_with_pass(self):
        case = make_case([{"type": "starts_with", "value": "Hello"}])
        result = self.evaluator.evaluate(case, "Hello World")
        assert result.passed is True

    def test_ends_with_pass(self):
        case = make_case([{"type": "ends_with", "value": "World"}])
        result = self.evaluator.evaluate(case, "Hello World")
        assert result.passed is True

    # -- multiple rules -------------------------------------------------------
    def test_multiple_rules_all_pass(self):
        case = make_case([
            {"type": "contains", "value": "hello"},
            {"type": "min_length", "value": 5},
            {"type": "max_length", "value": 100},
        ])
        result = self.evaluator.evaluate(case, "Hello World")
        assert result.passed is True
        assert len(result.failure_reasons) == 0

    def test_multiple_rules_one_fails(self):
        case = make_case([
            {"type": "contains", "value": "hello"},
            {"type": "contains", "value": "goodbye"},  # Will fail
        ])
        result = self.evaluator.evaluate(case, "Hello World")
        assert result.passed is False
        assert len(result.failure_reasons) == 1

    # -- unknown rule type ----------------------------------------------------
    def test_unknown_rule_type_fails(self):
        case = make_case([{"type": "nonexistent_rule"}])
        result = self.evaluator.evaluate(case, "Hello")
        assert result.passed is False
