"""Integration tests for the test runner with mock provider."""

import pytest

from chatbot_qa.config import Config, LLMJudgeConfig, RunnerConfig
from chatbot_qa.models import (
    Category,
    EvaluationType,
    Severity,
    TestCase,
    TestStatus,
)
from chatbot_qa.runner.providers.mock_provider import MockProvider
from chatbot_qa.runner.test_runner import TestRunner


def make_functional_case(case_id: str = "func-001", **overrides) -> TestCase:
    data = {
        "id": case_id,
        "category": "functional",
        "input": "Hello",
        "expected_behavior": "Should greet",
        "evaluation_type": "rule_based",
        "rule_checks": [{"type": "not_empty"}],
        **overrides,
    }
    return TestCase.model_validate(data)


def make_config(llm_judge_enabled: bool = False) -> Config:
    config = Config()
    config.llm_judge.enabled = llm_judge_enabled
    config.llm_judge.provider = "mock"
    return config


class TestTestRunner:
    def test_run_with_mock_provider(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        cases = [make_functional_case()]
        report = runner.run(cases=cases)

        assert report.metrics.total == 1
        assert report.provider == "mock"

    def test_all_cases_executed(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        cases = [make_functional_case(f"test-{i:03d}") for i in range(5)]
        report = runner.run(cases=cases)

        assert report.metrics.total == 5
        assert len(report.results) == 5

    def test_passing_case_produces_passed_status(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        cases = [make_functional_case(rule_checks=[{"type": "not_empty"}])]
        report = runner.run(cases=cases)

        assert report.results[0].status == TestStatus.PASSED

    def test_failing_rule_produces_failed_status(self):
        config = make_config()
        # Provider returns "Hello! How can I assist..." — should not contain "xyz_not_present"
        provider = MockProvider()
        runner = TestRunner(config, provider)
        cases = [
            make_functional_case(
                rule_checks=[{"type": "contains", "value": "xyz_definitely_not_in_response"}]
            )
        ]
        report = runner.run(cases=cases)

        assert report.results[0].status == TestStatus.FAILED
        assert report.metrics.failed == 1

    def test_empty_cases_produces_empty_report(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        report = runner.run(cases=[])

        assert report.metrics.total == 0
        assert report.metrics.passed == 0
        assert report.metrics.pass_rate == 0.0

    def test_metrics_by_category(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)

        cases = [
            make_functional_case("f-001", category="functional"),
            make_functional_case("f-002", category="functional"),
            make_functional_case("s-001", category="safety", evaluation_type="safety"),
        ]
        report = runner.run(cases=cases)

        assert "functional" in report.metrics.by_category
        assert "safety" in report.metrics.by_category
        assert report.metrics.by_category["functional"].total == 2

    def test_report_run_id_is_set(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        report = runner.run(cases=[], run_id="my-test-run")

        assert report.run_id == "my-test-run"

    def test_provider_name_in_report(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        report = runner.run(cases=[])

        assert report.provider == "mock"
        assert report.model == "mock-v1"

    def test_fail_fast_stops_after_first_failure(self):
        config = make_config()
        config.runner.fail_fast = True
        provider = MockProvider()
        runner = TestRunner(config, provider)

        # All cases fail (looking for content that won't be in response)
        cases = [
            make_functional_case(
                f"test-{i:03d}",
                rule_checks=[{"type": "contains", "value": "xyz_not_present"}],
            )
            for i in range(5)
        ]
        report = runner.run(cases=cases)

        # Should stop after first failure
        assert len(report.results) < 5

    def test_duration_recorded(self):
        config = make_config()
        provider = MockProvider()
        runner = TestRunner(config, provider)
        cases = [make_functional_case()]
        report = runner.run(cases=cases)

        assert report.results[0].duration_ms >= 0

    def test_with_llm_judge_enabled(self):
        config = make_config(llm_judge_enabled=True)
        provider = MockProvider()
        runner = TestRunner(config, provider)

        cases = [
            make_functional_case(
                evaluation_type="llm_judge",
                rule_checks=[],
                llm_judge_criteria={
                    "relevance": True,
                    "correctness": True,
                    "clarity": True,
                    "safety": False,
                    "instruction_following": True,
                    "min_score": 0.1,  # Low threshold so mock judge passes
                },
            )
        ]
        report = runner.run(cases=cases)

        assert report.metrics.total == 1
        result = report.results[0]
        # LLM judge should have been evaluated
        assert result.evaluation is not None
