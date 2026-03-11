"""Unit tests for the metrics aggregator."""

from datetime import datetime

import pytest

from chatbot_qa.metrics.aggregator import compute_metrics
from chatbot_qa.models import (
    Category,
    EvaluationResult,
    EvaluationType,
    Severity,
    TestCase,
    TestResult,
    TestStatus,
)


def make_result(
    case_id: str = "test-001",
    status: TestStatus = TestStatus.PASSED,
    category: str = "functional",
    severity: str = "medium",
    duration_ms: float = 100.0,
    score: float = None,
) -> TestResult:
    case = TestCase.model_validate(
        {
            "id": case_id,
            "category": category,
            "input": "test",
            "expected_behavior": "test",
            "evaluation_type": "rule_based",
            "severity": severity,
        }
    )
    evaluation = EvaluationResult(
        passed=status == TestStatus.PASSED,
        score=score,
    ) if status != TestStatus.ERROR else None

    return TestResult(
        test_case=case,
        chatbot_output="response" if status != TestStatus.ERROR else None,
        duration_ms=duration_ms,
        status=status,
        evaluation=evaluation,
        error="Some error" if status == TestStatus.ERROR else None,
    )


class TestComputeMetrics:
    def test_empty_results(self):
        metrics = compute_metrics([])
        assert metrics.total == 0
        assert metrics.passed == 0
        assert metrics.pass_rate == 0.0
        assert metrics.critical_failures == 0
        assert metrics.by_category == {}

    def test_all_passed(self):
        results = [make_result(f"test-{i}", TestStatus.PASSED) for i in range(5)]
        metrics = compute_metrics(results)
        assert metrics.total == 5
        assert metrics.passed == 5
        assert metrics.failed == 0
        assert metrics.pass_rate == 1.0

    def test_all_failed(self):
        results = [make_result(f"test-{i}", TestStatus.FAILED) for i in range(3)]
        metrics = compute_metrics(results)
        assert metrics.total == 3
        assert metrics.passed == 0
        assert metrics.failed == 3
        assert metrics.pass_rate == 0.0

    def test_mixed_results(self):
        results = [
            make_result("t1", TestStatus.PASSED),
            make_result("t2", TestStatus.PASSED),
            make_result("t3", TestStatus.FAILED),
            make_result("t4", TestStatus.ERROR),
        ]
        metrics = compute_metrics(results)
        assert metrics.total == 4
        assert metrics.passed == 2
        assert metrics.failed == 1
        assert metrics.errored == 1
        assert metrics.pass_rate == pytest.approx(0.5)

    def test_critical_failures_counted(self):
        results = [
            make_result("t1", TestStatus.FAILED, severity="critical"),
            make_result("t2", TestStatus.FAILED, severity="high"),
            make_result("t3", TestStatus.FAILED, severity="medium"),
            make_result("t4", TestStatus.PASSED, severity="critical"),  # Passed, not counted
        ]
        metrics = compute_metrics(results)
        assert metrics.critical_failures == 1

    def test_average_latency(self):
        results = [
            make_result("t1", duration_ms=100.0),
            make_result("t2", duration_ms=300.0),
        ]
        metrics = compute_metrics(results)
        assert metrics.avg_latency_ms == pytest.approx(200.0)

    def test_average_score_computed(self):
        results = [
            make_result("t1", TestStatus.PASSED, score=0.8),
            make_result("t2", TestStatus.PASSED, score=0.6),
        ]
        metrics = compute_metrics(results)
        assert metrics.avg_score == pytest.approx(0.7)

    def test_no_score_when_none(self):
        results = [make_result("t1", score=None)]
        metrics = compute_metrics(results)
        assert metrics.avg_score is None

    def test_by_category_breakdown(self):
        results = [
            make_result("f1", TestStatus.PASSED, category="functional"),
            make_result("f2", TestStatus.PASSED, category="functional"),
            make_result("s1", TestStatus.FAILED, category="safety"),
        ]
        metrics = compute_metrics(results)
        assert metrics.by_category["functional"].total == 2
        assert metrics.by_category["functional"].pass_rate == 1.0
        assert metrics.by_category["safety"].total == 1
        assert metrics.by_category["safety"].pass_rate == 0.0

    def test_pass_rate_calculation(self):
        results = [
            make_result("t1", TestStatus.PASSED),
            make_result("t2", TestStatus.PASSED),
            make_result("t3", TestStatus.FAILED),
        ]
        metrics = compute_metrics(results)
        assert metrics.pass_rate == pytest.approx(2 / 3)
