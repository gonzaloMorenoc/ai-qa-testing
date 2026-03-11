"""Unit tests for JSON and Markdown reporters."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from chatbot_qa.metrics.aggregator import compute_metrics
from chatbot_qa.models import (
    CategoryStats,
    EvaluationResult,
    Metrics,
    Report,
    TestCase,
    TestResult,
    TestStatus,
)
from chatbot_qa.reports.json_reporter import write_json_report
from chatbot_qa.reports.markdown_reporter import write_markdown_report


def make_minimal_report(results: list = None) -> Report:
    if results is None:
        results = []
    metrics = compute_metrics(results)
    return Report(
        run_id="test-run-001",
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
        provider="mock",
        model="mock-v1",
        metrics=metrics,
        results=results,
    )


def make_test_result(
    case_id: str, status: TestStatus, failure_reasons: list = None
) -> TestResult:
    case = TestCase.model_validate(
        {
            "id": case_id,
            "category": "functional",
            "input": "Test input",
            "expected_behavior": "Should work",
            "evaluation_type": "rule_based",
        }
    )
    evaluation = EvaluationResult(
        passed=status == TestStatus.PASSED,
        failure_reasons=failure_reasons or [],
    )
    return TestResult(
        test_case=case,
        chatbot_output="Some output" if status != TestStatus.ERROR else None,
        duration_ms=150.0,
        status=status,
        evaluation=evaluation,
    )


class TestJSONReporter:
    def test_creates_file(self, tmp_path):
        report = make_minimal_report()
        path = write_json_report(report, tmp_path)
        assert path.exists()
        assert path.suffix == ".json"

    def test_file_name_contains_run_id(self, tmp_path):
        report = make_minimal_report()
        path = write_json_report(report, tmp_path)
        assert "test-run-001" in path.name

    def test_valid_json_output(self, tmp_path):
        report = make_minimal_report()
        path = write_json_report(report, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["run_id"] == "test-run-001"
        assert data["provider"] == "mock"

    def test_metrics_in_json(self, tmp_path):
        results = [make_test_result("t1", TestStatus.PASSED)]
        report = make_minimal_report(results)
        path = write_json_report(report, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["metrics"]["total"] == 1
        assert data["metrics"]["passed"] == 1

    def test_creates_output_dir(self, tmp_path):
        nested_dir = tmp_path / "nested" / "reports"
        report = make_minimal_report()
        path = write_json_report(report, nested_dir)
        assert path.exists()


class TestMarkdownReporter:
    def test_creates_file(self, tmp_path):
        report = make_minimal_report()
        path = write_markdown_report(report, tmp_path)
        assert path.exists()
        assert path.suffix == ".md"

    def test_file_name_contains_run_id(self, tmp_path):
        report = make_minimal_report()
        path = write_markdown_report(report, tmp_path)
        assert "test-run-001" in path.name

    def test_contains_run_id(self, tmp_path):
        report = make_minimal_report()
        path = write_markdown_report(report, tmp_path)
        content = path.read_text()
        assert "test-run-001" in content

    def test_contains_provider(self, tmp_path):
        report = make_minimal_report()
        path = write_markdown_report(report, tmp_path)
        content = path.read_text()
        assert "mock" in content

    def test_contains_overview_section(self, tmp_path):
        report = make_minimal_report()
        path = write_markdown_report(report, tmp_path)
        content = path.read_text()
        assert "Overview" in content

    def test_contains_failed_tests_section(self, tmp_path):
        results = [make_test_result("fail-001", TestStatus.FAILED, ["Rule check failed"])]
        report = make_minimal_report(results)
        path = write_markdown_report(report, tmp_path)
        content = path.read_text()
        assert "Failed Tests" in content
        assert "fail-001" in content

    def test_no_failures_message(self, tmp_path):
        results = [make_test_result("pass-001", TestStatus.PASSED)]
        report = make_minimal_report(results)
        path = write_markdown_report(report, tmp_path)
        content = path.read_text()
        assert "No failures" in content

    def test_contains_recommendations(self, tmp_path):
        report = make_minimal_report()
        path = write_markdown_report(report, tmp_path)
        content = path.read_text()
        assert "Recommendations" in content
