"""
Metrics aggregator.

Computes aggregate statistics from a list of TestResult objects, broken
down by overall totals and per-category subtotals.
"""

from __future__ import annotations

from collections import defaultdict

from chatbot_qa.models import (
    CategoryStats,
    Metrics,
    Severity,
    TestResult,
    TestStatus,
)


def compute_metrics(results: list[TestResult]) -> Metrics:
    """
    Compute aggregate metrics from a completed test run.

    Args:
        results: List of TestResult objects from the runner.

    Returns:
        A Metrics object with overall and per-category statistics.
    """
    if not results:
        return Metrics(
            total=0,
            passed=0,
            failed=0,
            errored=0,
            pass_rate=0.0,
            avg_latency_ms=0.0,
            critical_failures=0,
            by_category={},
        )

    total = len(results)
    passed = sum(1 for r in results if r.status == TestStatus.PASSED)
    failed = sum(1 for r in results if r.status == TestStatus.FAILED)
    errored = sum(1 for r in results if r.status == TestStatus.ERROR)
    pass_rate = passed / total if total > 0 else 0.0

    # Overall average score (only for cases that have a score)
    scored = [r.evaluation.score for r in results if r.evaluation and r.evaluation.score is not None]
    avg_score = sum(scored) / len(scored) if scored else None

    avg_latency_ms = sum(r.duration_ms for r in results) / total

    critical_failures = sum(
        1
        for r in results
        if r.status in (TestStatus.FAILED, TestStatus.ERROR)
        and r.test_case.severity == Severity.CRITICAL
    )

    by_category = _compute_category_stats(results)

    return Metrics(
        total=total,
        passed=passed,
        failed=failed,
        errored=errored,
        pass_rate=pass_rate,
        avg_score=avg_score,
        avg_latency_ms=avg_latency_ms,
        critical_failures=critical_failures,
        by_category=by_category,
    )


def _compute_category_stats(results: list[TestResult]) -> dict[str, CategoryStats]:
    grouped: dict[str, list[TestResult]] = defaultdict(list)
    for r in results:
        grouped[r.test_case.category.value].append(r)

    stats: dict[str, CategoryStats] = {}
    for category, cat_results in grouped.items():
        total = len(cat_results)
        passed = sum(1 for r in cat_results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in cat_results if r.status == TestStatus.FAILED)
        errored = sum(1 for r in cat_results if r.status == TestStatus.ERROR)
        pass_rate = passed / total if total > 0 else 0.0
        avg_latency_ms = sum(r.duration_ms for r in cat_results) / total

        scored = [
            r.evaluation.score
            for r in cat_results
            if r.evaluation and r.evaluation.score is not None
        ]
        avg_score = sum(scored) / len(scored) if scored else None

        stats[category] = CategoryStats(
            total=total,
            passed=passed,
            failed=failed,
            errored=errored,
            pass_rate=pass_rate,
            avg_score=avg_score,
            avg_latency_ms=avg_latency_ms,
        )

    return stats
