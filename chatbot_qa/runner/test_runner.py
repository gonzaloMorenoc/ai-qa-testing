"""
Test runner: orchestrates loading, execution and evaluation of all test cases.

The runner is intentionally stateless between test cases. Each case is
independent. For multi-turn tests, conversation history is embedded in the
TestCase itself — the runner does not maintain session state externally.

Architecture:
  TestRunner
    ├── loads TestCase list from DatasetLoader
    ├── calls ChatbotProvider.chat() for each case
    ├── passes (TestCase, response) to each Evaluator
    └── collects TestResult list → passed to MetricsAggregator and Reporters
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from chatbot_qa.config import Config
from chatbot_qa.datasets.loader import load_all_datasets
from chatbot_qa.datasets.validator import validate_dataset
from chatbot_qa.evaluators.base_evaluator import BaseEvaluator
from chatbot_qa.evaluators.rule_based import RuleBasedEvaluator
from chatbot_qa.evaluators.safety_checks import SafetyEvaluator
from chatbot_qa.metrics.aggregator import compute_metrics
from chatbot_qa.models import (
    EvaluationResult,
    EvaluationType,
    Report,
    TestCase,
    TestResult,
    TestStatus,
)
from chatbot_qa.runner.providers.base_provider import (
    ChatbotProvider,
    ChatbotProviderError,
)

logger = logging.getLogger(__name__)


class TestRunner:
    """
    Orchestrates a full evaluation run.

    Usage:
        runner = TestRunner(config, provider)
        report = runner.run()
    """

    def __init__(
        self,
        config: Config,
        provider: ChatbotProvider,
        extra_evaluators: Optional[list[BaseEvaluator]] = None,
    ) -> None:
        self._config = config
        self._provider = provider
        self._evaluators = self._build_evaluators(extra_evaluators or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        cases: Optional[list[TestCase]] = None,
        run_id: Optional[str] = None,
    ) -> Report:
        """
        Execute all test cases and return a fully populated Report.

        Args:
            cases:  Optional pre-loaded test cases. If None, loads from
                    config.runner.datasets_dir.
            run_id: Optional identifier for this run. Auto-generated if None.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        logger.info("Starting test run %s with provider '%s'", run_id, self._provider.name)

        if cases is None:
            cases = self._load_cases()

        if not cases:
            logger.warning("No test cases found — check your datasets directory.")

        validation = validate_dataset(cases)
        if not validation.is_valid:
            logger.error("Dataset validation failed:\n%s", validation.summary())
            if self._config.runner.fail_fast:
                raise ValueError(f"Dataset validation errors:\n{validation.summary()}")
        elif validation.warnings:
            logger.warning("Dataset warnings:\n%s", validation.summary())

        results: list[TestResult] = []
        for i, case in enumerate(cases, start=1):
            logger.info("[%d/%d] Running: %s (%s)", i, len(cases), case.id, case.category.value)
            result = self._run_single(case)
            results.append(result)

            status_icon = "✓" if result.passed else "✗"
            logger.info(
                "  %s %s — status=%s duration=%.0fms",
                status_icon,
                case.id,
                result.status.value,
                result.duration_ms,
            )

            if self._config.runner.fail_fast and result.status == TestStatus.FAILED:
                logger.warning("Fail-fast triggered at case %s", case.id)
                break

        metrics = compute_metrics(results)
        report = Report(
            run_id=run_id,
            timestamp=datetime.utcnow(),
            provider=self._provider.name,
            model=self._provider.model,
            metrics=metrics,
            results=results,
        )
        logger.info(
            "Run %s complete — %d/%d passed (%.1f%%)",
            run_id,
            metrics.passed,
            metrics.total,
            metrics.pass_rate * 100,
        )
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_cases(self) -> list[TestCase]:
        datasets_dir = Path(self._config.runner.datasets_dir)
        categories = self._config.runner.categories or None
        return load_all_datasets(datasets_dir, categories=categories)

    def _run_single(self, case: TestCase) -> TestResult:
        """Execute one test case: invoke provider, then evaluate the response."""
        start = time.monotonic()
        chatbot_output: Optional[str] = None
        error: Optional[str] = None
        status = TestStatus.ERROR

        try:
            chatbot_output = self._provider.chat(
                user_message=case.input,
                history=case.conversation_history,
                system_prompt=case.system_prompt,
            )
            status = TestStatus.PASSED  # Tentative — evaluators may downgrade
        except ChatbotProviderError as e:
            error = str(e)
            logger.warning("Provider error for %s: %s", case.id, e)
        except Exception as e:
            error = f"Unexpected error: {e}"
            logger.exception("Unexpected error running case %s", case.id)

        duration_ms = (time.monotonic() - start) * 1000

        evaluation: Optional[EvaluationResult] = None
        if chatbot_output is not None:
            evaluation = self._evaluate(case, chatbot_output)
            status = TestStatus.PASSED if evaluation.passed else TestStatus.FAILED
        elif self._config.runner.skip_on_error:
            status = TestStatus.ERROR

        return TestResult(
            test_case=case,
            chatbot_output=chatbot_output,
            duration_ms=duration_ms,
            status=status,
            evaluation=evaluation,
            error=error,
        )

    def _evaluate(self, case: TestCase, output: str) -> EvaluationResult:
        """Run all applicable evaluators and merge results."""
        all_rule_results = []
        llm_judge_result = None
        safety_result = None
        failure_reasons: list[str] = []
        scores: list[float] = []

        for evaluator in self._evaluators:
            if not evaluator.applies_to(case):
                continue

            partial = evaluator.evaluate(case, output)
            all_rule_results.extend(partial.rule_results)

            if partial.llm_judge is not None:
                llm_judge_result = partial.llm_judge
                scores.append(partial.llm_judge.overall)

            if partial.safety is not None:
                safety_result = partial.safety

            if not partial.passed:
                failure_reasons.extend(partial.failure_reasons)

        # Compute overall pass/fail
        overall_passed = len(failure_reasons) == 0
        overall_score = sum(scores) / len(scores) if scores else None

        return EvaluationResult(
            passed=overall_passed,
            score=overall_score,
            rule_results=all_rule_results,
            llm_judge=llm_judge_result,
            safety=safety_result,
            failure_reasons=failure_reasons,
        )

    def _build_evaluators(self, extras: list[BaseEvaluator]) -> list[BaseEvaluator]:
        """Instantiate the default evaluator stack."""
        evaluators: list[BaseEvaluator] = [
            RuleBasedEvaluator(),
            SafetyEvaluator(),
        ]

        # LLM judge is only added if enabled in config
        if self._config.llm_judge.enabled:
            from chatbot_qa.evaluators.llm_judge import LLMJudgeEvaluator
            evaluators.append(LLMJudgeEvaluator(self._config.llm_judge))

        evaluators.extend(extras)
        return evaluators
