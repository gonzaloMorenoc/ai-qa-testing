"""
Abstract base class for all evaluators.

To add a new evaluator:
  1. Subclass BaseEvaluator
  2. Implement `applies_to` — return True for the cases this evaluator handles
  3. Implement `evaluate` — return a partial EvaluationResult
  4. Register the evaluator in TestRunner._build_evaluators()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chatbot_qa.models import EvaluationResult, TestCase


class BaseEvaluator(ABC):
    """Contract that all evaluators must satisfy."""

    @abstractmethod
    def applies_to(self, case: TestCase) -> bool:
        """Return True if this evaluator should run for the given test case."""

    @abstractmethod
    def evaluate(self, case: TestCase, output: str) -> EvaluationResult:
        """
        Evaluate the chatbot output against the test case expectations.

        Returns a *partial* EvaluationResult — only the fields this evaluator
        is responsible for need to be populated. The runner merges all partial
        results into a final EvaluationResult.
        """
