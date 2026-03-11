"""
Evaluador por reglas.

Realiza comprobaciones deterministas y rápidas sobre la salida del chatbot
sin ninguna llamada a API externa. Se ejecuta en todo caso de prueba que
tenga rule_checks definidos.

Tipos de regla soportados:
  - not_empty        La salida no debe estar vacía ni ser solo espacios
  - contains         La salida debe contener una subcadena literal
  - not_contains     La salida NO debe contener una subcadena literal
  - regex_match      La salida debe coincidir con un patrón regex
  - min_length       La salida debe tener al menos N caracteres
  - max_length       La salida debe tener como máximo N caracteres
  - json_valid       La salida debe ser JSON válido
  - starts_with      La salida debe comenzar con una subcadena
  - ends_with        La salida debe terminar con una subcadena
"""

from __future__ import annotations

import json
import re

from chatbot_qa.evaluators.base_evaluator import BaseEvaluator
from chatbot_qa.models import (
    EvaluationResult,
    EvaluationType,
    RuleCheck,
    RuleCheckResult,
    TestCase,
)


class RuleBasedEvaluator(BaseEvaluator):
    """Ejecuta todas las rule_checks configuradas para un caso de prueba."""

    def applies_to(self, case: TestCase) -> bool:
        return bool(case.rule_checks) or case.evaluation_type in (
            EvaluationType.RULE_BASED,
            EvaluationType.COMBINED,
        )

    def evaluate(self, case: TestCase, output: str) -> EvaluationResult:
        results: list[RuleCheckResult] = []
        failure_reasons: list[str] = []

        # Siempre comprobar que la salida no esté vacía como base
        empty_result = self._check_not_empty(output)
        results.append(empty_result)
        if not empty_result.passed:
            failure_reasons.append(empty_result.message)
            # No tiene sentido ejecutar más comprobaciones sobre una salida vacía
            return EvaluationResult(
                passed=False,
                rule_results=results,
                failure_reasons=failure_reasons,
            )

        for rule in case.rule_checks:
            result = self._apply_rule(rule, output)
            results.append(result)
            if not result.passed:
                failure_reasons.append(result.message)

        return EvaluationResult(
            passed=len(failure_reasons) == 0,
            rule_results=results,
            failure_reasons=failure_reasons,
        )

    # ------------------------------------------------------------------
    # Despacho de reglas
    # ------------------------------------------------------------------

    def _apply_rule(self, rule: RuleCheck, output: str) -> RuleCheckResult:
        handlers = {
            "not_empty": lambda r, o: self._check_not_empty(o),
            "contains": self._check_contains,
            "not_contains": self._check_not_contains,
            "regex_match": self._check_regex,
            "min_length": self._check_min_length,
            "max_length": self._check_max_length,
            "json_valid": lambda r, o: self._check_json_valid(o),
            "starts_with": self._check_starts_with,
            "ends_with": self._check_ends_with,
        }
        handler = handlers.get(rule.type)
        if handler is None:
            return RuleCheckResult(
                rule_type=rule.type,
                passed=False,
                message=f"Tipo de regla desconocido: '{rule.type}'",
            )
        return handler(rule, output)

    # ------------------------------------------------------------------
    # Implementaciones de comprobaciones individuales
    # ------------------------------------------------------------------

    @staticmethod
    def _check_not_empty(output: str) -> RuleCheckResult:
        passed = bool(output and output.strip())
        return RuleCheckResult(
            rule_type="not_empty",
            passed=passed,
            message="Output is not empty" if passed else "Output is empty or whitespace-only",
        )

    @staticmethod
    def _check_contains(rule: RuleCheck, output: str) -> RuleCheckResult:
        value = str(rule.value or "")
        passed = value.lower() in output.lower()
        return RuleCheckResult(
            rule_type="contains",
            passed=passed,
            value=value,
            message=(
                f"Output contains '{value}'"
                if passed
                else f"Output does not contain expected phrase: '{value}'"
            ),
        )

    @staticmethod
    def _check_not_contains(rule: RuleCheck, output: str) -> RuleCheckResult:
        value = str(rule.value or "")
        passed = value.lower() not in output.lower()
        return RuleCheckResult(
            rule_type="not_contains",
            passed=passed,
            value=value,
            message=(
                f"Output correctly does not contain '{value}'"
                if passed
                else f"Output contains forbidden phrase: '{value}'"
            ),
        )

    @staticmethod
    def _check_regex(rule: RuleCheck, output: str) -> RuleCheckResult:
        pattern = str(rule.value or "")
        try:
            match = bool(re.search(pattern, output, re.IGNORECASE | re.DOTALL))
        except re.error as e:
            return RuleCheckResult(
                rule_type="regex_match",
                passed=False,
                value=pattern,
                message=f"Invalid regex pattern '{pattern}': {e}",
            )
        return RuleCheckResult(
            rule_type="regex_match",
            passed=match,
            value=pattern,
            message=(
                f"Output matches pattern '{pattern}'"
                if match
                else f"Output does not match pattern: '{pattern}'"
            ),
        )

    @staticmethod
    def _check_min_length(rule: RuleCheck, output: str) -> RuleCheckResult:
        min_len = int(rule.value or 0)
        actual = len(output.strip())
        passed = actual >= min_len
        return RuleCheckResult(
            rule_type="min_length",
            passed=passed,
            value=min_len,
            message=(
                f"Output length {actual} meets minimum {min_len}"
                if passed
                else f"Output too short: {actual} chars (minimum {min_len})"
            ),
        )

    @staticmethod
    def _check_max_length(rule: RuleCheck, output: str) -> RuleCheckResult:
        max_len = int(rule.value or 0)
        actual = len(output.strip())
        passed = actual <= max_len
        return RuleCheckResult(
            rule_type="max_length",
            passed=passed,
            value=max_len,
            message=(
                f"Output length {actual} within maximum {max_len}"
                if passed
                else f"Output too long: {actual} chars (maximum {max_len})"
            ),
        )

    @staticmethod
    def _check_json_valid(output: str) -> RuleCheckResult:
        try:
            json.loads(output.strip())
            passed = True
            message = "Output is valid JSON"
        except json.JSONDecodeError as e:
            passed = False
            message = f"Output is not valid JSON: {e}"
        return RuleCheckResult(rule_type="json_valid", passed=passed, message=message)

    @staticmethod
    def _check_starts_with(rule: RuleCheck, output: str) -> RuleCheckResult:
        value = str(rule.value or "")
        passed = output.strip().startswith(value)
        return RuleCheckResult(
            rule_type="starts_with",
            passed=passed,
            value=value,
            message=(
                f"Output starts with '{value}'"
                if passed
                else f"Output does not start with '{value}'"
            ),
        )

    @staticmethod
    def _check_ends_with(rule: RuleCheck, output: str) -> RuleCheckResult:
        value = str(rule.value or "")
        passed = output.strip().endswith(value)
        return RuleCheckResult(
            rule_type="ends_with",
            passed=passed,
            value=value,
            message=(
                f"Output ends with '{value}'"
                if passed
                else f"Output does not end with '{value}'"
            ),
        )
