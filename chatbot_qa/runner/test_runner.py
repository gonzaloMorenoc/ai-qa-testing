"""
Runner de tests: orquesta la carga, ejecución y evaluación de todos los casos de prueba.

El runner es intencionalmente sin estado entre casos de prueba. Cada caso es
independiente. Para los tests multi-turno, el historial de conversación está
embebido en el propio TestCase — el runner no mantiene estado de sesión externamente.

Arquitectura:
  TestRunner
    ├── carga la lista TestCase desde DatasetLoader
    ├── llama a ChatbotProvider.chat() para cada caso
    ├── pasa (TestCase, respuesta) a cada Evaluador
    └── recopila la lista TestResult → se pasa a MetricsAggregator y Reporters
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import concurrent.futures

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
    Orquesta una ejecución completa de evaluación.

    Uso:
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
    # API pública
    # ------------------------------------------------------------------

    def run(
        self,
        cases: Optional[list[TestCase]] = None,
        run_id: Optional[str] = None,
    ) -> Report:
        """
        Ejecuta todos los casos de prueba y devuelve un Report completamente poblado.

        Args:
            cases:  Casos de prueba opcionales pre-cargados. Si es None, carga desde
                    config.runner.datasets_dir.
            run_id: Identificador opcional para este run. Se genera automáticamente si es None.
        """
        run_id = run_id or str(uuid.uuid4())[:8]
        logger.info("Iniciando ejecución %s con el proveedor '%s'", run_id, self._provider.name)

        if cases is None:
            cases = self._load_cases()

        if not cases:
            logger.warning("No se encontraron casos de prueba — comprueba el directorio de datasets.")

        validation = validate_dataset(cases)
        if not validation.is_valid:
            logger.error("La validación del dataset falló:\n%s", validation.summary())
            if self._config.runner.fail_fast:
                raise ValueError(f"Errores de validación del dataset:\n{validation.summary()}")
        elif validation.warnings:
            logger.warning("Advertencias del dataset:\n%s", validation.summary())

        results: list[TestResult] = []
        max_workers = self._config.runner.max_workers

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_case = {executor.submit(self._run_single, case): case for case in cases}

            for i, future in enumerate(concurrent.futures.as_completed(future_to_case), start=1):
                case = future_to_case[future]
                try:
                    result = future.result()
                    results.append(result)

                    status_icon = "✓" if result.passed else "✗"
                    logger.info(
                        "  [%d/%d] %s %s — status=%s duration=%.0fms",
                        i,
                        len(cases),
                        status_icon,
                        case.id,
                        result.status.value,
                        result.duration_ms,
                    )

                    if self._config.runner.fail_fast and result.status == TestStatus.FAILED:
                        logger.warning("Fail-fast activado en el caso %s. Cancelando el resto...", case.id)
                        for pending_future in future_to_case:
                            if not pending_future.done():
                                pending_future.cancel()
                        break

                except Exception as exc:
                    logger.exception("Error fatal inesperado durante la ejecución del caso %s: %s", case.id, exc)

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
            "Ejecución %s completada — %d/%d aprobados (%.1f%%)",
            run_id,
            metrics.passed,
            metrics.total,
            metrics.pass_rate * 100,
        )
        return report

    # ------------------------------------------------------------------
    # Métodos auxiliares internos
    # ------------------------------------------------------------------

    def _load_cases(self) -> list[TestCase]:
        datasets_dir = Path(self._config.runner.datasets_dir)
        categories = self._config.runner.categories or None
        return load_all_datasets(datasets_dir, categories=categories)

    def _run_single(self, case: TestCase) -> TestResult:
        """Ejecuta un caso de prueba: invoca el proveedor y luego evalúa la respuesta."""
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
            status = TestStatus.PASSED  # Provisional — los evaluadores pueden rebajar esto
        except ChatbotProviderError as e:
            error = str(e)
            logger.warning("Error del proveedor para %s: %s", case.id, e)
        except Exception as e:
            error = f"Error inesperado: {e}"
            logger.exception("Error inesperado ejecutando el caso %s", case.id)

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
        """Ejecuta todos los evaluadores aplicables y fusiona los resultados."""
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
        """Instancia la pila de evaluadores por defecto."""
        evaluators: list[BaseEvaluator] = [
            RuleBasedEvaluator(),
            SafetyEvaluator(),
        ]

        # El juez LLM solo se añade si está habilitado en la configuración
        if self._config.llm_judge.enabled:
            from chatbot_qa.evaluators.llm_judge import LLMJudgeEvaluator
            evaluators.append(LLMJudgeEvaluator(self._config.llm_judge))

        evaluators.extend(extras)
        return evaluators
