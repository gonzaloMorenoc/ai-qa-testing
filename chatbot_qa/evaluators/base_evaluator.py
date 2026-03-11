"""
Clase base abstracta para todos los evaluadores.

Para añadir un nuevo evaluador:
  1. Heredar de BaseEvaluator
  2. Implementar `applies_to` — devuelve True para los casos que maneja este evaluador
  3. Implementar `evaluate` — devuelve un EvaluationResult parcial
  4. Registrar el evaluador en TestRunner._build_evaluators()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chatbot_qa.models import EvaluationResult, TestCase


class BaseEvaluator(ABC):
    """Contrato que todos los evaluadores deben cumplir."""

    @abstractmethod
    def applies_to(self, case: TestCase) -> bool:
        """Devuelve True si este evaluador debe ejecutarse para el caso de prueba dado."""

    @abstractmethod
    def evaluate(self, case: TestCase, output: str) -> EvaluationResult:
        """
        Evalúa la salida del chatbot contra las expectativas del caso de prueba.

        Devuelve un EvaluationResult *parcial* — solo deben poblarse los campos
        de los que este evaluador es responsable. El runner fusiona todos los
        resultados parciales en un EvaluationResult final.
        """
