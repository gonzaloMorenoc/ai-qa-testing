"""
Utilidades de validación de datasets.

Comprueba errores comunes de autoría antes de ejecutar un run, proporcionando
mensajes de error claros para que los problemas puedan corregirse antes de
consumir créditos de API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chatbot_qa.models import EvaluationType, TestCase

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [f"Validation: {len(self.errors)} errors, {len(self.warnings)} warnings"]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN:  {w}")
        return "\n".join(lines)


def validate_dataset(cases: list[TestCase]) -> ValidationReport:
    """
    Ejecuta comprobaciones básicas sobre una lista de casos de prueba.

    Comprobaciones realizadas:
      - IDs duplicados
      - Los casos llm_judge tienen llm_judge_criteria configurado
      - Los casos safety tienen safety_config configurado
      - Los casos multi-turno tienen conversation_history no vacía
      - Las comprobaciones de reglas tienen tipos de cadena válidos
    """
    report = ValidationReport()
    seen_ids: set[str] = set()

    valid_rule_types = {
        "contains", "not_contains", "regex_match", "min_length",
        "max_length", "not_empty", "json_valid", "starts_with", "ends_with",
    }

    for tc in cases:
        prefix = f"[{tc.id}]"

        # ID duplicado
        if tc.id in seen_ids:
            report.errors.append(f"{prefix} ID de caso de prueba duplicado")
        seen_ids.add(tc.id)

        # Consistencia del tipo de evaluación
        if tc.evaluation_type in (EvaluationType.LLM_JUDGE, EvaluationType.COMBINED):
            if tc.llm_judge_criteria is None:
                report.warnings.append(
                    f"{prefix} Usa llm_judge pero no tiene llm_judge_criteria — "
                    "se usarán los valores por defecto"
                )

        if tc.evaluation_type == EvaluationType.SAFETY:
            if tc.safety_config is None:
                report.warnings.append(
                    f"{prefix} Usa evaluación safety pero no tiene safety_config — "
                    "se usarán los valores por defecto"
                )

        # Los casos multi-turno deben tener historial
        if tc.category.value == "multi_turn" and not tc.conversation_history:
            report.warnings.append(
                f"{prefix} La categoría es multi_turn pero conversation_history está vacío"
            )

        # Tipos de comprobaciones de reglas
        for rule in tc.rule_checks:
            if rule.type not in valid_rule_types:
                report.errors.append(
                    f"{prefix} Tipo de regla desconocido '{rule.type}'. "
                    f"Tipos válidos: {sorted(valid_rule_types)}"
                )

        # Input vacío
        if not tc.input.strip():
            report.errors.append(f"{prefix} El input está vacío")

        # expected_behavior vacío
        if not tc.expected_behavior.strip():
            report.warnings.append(f"{prefix} expected_behavior está vacío")

    return report
