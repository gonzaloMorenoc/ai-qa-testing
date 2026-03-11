"""
Dataset validation utilities.

Checks for common authoring mistakes before a test run, providing clear
error messages so issues can be fixed before wasting API credits.
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
    Run sanity checks on a list of test cases.

    Checks performed:
      - Duplicate IDs
      - LLM judge cases have llm_judge_criteria configured
      - Safety cases have safety_config configured
      - Multi-turn cases have non-empty conversation_history
      - Rule-based checks have valid type strings
    """
    report = ValidationReport()
    seen_ids: set[str] = set()

    valid_rule_types = {
        "contains", "not_contains", "regex_match", "min_length",
        "max_length", "not_empty", "json_valid", "starts_with", "ends_with",
    }

    for tc in cases:
        prefix = f"[{tc.id}]"

        # Duplicate ID
        if tc.id in seen_ids:
            report.errors.append(f"{prefix} Duplicate test case ID")
        seen_ids.add(tc.id)

        # Evaluation type consistency
        if tc.evaluation_type in (EvaluationType.LLM_JUDGE, EvaluationType.COMBINED):
            if tc.llm_judge_criteria is None:
                report.warnings.append(
                    f"{prefix} Uses llm_judge but has no llm_judge_criteria — "
                    "defaults will be used"
                )

        if tc.evaluation_type == EvaluationType.SAFETY:
            if tc.safety_config is None:
                report.warnings.append(
                    f"{prefix} Uses safety evaluation but has no safety_config — "
                    "defaults will be used"
                )

        # Multi-turn should have history
        if tc.category.value == "multi_turn" and not tc.conversation_history:
            report.warnings.append(
                f"{prefix} Category is multi_turn but conversation_history is empty"
            )

        # Rule check types
        for rule in tc.rule_checks:
            if rule.type not in valid_rule_types:
                report.errors.append(
                    f"{prefix} Unknown rule type '{rule.type}'. "
                    f"Valid types: {sorted(valid_rule_types)}"
                )

        # Empty input
        if not tc.input.strip():
            report.errors.append(f"{prefix} Input is empty")

        # Empty expected_behavior
        if not tc.expected_behavior.strip():
            report.warnings.append(f"{prefix} expected_behavior is empty")

    return report
