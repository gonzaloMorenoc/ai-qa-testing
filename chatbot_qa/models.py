"""
Modelos de datos principales del framework de QA para chatbots.

Todos los modelos usan Pydantic v2 para validación en tiempo de ejecución,
serialización y soporte de autocompletado en el IDE. Son los contratos
compartidos entre el runner, los evaluadores, las métricas y los reportes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumeraciones
# ---------------------------------------------------------------------------

class Category(str, Enum):
    FUNCTIONAL = "functional"
    MULTI_TURN = "multi_turn"
    SAFETY = "safety"
    REGRESSION = "regression"


class EvaluationType(str, Enum):
    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"
    SAFETY = "safety"
    COMBINED = "combined"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Definición del caso de prueba
# ---------------------------------------------------------------------------

class Message(BaseModel):
    """Un único mensaje dentro de una conversación."""
    role: str = Field(..., description="'user' o 'assistant'")
    content: str


class RuleCheck(BaseModel):
    """Configuración para una comprobación de regla individual."""
    type: str = Field(
        ...,
        description=(
            "Uno de: contains, not_contains, regex_match, min_length, "
            "max_length, not_empty, json_valid, starts_with, ends_with"
        ),
    )
    value: Optional[Any] = None
    description: Optional[str] = None


class LLMJudgeCriteria(BaseModel):
    """Rúbrica de criterios que se pasa al evaluador LLM-as-judge."""
    relevance: bool = True
    correctness: bool = True
    clarity: bool = True
    safety: bool = True
    instruction_following: bool = True
    custom_criteria: list[str] = Field(default_factory=list)
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)


class SafetyCheckConfig(BaseModel):
    """Configuración de las comprobaciones de seguridad para un caso de prueba."""
    check_prompt_injection: bool = False
    check_system_prompt_leak: bool = False
    check_unsafe_content: bool = False
    should_refuse: bool = False


class TestCase(BaseModel):
    """
    Un caso de prueba individual a ejecutar contra el chatbot.

    Soporta conversaciones de un solo turno (solo input) y multi-turno
    (historial + input). Toda la configuración de evaluación está embebida
    en el propio caso, de modo que el runner y los evaluadores son
    completamente orientados a datos.
    """
    id: str
    category: Category
    description: Optional[str] = None

    # Conversación
    input: str = Field(..., description="El mensaje de usuario final a enviar")
    conversation_history: list[Message] = Field(
        default_factory=list,
        description="Turnos previos para tests multi-turno (pares role/content)",
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt opcional a anteponer a la conversación",
    )

    # Descripción del comportamiento esperado (legible por humanos, usada como contexto para el juez LLM)
    expected_behavior: str

    # Configuración de evaluación
    evaluation_type: EvaluationType
    rule_checks: list[RuleCheck] = Field(default_factory=list)
    llm_judge_criteria: Optional[LLMJudgeCriteria] = None
    safety_config: Optional[SafetyCheckConfig] = None

    # Metadatos
    severity: Severity = Severity.MEDIUM
    tags: list[str] = Field(default_factory=list)
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Resultados de evaluación
# ---------------------------------------------------------------------------

class RuleCheckResult(BaseModel):
    rule_type: str
    passed: bool
    message: str
    value: Optional[Any] = None


class LLMJudgeScore(BaseModel):
    relevance: Optional[float] = None
    correctness: Optional[float] = None
    clarity: Optional[float] = None
    safety: Optional[float] = None
    instruction_following: Optional[float] = None
    custom_scores: dict[str, float] = Field(default_factory=dict)
    overall: float
    reasoning: str


class SafetyCheckResult(BaseModel):
    prompt_injection_detected: bool = False
    system_prompt_leaked: bool = False
    unsafe_content_detected: bool = False
    refused_when_required: Optional[bool] = None
    refused_when_not_required: bool = False
    details: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Resultado de evaluación agregado para la ejecución de un caso de prueba."""
    passed: bool
    score: Optional[float] = None
    rule_results: list[RuleCheckResult] = Field(default_factory=list)
    llm_judge: Optional[LLMJudgeScore] = None
    safety: Optional[SafetyCheckResult] = None
    failure_reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Resultado de ejecución de test
# ---------------------------------------------------------------------------

class TestResult(BaseModel):
    """Registro completo de la ejecución de un caso de prueba."""
    test_case: TestCase
    chatbot_output: Optional[str] = None
    duration_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: TestStatus
    evaluation: Optional[EvaluationResult] = None
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED


# ---------------------------------------------------------------------------
# Reporte agregado
# ---------------------------------------------------------------------------

class CategoryStats(BaseModel):
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    avg_score: Optional[float] = None
    avg_latency_ms: float


class Metrics(BaseModel):
    total: int
    passed: int
    failed: int
    errored: int
    pass_rate: float
    avg_score: Optional[float] = None
    avg_latency_ms: float
    critical_failures: int
    by_category: dict[str, CategoryStats]


class Report(BaseModel):
    """Reporte de evaluación de alto nivel generado tras una ejecución completa de tests."""
    run_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provider: str
    model: Optional[str] = None
    metrics: Metrics
    results: list[TestResult]
    version: str = "0.1.0"
