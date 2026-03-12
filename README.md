# chatbot-qa-framework

Framework modular y extensible de QA para evaluar chatbots de IA. Diseñado para uso profesional en pipelines de CI/CD, comparación de modelos y monitoreo continuo de calidad.

---

## Tabla de contenidos

- [Objetivo](#objetivo)
- [Arquitectura](#arquitectura)
- [Inicio rápido](#inicio-rápido)
- [Cómo ejecutar los tests](#cómo-ejecutar-los-tests)
- [Cómo añadir casos de prueba](#cómo-añadir-casos-de-prueba)
- [Cómo añadir nuevos evaluadores](#cómo-añadir-nuevos-evaluadores)
- [Cómo conectar un chatbot real](#cómo-conectar-un-chatbot-real)
- [Cómo interpretar los resultados](#cómo-interpretar-los-resultados)
- [Integración con CI/CD](#integración-con-cicd)
- [Referencia de configuración](#referencia-de-configuración)

---

## Objetivo

Este framework permite a los equipos:

1. **Ejecutar tests** contra cualquier chatbot de IA a través de una interfaz independiente del proveedor
2. **Evaluar la calidad** de las respuestas automáticamente mediante reglas, puntuación LLM-as-judge y heurísticas de seguridad
3. **Detectar regresiones** comparando resultados entre versiones del modelo
4. **Generar reportes** en JSON (legible por máquinas) y Markdown (legible por personas)
5. **Integrarse en CI/CD** con códigos de salida significativos

---

## Arquitectura

```
chatbot_qa/
├── models.py          # Contratos de datos Pydantic (TestCase, TestResult, Report, etc.)
├── config.py          # Configuración vía YAML y variables de entorno
├── runner/
│   ├── test_runner.py             # Orquestador: cargar → ejecutar → evaluar → reportar
│   └── providers/
│       ├── base_provider.py       # Interfaz abstracta ChatbotProvider + fábrica
│       ├── mock_provider.py       # Mock determinista (sin necesidad de API)
│       ├── openai_provider.py     # OpenAI / endpoints compatibles con OpenAI
│       └── claude_provider.py     # Anthropic Claude
├── evaluators/
│   ├── base_evaluator.py          # Interfaz abstracta de evaluadores
│   ├── rule_based.py              # Comprobaciones rápidas y deterministas
│   ├── llm_judge.py               # LLM-as-judge con rúbrica JSON
│   └── safety_checks.py          # Seguridad: detección de inyecciones y contenido dañino
├── datasets/
│   ├── loader.py                  # Lector de datasets JSONL y YAML
│   └── validator.py               # Validación previa al run
├── metrics/
│   └── aggregator.py              # Cálculo de estadísticas agregadas
└── reports/
    ├── json_reporter.py           # Salida JSON legible por máquinas
    └── markdown_reporter.py       # Resumen Markdown legible por personas

datasets/                          # Archivos de casos de prueba (JSONL y YAML)
reports/                           # Reportes generados (en .gitignore)
tests/                             # Tests unitarios del framework
run_evaluation.py                  # Punto de entrada CLI
config.yaml                        # Configuración por defecto
```

### Flujo de datos

```
datasets/*.jsonl / *.yaml
         │
         ▼
   DatasetLoader  ──► DatasetValidator
         │
         ▼
     TestRunner
         │
    por cada caso:
         │
         ├──► ChatbotProvider.chat()    ← backend intercambiable
         │
         ├──► RuleBasedEvaluator
         ├──► SafetyEvaluator
         └──► LLMJudgeEvaluator  ← opcional, usa un segundo proveedor
                    │
                    ▼
             EvaluationResult
                    │
                    ▼
            MetricsAggregator
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
   JSONReporter        MarkdownReporter
```

### Principios de diseño

- **Modular**: runner, evaluadores, datasets y reportes están completamente desacoplados
- **Orientado a datos**: toda la lógica de evaluación vive en la definición del caso, no en el código
- **Independiente del proveedor**: añade cualquier backend LLM sin tocar el runner
- **Compatible con CI**: códigos de salida con semántica clara, salida JSON legible por máquinas

---

## Inicio rápido

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

Para proveedores de chatbots reales, instala los extras opcionales:

```bash
pip install openai      # para OpenAI
pip install anthropic   # para Claude
```

### 2. Ejecutar con el mock (sin API key)

```bash
python run_evaluation.py --no-judge
```

### 3. Ejecutar con OpenAI

```bash
export CHATBOT_PROVIDER=openai
export CHATBOT_API_KEY=sk-...
export CHATBOT_MODEL=gpt-4o
export LLM_JUDGE_PROVIDER=openai
export LLM_JUDGE_API_KEY=sk-...
export LLM_JUDGE_MODEL=gpt-4o

python run_evaluation.py
```

### 4. Ejecutar con Claude

```bash
export CHATBOT_PROVIDER=claude
export CHATBOT_API_KEY=...
export CHATBOT_MODEL=claude-sonnet-4-6

python run_evaluation.py
```

---

## Cómo ejecutar los tests

### Suite completa (todas las categorías)

```bash
python run_evaluation.py
```

### Categorías específicas

```bash
python run_evaluation.py --categories functional safety
python run_evaluation.py --categories regression
```

### Sólo validar datasets (sin llamadas a la API)

```bash
python run_evaluation.py --validate-only
```

### Detener al primer fallo

```bash
python run_evaluation.py --fail-fast
```

### Ejecutar los tests unitarios del framework

```bash
pytest tests/ -v
pytest tests/ -v --cov=chatbot_qa
```

---

## Cómo añadir casos de prueba

Los casos de prueba pueden añadirse a cualquier archivo `.jsonl` o `.yaml` dentro del directorio `datasets/`.

### Campos mínimos requeridos

```json
{
  "id": "id-unico-001",
  "category": "functional",
  "input": "¿Cuál es la capital de Francia?",
  "expected_behavior": "Debe responder correctamente que es París.",
  "evaluation_type": "rule_based",
  "severity": "medium"
}
```

### Ejemplo completo con todas las opciones (JSONL)

```json
{
  "id": "func-042",
  "category": "functional",
  "description": "Consulta de capital de país",
  "input": "¿Cuál es la capital de Francia?",
  "expected_behavior": "Debe identificar correctamente París como la capital de Francia.",
  "conversation_history": [],
  "system_prompt": "Eres un asistente de geografía.",
  "evaluation_type": "combined",
  "rule_checks": [
    { "type": "contains", "value": "París" },
    { "type": "min_length", "value": 5 },
    { "type": "max_length", "value": 500 }
  ],
  "llm_judge_criteria": {
    "relevance": true,
    "correctness": true,
    "clarity": true,
    "safety": false,
    "instruction_following": true,
    "min_score": 0.75
  },
  "severity": "medium",
  "tags": ["geografía", "factual"]
}
```

### Ejemplo multi-turno (YAML)

```yaml
- id: mt-010
  category: multi_turn
  input: "¿Cuál era la ciudad que mencioné?"
  conversation_history:
    - role: user
      content: "Estoy planeando visitar Tokio."
    - role: assistant
      content: "¡Tokio es un destino fantástico!"
  expected_behavior: >
    Debe recordar Tokio del historial de conversación.
  evaluation_type: combined
  rule_checks:
    - type: contains
      value: "Tokio"
  severity: high
```

### Tipos de evaluación

| Tipo | Descripción |
|------|-------------|
| `rule_based` | Solo ejecuta comprobaciones de reglas (rápido, determinista) |
| `llm_judge` | Solo puntuación LLM-as-judge |
| `safety` | Solo comprobaciones de seguridad heurísticas |
| `combined` | Se ejecutan todos los evaluadores aplicables |

### Tipos de reglas disponibles

| Tipo | Ejemplo de valor | Descripción |
|------|-----------------|-------------|
| `not_empty` | — | La respuesta no debe estar vacía |
| `contains` | `"París"` | La respuesta debe contener la subcadena (sin distinción de mayúsculas) |
| `not_contains` | `"error"` | La respuesta NO debe contener la subcadena |
| `regex_match` | `"\\d+"` | La respuesta debe coincidir con el patrón regex |
| `min_length` | `50` | La respuesta debe tener ≥ N caracteres |
| `max_length` | `500` | La respuesta debe tener ≤ N caracteres |
| `json_valid` | — | La respuesta debe ser JSON válido |
| `starts_with` | `"Aquí"` | La respuesta debe comenzar con la subcadena |
| `ends_with` | `"."` | La respuesta debe terminar con la subcadena |

---

## Cómo añadir nuevos evaluadores

1. Crea un nuevo archivo en `chatbot_qa/evaluators/`:

```python
# chatbot_qa/evaluators/mi_evaluador.py
from chatbot_qa.evaluators.base_evaluator import BaseEvaluator
from chatbot_qa.models import EvaluationResult, TestCase

class MiEvaluador(BaseEvaluator):
    def applies_to(self, case: TestCase) -> bool:
        # Devuelve True para los casos en los que debe ejecutarse este evaluador
        return "mi-etiqueta" in case.tags

    def evaluate(self, case: TestCase, output: str) -> EvaluationResult:
        passed = "frase esperada" in output.lower()
        return EvaluationResult(
            passed=passed,
            failure_reasons=[] if passed else ["No se encontró la frase esperada"],
        )
```

2. Registrarlo en `TestRunner._build_evaluators()` en `chatbot_qa/runner/test_runner.py`:

```python
def _build_evaluators(self, extras):
    evaluators = [
        RuleBasedEvaluator(),
        SafetyEvaluator(),
        MiEvaluador(),   # ← añadir aquí
    ]
    ...
```

O inyectarlo en tiempo de ejecución:

```python
from chatbot_qa.evaluators.mi_evaluador import MiEvaluador
runner = TestRunner(config, provider, extra_evaluators=[MiEvaluador()])
```

---

## Cómo conectar un chatbot real

### Usar un proveedor existente

Establece las variables de entorno necesarias (ver [Inicio rápido](#inicio-rápido)).

### Añadir un proveedor personalizado

```python
# chatbot_qa/runner/providers/mi_proveedor.py
from chatbot_qa.runner.providers.base_provider import ChatbotProvider, ChatbotProviderError
from chatbot_qa.models import Message
from typing import Optional

class MiProveedor(ChatbotProvider):
    def __init__(self, api_key: str, **kwargs):
        self._client = MiCliente(api_key=api_key)

    @property
    def name(self) -> str:
        return "mi-chatbot"

    def chat(self, user_message: str, history: list[Message], system_prompt: Optional[str] = None) -> str:
        try:
            return self._client.enviar(user_message)
        except Exception as e:
            raise ChatbotProviderError(str(e)) from e
```

Registrarlo en `chatbot_qa/runner/providers/base_provider.py`:

```python
providers = {
    "mock": MockProvider,
    "openai": OpenAIProvider,
    "claude": ClaudeProvider,
    "mi-chatbot": MiProveedor,  # ← añadir aquí
}
```

Usarlo con: `CHATBOT_PROVIDER=mi-chatbot python run_evaluation.py`

---

## Cómo interpretar los resultados

### Códigos de salida

| Código | Significado |
|--------|-------------|
| `0` | Todos los tests pasaron |
| `1` | Algunos tests no críticos fallaron |
| `2` | Se detectaron fallos críticos — bloquear el despliegue |

### Estructura del reporte JSON

```json
{
  "run_id": "abc12345",
  "provider": "openai",
  "model": "gpt-4o",
  "metrics": {
    "total": 27,
    "passed": 24,
    "failed": 3,
    "pass_rate": 0.889,
    "avg_score": 0.82,
    "avg_latency_ms": 1234,
    "critical_failures": 0,
    "by_category": { "...": "..." }
  },
  "results": ["..."]
}
```

### Secciones del reporte Markdown

| Sección | Contenido |
|---------|-----------|
| Overview | Tabla resumen de métricas de aprobación/fallo |
| Resultados por categoría | Desglose por categoría |
| Tests fallidos | Vista detallada de cada fallo con entrada, salida y motivos |
| Recomendaciones | Acciones automatizadas basadas en los patrones de fallo |

### Comparación entre versiones

Para detectar regresiones, guarda los reportes por run_id y compara el JSON:

```bash
# Ejecutar antes de la actualización del modelo
python run_evaluation.py --output-dir reports/v1

# Ejecutar después de la actualización del modelo
python run_evaluation.py --output-dir reports/v2

# Comparar métricas
diff <(jq '.metrics' reports/v1/*.json) <(jq '.metrics' reports/v2/*.json)
```

---

## Integración con CI/CD

### Ejemplo con GitHub Actions

```yaml
name: Chatbot QA

on:
  pull_request:
  schedule:
    - cron: "0 8 * * *"  # Diario a las 08:00 UTC

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configurar Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Instalar dependencias
        run: pip install -r requirements.txt openai

      - name: Ejecutar evaluación QA
        env:
          CHATBOT_PROVIDER: openai
          CHATBOT_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          CHATBOT_MODEL: gpt-4o
          LLM_JUDGE_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python run_evaluation.py

      - name: Subir reportes
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: qa-reports
          path: reports/
```

El pipeline fallará automáticamente (`exit 1` o `exit 2`) si los tests fallan, bloqueando el PR.

---

## Referencia de configuración

Todas las configuraciones pueden sobreescribirse con variables de entorno (mayor prioridad que `config.yaml`).

| Clave YAML | Variable de entorno | Por defecto | Descripción |
|------------|---------------------|-------------|-------------|
| `provider.name` | `CHATBOT_PROVIDER` | `mock` | Nombre del proveedor |
| `provider.model` | `CHATBOT_MODEL` | `null` | Identificador del modelo |
| `provider.api_key` | `CHATBOT_API_KEY` | `null` | Clave de API |
| `provider.base_url` | `CHATBOT_BASE_URL` | `null` | Endpoint personalizado |
| `provider.timeout_s` | — | `30.0` | Tiempo límite de petición (segundos) |
| `provider.max_retries` | — | `2` | Intentos de reintento |
| `llm_judge.enabled` | — | `true` | Activar/desactivar el juez LLM |
| `llm_judge.provider` | `LLM_JUDGE_PROVIDER` | `mock` | Proveedor del juez |
| `llm_judge.model` | `LLM_JUDGE_MODEL` | `null` | Modelo del juez |
| `llm_judge.api_key` | `LLM_JUDGE_API_KEY` | `null` | Clave de API del juez |
| `runner.datasets_dir` | — | `datasets` | Directorio de archivos de datasets |
| `runner.categories` | — | `[]` (todos) | Filtro de categorías |
| `runner.max_workers` | — | `10` | Nivel de concurrencia: hilos máximos a lanzar en paralelo para evaluación |
| `runner.fail_fast` | — | `false` | Detener al primer fallo |
| `report.output_dir` | — | `reports` | Directorio de salida de reportes |
| `report.json_enabled` | — | `true` | Generar reporte JSON |
| `report.markdown_enabled` | — | `true` | Generar reporte Markdown |
