"""
chatbot_qa — Framework modular de QA para evaluación de chatbots de IA.

Módulos:
    models      - Estructuras de datos principales (TestCase, TestResult, EvaluationResult, Report)
    config      - Gestión de configuración
    runner      - Orquestación de la ejecución y abstracción del proveedor de chatbot
    evaluators  - Evaluadores por reglas, LLM-as-judge y comprobaciones de seguridad
    datasets    - Carga y validación de datasets
    metrics     - Cálculo de estadísticas agregadas
    reports     - Generación de reportes en JSON y Markdown
"""

__version__ = "0.1.0"
