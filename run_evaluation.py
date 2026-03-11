#!/usr/bin/env python3
"""
Punto de entrada CLI del framework de evaluación QA para chatbots.

Ejemplos de uso:

  # Ejecutar todos los tests con el proveedor mock (por defecto, sin necesidad de API key):
  python run_evaluation.py

  # Ejecutar con OpenAI:
  CHATBOT_PROVIDER=openai CHATBOT_API_KEY=sk-... python run_evaluation.py

  # Ejecutar con Claude:
  CHATBOT_PROVIDER=claude CHATBOT_API_KEY=... python run_evaluation.py

  # Ejecutar solo categorías específicas:
  python run_evaluation.py --categories functional safety

  # Usar un archivo de configuración personalizado:
  python run_evaluation.py --config mi_config.yaml

  # Ejecución en seco: solo validar datasets, sin ejecutar tests:
  python run_evaluation.py --validate-only

  # Desactivar el juez LLM (más rápido, sin conexión):
  python run_evaluation.py --no-judge
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Framework de Evaluación QA para Chatbots de IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Ruta al archivo YAML de configuración (por defecto: config.yaml)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CATEGORIA",
        help="Ejecutar solo categorías específicas (ej. functional safety)",
    )
    parser.add_argument(
        "--provider",
        help="Sobreescribir el proveedor del chatbot (mock | openai | claude)",
    )
    parser.add_argument(
        "--model",
        help="Sobreescribir el nombre del modelo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Sobreescribir el directorio de salida de reportes",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Solo validar los datasets; no ejecutar los tests",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Desactivar la evaluación LLM-as-judge",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Detener la ejecución tras el primer fallo",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Activar logging de depuración",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("run_evaluation")

    # -- Cargar configuración -----------------------------------------------
    from chatbot_qa.config import load_config
    config = load_config(args.config)

    # Aplicar sobreescrituras de la CLI
    if args.categories:
        config.runner.categories = args.categories
    if args.provider:
        config.provider.name = args.provider
    if args.model:
        config.provider.model = args.model
    if args.output_dir:
        config.report.output_dir = args.output_dir
    if args.no_judge:
        config.llm_judge.enabled = False
    if args.fail_fast:
        config.runner.fail_fast = True

    # -- Cargar y validar datasets ------------------------------------------
    from chatbot_qa.datasets.loader import load_all_datasets
    from chatbot_qa.datasets.validator import validate_dataset
    from pathlib import Path as _Path

    datasets_dir = _Path(config.runner.datasets_dir)
    logger.info("Cargando datasets desde '%s'...", datasets_dir)

    try:
        cases = load_all_datasets(datasets_dir, categories=config.runner.categories or None)
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error("Error en el dataset: %s", e)
        return 1

    logger.info("Cargados %d casos de prueba en total", len(cases))

    validation = validate_dataset(cases)
    if not validation.is_valid:
        logger.error("La validación del dataset falló:\n%s", validation.summary())
        return 1

    if validation.warnings:
        logger.warning(validation.summary())

    if args.validate_only:
        logger.info("Validación completada — no se encontraron errores. (modo --validate-only)")
        return 0

    # -- Construir proveedor ------------------------------------------------
    from chatbot_qa.runner.providers.base_provider import get_provider

    provider_kwargs = {}
    if config.provider.api_key:
        provider_kwargs["api_key"] = config.provider.api_key
    if config.provider.model:
        provider_kwargs["model"] = config.provider.model
    if config.provider.base_url:
        provider_kwargs["base_url"] = config.provider.base_url
    provider_kwargs["timeout_s"] = config.provider.timeout_s
    provider_kwargs["max_retries"] = config.provider.max_retries

    try:
        provider = get_provider(config.provider.name, **provider_kwargs)
    except (ValueError, ImportError) as e:
        logger.error("Error al inicializar el proveedor: %s", e)
        return 1

    logger.info("Usando proveedor: %s (modelo: %s)", provider.name, provider.model or "por defecto")

    # -- Ejecutar tests -----------------------------------------------------
    from chatbot_qa.runner.test_runner import TestRunner

    runner = TestRunner(config=config, provider=provider)

    try:
        report = runner.run(cases=cases)
    except Exception as e:
        logger.exception("La ejecución de tests falló: %s", e)
        return 1

    # -- Generar reportes ---------------------------------------------------
    output_dir = _Path(config.report.output_dir)

    if config.report.json_enabled:
        from chatbot_qa.reports.json_reporter import write_json_report
        json_path = write_json_report(report, output_dir)
        print(f"Reporte JSON:     {json_path}")

    if config.report.markdown_enabled:
        from chatbot_qa.reports.markdown_reporter import write_markdown_report
        md_path = write_markdown_report(report, output_dir)
        print(f"Reporte Markdown: {md_path}")

    # -- Imprimir resumen ---------------------------------------------------
    m = report.metrics
    print("\n" + "=" * 55)
    print(f"  RESUMEN DE EJECUCIÓN  (id: {report.run_id})")
    print("=" * 55)
    print(f"  Proveedor    : {report.provider} / {report.model or 'por defecto'}")
    print(f"  Total tests  : {m.total}")
    print(f"  Aprobados    : {m.passed}  ({m.pass_rate:.1%})")
    print(f"  Fallidos     : {m.failed}")
    print(f"  Errores      : {m.errored}")
    print(f"  Críticos     : {m.critical_failures}")
    print(f"  Lat. media   : {m.avg_latency_ms:.0f} ms")
    if m.avg_score is not None:
        print(f"  Puntuación   : {m.avg_score:.2f}")
    print("=" * 55)
    for cat, stats in sorted(m.by_category.items()):
        print(f"  [{cat:14s}] {stats.passed}/{stats.total} aprobados  ({stats.pass_rate:.1%})")
    print("=" * 55)

    # Código de salida no cero si hay fallos (útil para CI)
    if m.critical_failures > 0:
        return 2  # Fallos críticos
    if m.failed > 0:
        return 1  # Fallos no críticos
    return 0


if __name__ == "__main__":
    sys.exit(main())
