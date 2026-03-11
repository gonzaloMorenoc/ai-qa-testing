#!/usr/bin/env python3
"""
CLI entry point for the chatbot QA evaluation framework.

Usage examples:

  # Run all tests with the mock provider (default, no API key needed):
  python run_evaluation.py

  # Run with OpenAI:
  CHATBOT_PROVIDER=openai CHATBOT_API_KEY=sk-... python run_evaluation.py

  # Run with Claude:
  CHATBOT_PROVIDER=claude CHATBOT_API_KEY=... python run_evaluation.py

  # Run only specific categories:
  python run_evaluation.py --categories functional safety

  # Use a custom config file:
  python run_evaluation.py --config my_config.yaml

  # Dry-run: validate datasets only, don't execute tests:
  python run_evaluation.py --validate-only

  # Suppress LLM judge (faster, offline):
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
        description="AI Chatbot QA Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to config YAML file (default: config.yaml)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CATEGORY",
        help="Run only specific categories (e.g. functional safety)",
    )
    parser.add_argument(
        "--provider",
        help="Override chatbot provider (mock | openai | claude)",
    )
    parser.add_argument(
        "--model",
        help="Override model name",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override report output directory",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate datasets; do not run tests",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Disable LLM-as-judge evaluation",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first test failure",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("run_evaluation")

    # -- Load config --------------------------------------------------------
    from chatbot_qa.config import load_config
    config = load_config(args.config)

    # Apply CLI overrides
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

    # -- Load and validate datasets -----------------------------------------
    from chatbot_qa.datasets.loader import load_all_datasets
    from chatbot_qa.datasets.validator import validate_dataset
    from pathlib import Path as _Path

    datasets_dir = _Path(config.runner.datasets_dir)
    logger.info("Loading datasets from '%s'...", datasets_dir)

    try:
        cases = load_all_datasets(datasets_dir, categories=config.runner.categories or None)
    except (FileNotFoundError, NotADirectoryError) as e:
        logger.error("Dataset error: %s", e)
        return 1

    logger.info("Loaded %d test cases total", len(cases))

    validation = validate_dataset(cases)
    if not validation.is_valid:
        logger.error("Dataset validation failed:\n%s", validation.summary())
        return 1

    if validation.warnings:
        logger.warning(validation.summary())

    if args.validate_only:
        logger.info("Validation complete — no errors found. (--validate-only mode)")
        return 0

    # -- Build provider -----------------------------------------------------
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
        logger.error("Failed to initialise provider: %s", e)
        return 1

    logger.info("Using provider: %s (model: %s)", provider.name, provider.model or "default")

    # -- Run tests ----------------------------------------------------------
    from chatbot_qa.runner.test_runner import TestRunner

    runner = TestRunner(config=config, provider=provider)

    try:
        report = runner.run(cases=cases)
    except Exception as e:
        logger.exception("Test run failed: %s", e)
        return 1

    # -- Generate reports ---------------------------------------------------
    output_dir = _Path(config.report.output_dir)

    if config.report.json_enabled:
        from chatbot_qa.reports.json_reporter import write_json_report
        json_path = write_json_report(report, output_dir)
        print(f"JSON report:     {json_path}")

    if config.report.markdown_enabled:
        from chatbot_qa.reports.markdown_reporter import write_markdown_report
        md_path = write_markdown_report(report, output_dir)
        print(f"Markdown report: {md_path}")

    # -- Print summary ------------------------------------------------------
    m = report.metrics
    print("\n" + "=" * 55)
    print(f"  RUN SUMMARY  (id: {report.run_id})")
    print("=" * 55)
    print(f"  Provider   : {report.provider} / {report.model or 'default'}")
    print(f"  Total tests: {m.total}")
    print(f"  Passed     : {m.passed}  ({m.pass_rate:.1%})")
    print(f"  Failed     : {m.failed}")
    print(f"  Errors     : {m.errored}")
    print(f"  Critical   : {m.critical_failures}")
    print(f"  Avg latency: {m.avg_latency_ms:.0f} ms")
    if m.avg_score is not None:
        print(f"  Avg score  : {m.avg_score:.2f}")
    print("=" * 55)
    for cat, stats in sorted(m.by_category.items()):
        print(f"  [{cat:14s}] {stats.passed}/{stats.total} passed  ({stats.pass_rate:.1%})")
    print("=" * 55)

    # Exit code: non-zero if any failures (useful for CI)
    if m.critical_failures > 0:
        return 2  # Critical failures
    if m.failed > 0:
        return 1  # Non-critical failures
    return 0


if __name__ == "__main__":
    sys.exit(main())
