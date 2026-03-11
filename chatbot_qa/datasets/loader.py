"""
Dataset loader: reads JSONL and YAML test case files.

Supports:
  - .jsonl  - one JSON object per line (preferred for large datasets)
  - .yaml / .yml  - YAML list of test case objects (preferred for readability)

Usage:
    from chatbot_qa.datasets.loader import load_dataset, load_all_datasets

    cases = load_dataset(Path("datasets/functional.jsonl"))
    all_cases = load_all_datasets(Path("datasets"), categories=["functional"])
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from chatbot_qa.models import Category, TestCase

logger = logging.getLogger(__name__)


def load_dataset(path: Path) -> list[TestCase]:
    """
    Load test cases from a single JSONL or YAML file.

    Malformed records are logged and skipped rather than crashing the run.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    elif suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    else:
        raise ValueError(f"Unsupported dataset format '{suffix}'. Use .jsonl or .yaml")


def load_all_datasets(
    directory: Path,
    categories: Optional[list[str]] = None,
) -> list[TestCase]:
    """
    Load all dataset files from a directory, optionally filtering by category.

    Files are matched by name prefix: functional.*, safety.*, etc.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Datasets directory not found: {directory}")

    all_cases: list[TestCase] = []
    target_categories = set(categories) if categories else None

    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".jsonl", ".yaml", ".yml"):
            continue

        # Optional category filter by file stem
        stem = path.stem.lower()
        if target_categories and not any(
            stem.startswith(cat) for cat in target_categories
        ):
            logger.debug("Skipping %s (not in requested categories)", path.name)
            continue

        try:
            cases = load_dataset(path)
            logger.info("Loaded %d test cases from %s", len(cases), path.name)
            all_cases.extend(cases)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path.name, exc)

    return all_cases


def _load_jsonl(path: Path) -> list[TestCase]:
    cases: list[TestCase] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
                cases.append(TestCase.model_validate(data))
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON at %s:%d — %s", path.name, line_no, e)
            except ValidationError as e:
                logger.warning(
                    "Skipping invalid test case at %s:%d — %s",
                    path.name,
                    line_no,
                    e,
                )
    return cases


def _load_yaml(path: Path) -> list[TestCase]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(f"YAML dataset must be a list of test cases, got {type(data)}")

    cases: list[TestCase] = []
    for i, item in enumerate(data):
        try:
            cases.append(TestCase.model_validate(item))
        except ValidationError as e:
            logger.warning("Skipping invalid test case #%d in %s — %s", i, path.name, e)

    return cases


def filter_by_category(
    cases: list[TestCase], categories: list[Category]
) -> list[TestCase]:
    """Return only test cases belonging to the specified categories."""
    category_set = set(categories)
    return [c for c in cases if c.category in category_set]
