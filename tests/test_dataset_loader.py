"""Unit tests for dataset loading and validation."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from chatbot_qa.datasets.loader import load_dataset, load_all_datasets
from chatbot_qa.datasets.validator import validate_dataset
from chatbot_qa.models import Category, TestCase


MINIMAL_CASE = {
    "id": "test-001",
    "category": "functional",
    "input": "Hello",
    "expected_behavior": "Should greet",
    "evaluation_type": "rule_based",
}


class TestDatasetLoader:
    def test_load_jsonl(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(MINIMAL_CASE) + "\n")
        cases = load_dataset(path)
        assert len(cases) == 1
        assert cases[0].id == "test-001"

    def test_load_jsonl_multiple_cases(self, tmp_path):
        path = tmp_path / "test.jsonl"
        case1 = {**MINIMAL_CASE, "id": "test-001"}
        case2 = {**MINIMAL_CASE, "id": "test-002"}
        path.write_text(json.dumps(case1) + "\n" + json.dumps(case2) + "\n")
        cases = load_dataset(path)
        assert len(cases) == 2

    def test_load_jsonl_skips_empty_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(MINIMAL_CASE) + "\n\n\n")
        cases = load_dataset(path)
        assert len(cases) == 1

    def test_load_jsonl_skips_invalid_json(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps(MINIMAL_CASE) + "\n{invalid json}\n")
        cases = load_dataset(path)
        assert len(cases) == 1  # Invalid line is skipped

    def test_load_yaml(self, tmp_path):
        path = tmp_path / "test.yaml"
        path.write_text(yaml.dump([MINIMAL_CASE]))
        cases = load_dataset(path)
        assert len(cases) == 1
        assert cases[0].id == "test-001"

    def test_load_yaml_list(self, tmp_path):
        path = tmp_path / "test.yaml"
        cases_data = [{**MINIMAL_CASE, "id": f"test-{i:03d}"} for i in range(3)]
        path.write_text(yaml.dump(cases_data))
        cases = load_dataset(path)
        assert len(cases) == 3

    def test_load_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.jsonl")

    def test_load_unsupported_format(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("id,category\ntest-001,functional\n")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            load_dataset(path)

    def test_load_all_datasets(self, tmp_path):
        (tmp_path / "functional.jsonl").write_text(
            json.dumps({**MINIMAL_CASE, "id": "func-001"}) + "\n"
        )
        safety_case = {
            **MINIMAL_CASE,
            "id": "safe-001",
            "category": "safety",
            "evaluation_type": "safety",
        }
        (tmp_path / "safety.jsonl").write_text(json.dumps(safety_case) + "\n")

        cases = load_all_datasets(tmp_path)
        assert len(cases) == 2

    def test_load_all_datasets_with_category_filter(self, tmp_path):
        (tmp_path / "functional.jsonl").write_text(
            json.dumps({**MINIMAL_CASE, "id": "func-001"}) + "\n"
        )
        safety_case = {
            **MINIMAL_CASE,
            "id": "safe-001",
            "category": "safety",
            "evaluation_type": "safety",
        }
        (tmp_path / "safety.jsonl").write_text(json.dumps(safety_case) + "\n")

        cases = load_all_datasets(tmp_path, categories=["functional"])
        assert len(cases) == 1
        assert cases[0].id == "func-001"

    def test_load_all_datasets_directory_not_found(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            load_all_datasets(tmp_path / "nonexistent")


class TestDatasetValidator:
    def test_valid_dataset_passes(self):
        cases = [TestCase.model_validate(MINIMAL_CASE)]
        report = validate_dataset(cases)
        assert report.is_valid

    def test_duplicate_ids_detected(self):
        case1 = TestCase.model_validate({**MINIMAL_CASE, "id": "dup-001"})
        case2 = TestCase.model_validate({**MINIMAL_CASE, "id": "dup-001"})
        report = validate_dataset([case1, case2])
        assert not report.is_valid
        assert any("Duplicate" in e for e in report.errors)

    def test_unknown_rule_type_detected(self):
        cases = [
            TestCase.model_validate(
                {**MINIMAL_CASE, "rule_checks": [{"type": "nonexistent"}]}
            )
        ]
        report = validate_dataset(cases)
        assert not report.is_valid

    def test_multi_turn_without_history_warns(self):
        cases = [
            TestCase.model_validate(
                {**MINIMAL_CASE, "category": "multi_turn", "conversation_history": []}
            )
        ]
        report = validate_dataset(cases)
        assert report.is_valid  # Warning, not error
        assert any("multi_turn" in w for w in report.warnings)

    def test_empty_dataset_passes(self):
        report = validate_dataset([])
        assert report.is_valid
