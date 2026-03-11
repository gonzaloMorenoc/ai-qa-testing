"""
Configuration management for the chatbot QA framework.

Reads from a YAML config file and/or environment variables.
Environment variables take precedence over file config (12-factor app style).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    name: str = "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: float = 30.0
    max_retries: int = 2


class LLMJudgeConfig(BaseModel):
    enabled: bool = True
    provider: str = "mock"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout_s: float = 60.0


class RunnerConfig(BaseModel):
    datasets_dir: Path = Path("datasets")
    categories: list[str] = Field(default_factory=list)  # empty = all
    max_concurrency: int = 1  # sequential by default; increase for async
    fail_fast: bool = False
    skip_on_error: bool = True


class ReportConfig(BaseModel):
    output_dir: Path = Path("reports")
    json_enabled: bool = True
    markdown_enabled: bool = True
    include_full_outputs: bool = True


class Config(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    llm_judge: LLMJudgeConfig = Field(default_factory=LLMJudgeConfig)
    runner: RunnerConfig = Field(default_factory=RunnerConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


def load_config(path: Optional[Path] = None) -> Config:
    """
    Load configuration from a YAML file, then apply environment variable
    overrides for sensitive values (API keys, provider name, model).

    Lookup order (highest precedence first):
      1. Environment variables
      2. YAML config file
      3. Built-in defaults
    """
    raw: dict = {}

    config_path = path or Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    config = Config.model_validate(raw)

    # Environment variable overrides
    if api_key := os.getenv("CHATBOT_API_KEY"):
        config.provider.api_key = api_key
    if provider := os.getenv("CHATBOT_PROVIDER"):
        config.provider.name = provider
    if model := os.getenv("CHATBOT_MODEL"):
        config.provider.model = model
    if base_url := os.getenv("CHATBOT_BASE_URL"):
        config.provider.base_url = base_url

    if judge_key := os.getenv("LLM_JUDGE_API_KEY"):
        config.llm_judge.api_key = judge_key
    if judge_provider := os.getenv("LLM_JUDGE_PROVIDER"):
        config.llm_judge.provider = judge_provider
    if judge_model := os.getenv("LLM_JUDGE_MODEL"):
        config.llm_judge.model = judge_model

    return config
