"""
Anthropic Claude chatbot provider.

Requires: anthropic>=0.25.0
Install: pip install anthropic

Set credentials via environment variables or config.yaml:
  CHATBOT_API_KEY  - your Anthropic API key
  CHATBOT_MODEL    - e.g. 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'
"""

from __future__ import annotations

from typing import Optional

from chatbot_qa.models import Message
from chatbot_qa.runner.providers.base_provider import (
    ChatbotProvider,
    ChatbotProviderError,
)


class ClaudeProvider(ChatbotProvider):
    """Wraps the Anthropic Messages API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        **_kwargs: object,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package is required for ClaudeProvider. "
                "Install with: pip install anthropic"
            ) from e

        self._model = model
        self._timeout = timeout_s

        client_kwargs: dict = {"max_retries": max_retries}
        if api_key:
            client_kwargs["api_key"] = api_key

        self._client = anthropic.Anthropic(**client_kwargs)
        self._anthropic = anthropic

    @property
    def name(self) -> str:
        return "claude"

    @property
    def model(self) -> Optional[str]:
        return self._model

    def chat(
        self,
        user_message: str,
        history: list[Message],
        system_prompt: Optional[str] = None,
    ) -> str:
        messages: list[dict] = []

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        create_kwargs: dict = {
            "model": self._model,
            "max_tokens": 2048,
            "messages": messages,
        }
        if system_prompt:
            create_kwargs["system"] = system_prompt

        try:
            response = self._client.messages.create(**create_kwargs)
            block = response.content[0]
            if hasattr(block, "text"):
                return block.text
            raise ChatbotProviderError("Claude returned a non-text content block.")
        except self._anthropic.APITimeoutError as e:
            raise ChatbotProviderError(f"Claude request timed out: {e}") from e
        except self._anthropic.APIError as e:
            raise ChatbotProviderError(f"Claude API error: {e}") from e
