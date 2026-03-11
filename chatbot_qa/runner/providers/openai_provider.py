"""
OpenAI chatbot provider.

Requires: openai>=1.0.0
Install: pip install openai

Set credentials via environment variables or config.yaml:
  CHATBOT_API_KEY  - your OpenAI API key
  CHATBOT_MODEL    - e.g. 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'
"""

from __future__ import annotations

from typing import Optional

from chatbot_qa.models import Message
from chatbot_qa.runner.providers.base_provider import (
    ChatbotProvider,
    ChatbotProviderError,
)


class OpenAIProvider(ChatbotProvider):
    """
    Wraps the OpenAI Chat Completions API.

    This provider is also compatible with any OpenAI-compatible endpoint
    (e.g. local Ollama, Azure OpenAI) by overriding base_url.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        **_kwargs: object,
    ) -> None:
        try:
            from openai import OpenAI, APIError, APITimeoutError
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIProvider. "
                "Install with: pip install openai"
            ) from e

        self._model = model
        self._timeout = timeout_s
        self._APIError = APIError
        self._APITimeoutError = APITimeoutError

        client_kwargs: dict = {"max_retries": max_retries}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

    @property
    def name(self) -> str:
        return "openai"

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

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": user_message})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                timeout=self._timeout,
            )
            content = response.choices[0].message.content
            if content is None:
                raise ChatbotProviderError("OpenAI returned an empty response.")
            return content
        except self._APITimeoutError as e:
            raise ChatbotProviderError(f"OpenAI request timed out: {e}") from e
        except self._APIError as e:
            raise ChatbotProviderError(f"OpenAI API error: {e}") from e
