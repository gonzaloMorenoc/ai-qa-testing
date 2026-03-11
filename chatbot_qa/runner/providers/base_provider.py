"""
Abstract base class for chatbot providers.

To add a new chatbot backend:
  1. Subclass ChatbotProvider
  2. Implement the `chat` method
  3. Register it in get_provider() below
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from chatbot_qa.models import Message


class ChatbotProvider(ABC):
    """
    Minimal interface that every chatbot backend must implement.

    The runner only depends on this interface, making it trivial to swap
    providers without changing any evaluation logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this provider (used in reports)."""

    @property
    def model(self) -> Optional[str]:
        """Model identifier, if applicable."""
        return None

    @abstractmethod
    def chat(
        self,
        user_message: str,
        history: list[Message],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Send a message to the chatbot and return its text response.

        Args:
            user_message:  The current user turn.
            history:       Prior conversation turns (role + content).
            system_prompt: Optional system-level instruction.

        Returns:
            The chatbot's text response.

        Raises:
            ChatbotProviderError: On any communication or API error.
        """


class ChatbotProviderError(Exception):
    """Raised when a provider fails to produce a response."""


def get_provider(provider_name: str, **kwargs: object) -> ChatbotProvider:
    """
    Factory: return a ChatbotProvider instance by name.

    Supported names: 'mock', 'openai', 'claude'
    """
    from chatbot_qa.runner.providers.mock_provider import MockProvider
    from chatbot_qa.runner.providers.openai_provider import OpenAIProvider
    from chatbot_qa.runner.providers.claude_provider import ClaudeProvider

    providers: dict[str, type[ChatbotProvider]] = {
        "mock": MockProvider,
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
    }

    cls = providers.get(provider_name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown provider '{provider_name}'. "
            f"Available: {list(providers.keys())}"
        )
    return cls(**kwargs)  # type: ignore[arg-type]
