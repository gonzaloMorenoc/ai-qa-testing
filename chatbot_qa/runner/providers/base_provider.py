"""
Clase base abstracta para proveedores de chatbot.

Para añadir un nuevo backend de chatbot:
  1. Heredar de ChatbotProvider
  2. Implementar el método `chat`
  3. Registrarlo en get_provider() más abajo
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from chatbot_qa.models import Message


class ChatbotProvider(ABC):
    """
    Interfaz mínima que todo backend de chatbot debe implementar.

    El runner solo depende de esta interfaz, lo que permite intercambiar
    proveedores sin modificar ninguna lógica de evaluación.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador legible por humanos para este proveedor (usado en reportes)."""

    @property
    def model(self) -> Optional[str]:
        """Identificador del modelo, si aplica."""
        return None

    @abstractmethod
    def chat(
        self,
        user_message: str,
        history: list[Message],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Envía un mensaje al chatbot y devuelve su respuesta en texto.

        Args:
            user_message:  El turno actual del usuario.
            history:       Turnos previos de la conversación (role + content).
            system_prompt: Instrucción de sistema opcional.

        Returns:
            La respuesta en texto del chatbot.

        Raises:
            ChatbotProviderError: Ante cualquier error de comunicación o API.
        """


class ChatbotProviderError(Exception):
    """Se lanza cuando un proveedor no consigue producir una respuesta."""


def get_provider(provider_name: str, **kwargs: object) -> ChatbotProvider:
    """
    Fábrica: devuelve una instancia de ChatbotProvider por nombre.

    Nombres soportados: 'mock', 'openai', 'claude'
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
            f"Proveedor desconocido '{provider_name}'. "
            f"Disponibles: {list(providers.keys())}"
        )
    return cls(**kwargs)  # type: ignore[arg-type]
