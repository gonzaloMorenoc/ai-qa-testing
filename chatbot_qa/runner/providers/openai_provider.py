"""
Proveedor de chatbot OpenAI.

Requiere: openai>=1.0.0
Instalar: pip install openai

Configurar credenciales via variables de entorno o config.yaml:
  CHATBOT_API_KEY  - tu clave de API de OpenAI
  CHATBOT_MODEL    - ej. 'gpt-4o', 'gpt-4-turbo', 'gpt-3.5-turbo'
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
    Envuelve la API de Chat Completions de OpenAI.

    Este proveedor también es compatible con cualquier endpoint compatible
    con OpenAI (p.ej. Ollama local, Azure OpenAI) sobreescribiendo base_url.
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
                "El paquete openai es necesario para OpenAIProvider. "
                "Instalar con: pip install openai"
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
                raise ChatbotProviderError("OpenAI devolvió una respuesta vacía.")
            return content
        except self._APITimeoutError as e:
            raise ChatbotProviderError(f"La petición a OpenAI excedió el tiempo límite: {e}") from e
        except self._APIError as e:
            raise ChatbotProviderError(f"Error de la API de OpenAI: {e}") from e
