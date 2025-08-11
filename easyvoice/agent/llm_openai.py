"""OpenAI LLM integration for voice agent"""

import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import openai
    from openai import AsyncOpenAI

    HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore
    AsyncOpenAI = None  # type: ignore
    HAS_OPENAI = False

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIInterface:
    """OpenAI LLM interface for voice agent"""

    def __init__(self, settings: Settings):
        """Initialize OpenAI interface

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.client = None

        if not HAS_OPENAI:
            logger.warning("OpenAI package not available - using mock responses")
            return

        if not settings.openai_api_key:
            logger.warning("No OpenAI API key provided - using mock responses")
            return

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAI client initialized")

    async def generate_response(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """Generate response from OpenAI

        Args:
            message: User message to process
            context: Optional conversation context
            timeout: Optional timeout override

        Returns:
            LLM response text or None if failed
        """
        if not self.client or not HAS_OPENAI:
            return f"Mock response to: {message}"

        messages = None
        try:
            messages = self._prepare_messages(message, context)
            logger.debug(f"Context received: {context}")
            logger.debug(
                f"Sending to OpenAI model {self.settings.openai_model}: {messages}"
            )

            response = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=self.settings.max_tokens,
                timeout=timeout or self.settings.llm_timeout,
            )

            if response.choices:
                content = response.choices[0].message.content
                logger.debug(f"OpenAI response: {content}")
                logger.debug(f"OpenAI full response: {response}")
                if not content:
                    logger.warning(
                        "OpenAI returned empty content. Full response: "
                        f"{response.model_dump()}"
                    )
                return content
            else:
                logger.warning("OpenAI response has no choices")
                logger.warning(f"Full response: {response.model_dump()}")
                return None

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            logger.error(f"Failed request context: {context}")
            if messages:
                logger.error(f"Failed request messages: {messages}")
            return None

    async def generate_response_stream(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from OpenAI

        Args:
            message: User message to process
            context: Optional conversation context

        Yields:
            Response chunks as they arrive
        """
        if not self.client or not HAS_OPENAI:
            # Mock streaming response
            mock_response = f"Mock streaming response to: {message}"
            for word in mock_response.split():
                yield word + " "
            return

        try:
            messages = self._prepare_messages(message, context)

            stream = await self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,  # type: ignore[arg-type]
                max_completion_tokens=self.settings.max_tokens,
                stream=True,
            )

            async for chunk in stream:  # type: ignore[union-attr]
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    def _prepare_messages(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for OpenAI API

        Args:
            message: Current user message
            context: Optional conversation history

        Returns:
            List of messages formatted for OpenAI API
        """
        messages = []

        # Add system prompt
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. Respond conversationally "
                    "and concisely. Keep responses under 100 words unless more "
                    "detail is specifically requested. You are designed for voice "
                    "interaction, so avoid formatting like lists or code blocks "
                    "unless specifically asked."
                ),
            }
        )

        # Add conversation context if provided
        if context:
            # Filter out non-OpenAI fields like timestamp
            filtered_context = [
                {"role": msg["role"], "content": msg["content"]} for msg in context
            ]
            messages.extend(filtered_context)

        # Add current user message
        messages.append({"role": "user", "content": message})

        return messages

    def get_configuration(self) -> Dict[str, Any]:
        """Get current LLM configuration

        Returns:
            Dictionary with configuration details
        """
        return {
            "provider": "openai",
            "model_name": self.settings.openai_model,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
            "timeout": self.settings.llm_timeout,
            "has_openai": HAS_OPENAI,
            "client_initialized": self.client is not None,
            "api_key_provided": bool(self.settings.openai_api_key),
        }

    async def generate_with_tools(
        self,
        message: str,
        tools_manager: Any,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Generate response with tool calling capability

        Args:
            message: User message
            tools_manager: Available tools manager
            context: Conversation context

        Returns:
            Response text, potentially enhanced with tool results
        """
        # For now, just use basic response generation
        # Tool calling can be enhanced later with OpenAI function calls
        _ = tools_manager  # Acknowledge parameter to avoid linting error
        response = await self.generate_response(message, context)
        if not response:
            logger.warning(
                f"OpenAI generate_response returned None for message: {message[:50]}..."
            )
        return response or "I'm sorry, I couldn't process that request."

    async def close(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.close()
            self.client = None
