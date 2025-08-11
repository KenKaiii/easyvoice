"""LLM integration with Ollama and timeout handling"""

import asyncio
import concurrent.futures
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class LLMInterface:
    """Interface for LLM communication with timeout and error handling"""

    def __init__(self, settings: Settings):
        """Initialize LLM interface

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.client = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Initialize connection if not in test mode
        if not settings.is_development():
            asyncio.create_task(self._initialize_client())

    async def _initialize_client(self) -> None:
        """Initialize Ollama client connection"""
        try:
            # Import here to handle missing dependency gracefully
            from langchain_ollama import ChatOllama  # type: ignore[import-not-found]

            self.client = ChatOllama(
                model=self.settings.model_name,
                base_url=self.settings.ollama_host,
                temperature=self.settings.temperature,
            )

            # Test connection
            await self._test_connection()
            logger.info("LLM client initialized successfully")

        except ImportError:
            logger.error(
                "langchain-ollama not available - install with: "
                "pip install langchain-ollama"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    async def _test_connection(self) -> None:
        """Test connection to Ollama"""
        try:
            # Simple test message
            test_response = await self.generate_response("Test", timeout=5)
            if test_response is None:
                raise ConnectionError("LLM test connection failed")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.settings.ollama_host}: {e}"
            )

    async def generate_response(
        self,
        message: str,
        context: Optional[List[Dict[str, str]]] = None,
        timeout: Optional[int] = None,
    ) -> Optional[str]:
        """Generate response from LLM

        Args:
            message: User message to process
            context: Optional conversation context
            timeout: Optional timeout override

        Returns:
            LLM response text or None if failed
        """
        if timeout is None:
            timeout = self.settings.llm_timeout

        try:
            # Prepare messages for LLM
            messages = self._prepare_messages(message, context)

            # Generate response with timeout
            response_task = asyncio.create_task(self._generate_response_async(messages))

            response = await asyncio.wait_for(response_task, timeout=timeout)

            logger.info(f"LLM response generated: {len(response)} characters")
            return response

        except asyncio.TimeoutError:
            logger.error(f"LLM timeout after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _prepare_messages(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM consumption

        Args:
            message: Current user message
            context: Optional conversation history

        Returns:
            List of messages formatted for LLM
        """
        messages = []

        # Add system prompt
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. "
                    "Respond conversationally and concisely."
                ),
            }
        )

        # Add conversation context if provided
        if context:
            messages.extend(context)

        # Add current user message
        messages.append({"role": "user", "content": message})

        return messages

    async def _generate_response_async(self, messages: List[Dict[str, str]]) -> str:
        """Generate response asynchronously

        Args:
            messages: Messages to send to LLM

        Returns:
            Generated response text
        """
        if self.client is None:
            # For testing, return a mock response
            if self.settings.is_development():
                return f"Mock response to: {messages[-1]['content']}"
            raise RuntimeError("LLM client not initialized")

        try:  # type: ignore[unreachable]
            # Use langchain-ollama to generate response
            response = await self.client.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"LLM client error: {e}")
            raise

    async def generate_response_stream(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM

        Args:
            message: User message to process
            context: Optional conversation context

        Yields:
            Response chunks as they arrive
        """
        messages = self._prepare_messages(message, context)

        if self.client is None:
            # For testing, yield mock chunks
            if self.settings.is_development():
                mock_response = f"Mock streaming response to: {message}"
                for word in mock_response.split():
                    yield word + " "
                    await asyncio.sleep(0.01)  # Simulate streaming delay
                return
            raise RuntimeError("LLM client not initialized")

        try:  # type: ignore[unreachable]
            # Use langchain-ollama streaming
            async for chunk in self.client.astream(messages):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise

    def get_configuration(self) -> Dict[str, Any]:
        """Get current LLM configuration

        Returns:
            Dictionary with configuration details
        """
        return {
            "model_name": self.settings.model_name,
            "base_url": self.settings.ollama_host,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.llm_timeout,
            "client_initialized": self.client is not None,
        }

    async def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        self.client = None
