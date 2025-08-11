"""Custom lightweight LLM integration for Ollama without external dependencies"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    httpx = None  # type: ignore
    HAS_HTTPX = False

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class CustomLLMInterface:
    """Lightweight LLM interface using direct HTTP calls to Ollama"""

    def __init__(self, settings: Settings):
        """Initialize LLM interface

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.client = None
        self.base_url = settings.ollama_host.rstrip("/")
        self.model_name = settings.model_name

        # Initialize HTTP client if available
        if HAS_HTTPX and not settings.is_development():
            self.client = httpx.AsyncClient(timeout=settings.llm_timeout)

        logger.info("Custom LLM interface initialized")

    async def _test_connection(self) -> None:
        """Test connection to Ollama"""
        if not HAS_HTTPX:
            logger.warning("httpx not available - using mock responses")
            return

        try:
            if self.client is None:
                raise ConnectionError("No HTTP client available")
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                logger.info("Ollama connection successful")
            else:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}")

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
        if self.settings.is_development() or not HAS_HTTPX:
            # Return mock response for development
            return f"Mock response to: {message}"

        if not self.client:
            # If no client available, use mock mode
            return f"Mock response to: {message}"

        if timeout is None:
            timeout = self.settings.llm_timeout

        try:
            # Prepare messages
            messages = self._prepare_messages(message, context)

            # Make request to Ollama
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.settings.temperature,
                    "num_predict": self.settings.max_tokens,
                },
            }

            response = await self.client.post(
                f"{self.base_url}/api/chat", json=payload, timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                return str(result["message"]["content"])
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None

        except asyncio.TimeoutError:
            logger.error(f"LLM timeout after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

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
        if self.settings.is_development() or not HAS_HTTPX:
            # Mock streaming response
            mock_response = f"Mock streaming response to: {message}"
            for word in mock_response.split():
                yield word + " "
                await asyncio.sleep(0.01)
            return

        try:
            messages = self._prepare_messages(message, context)

            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": self.settings.temperature,
                    "num_predict": self.settings.max_tokens,
                },
            }

            if self.client is None:
                raise RuntimeError("No HTTP client available")
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.settings.llm_timeout,
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk_data = json.loads(line)
                            if "message" in chunk_data:
                                content = chunk_data["message"].get("content", "")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise

    def _prepare_messages(
        self, message: str, context: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM consumption

        Args:
            message: Current user message
            context: Optional conversation history

        Returns:
            List of messages formatted for Ollama API
        """
        messages = []

        # Add system prompt
        messages.append(
            {
                "role": "system",
                "content": (
                    "You are a helpful voice assistant. Respond conversationally "
                    "and concisely. Keep responses under 100 words unless more "
                    "detail is specifically requested."
                ),
            }
        )

        # Add conversation context if provided
        if context:
            messages.extend(context)

        # Add current user message
        messages.append({"role": "user", "content": message})

        return messages

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
        # Simple tool detection (could be enhanced with better NLP)
        response = ""
        tools_used = []

        message_lower = message.lower()

        # Check for tool triggers
        if any(word in message_lower for word in ["time", "clock", "what time"]):
            try:
                time_result = await tools_manager.execute_tool("time")
                tools_used.append(("time", time_result))
                response += f"Current time: {time_result}\n"
            except Exception as e:
                logger.error(f"Time tool failed: {e}")

        if any(word in message_lower for word in ["memory", "system", "performance"]):
            try:
                sys_result = await tools_manager.execute_tool(
                    "system_info", {"info_type": "memory"}
                )
                tools_used.append(("system_info", sys_result))
                response += f"System status: {sys_result}\n"
            except Exception as e:
                logger.error(f"System tool failed: {e}")

        if any(word in message_lower for word in ["weather", "temperature"]):
            # Extract location if possible (simple extraction)
            location = "current location"
            words = message_lower.split()
            for i, word in enumerate(words):
                if word in ["in", "for", "at"] and i + 1 < len(words):
                    location = words[i + 1]
                    break

            try:
                weather_result = await tools_manager.execute_tool(
                    "weather", {"location": location}
                )
                tools_used.append(("weather", weather_result))
                response += f"{weather_result}\n"
            except Exception as e:
                logger.error(f"Weather tool failed: {e}")

        # If tools were used, enhance the message context
        if tools_used:
            enhanced_message = f"{message}\n\nTool results:\n"
            for tool_name, result in tools_used:
                enhanced_message += f"- {tool_name}: {result}\n"
            enhanced_message += (
                "\nPlease incorporate this information in your response."
            )
        else:
            enhanced_message = message

        # Generate LLM response
        llm_response = await self.generate_response(enhanced_message, context)

        if tools_used and llm_response:
            # Combine tool results with LLM response
            return f"{response.strip()}\n\n{llm_response}"
        else:
            return llm_response or "I'm sorry, I couldn't process that request."

    def get_configuration(self) -> Dict[str, Any]:
        """Get current LLM configuration

        Returns:
            Dictionary with configuration details
        """
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "timeout": self.settings.llm_timeout,
            "has_httpx": HAS_HTTPX,
            "client_initialized": self.client is not None,
        }

    async def close(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.aclose()
            self.client = None
