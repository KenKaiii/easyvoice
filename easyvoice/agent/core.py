"""Core voice agent orchestrating all components"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from easyvoice.agent.llm_custom import CustomLLMInterface
from easyvoice.agent.memory import ConversationMemory
from easyvoice.agent.tools import ToolsManager
from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class VoiceAgent:
    """Main voice agent coordinating speech, LLM, and memory"""

    def __init__(self, settings: Settings):
        """Initialize voice agent

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.memory = ConversationMemory(settings)
        self.llm = CustomLLMInterface(settings)
        self.tools = ToolsManager(settings)

        # Component availability tracking
        self.stt_available = True
        self.audio_available = True

        # Session management
        self.session_active = False
        self.session_start_time: Optional[float] = None
        self.notifications: List[str] = []

        # State tracking
        self.last_transcription: Optional[str] = None
        self.last_audio_output: Optional[str] = None
        self.last_tools_used: List[Dict[str, Any]] = []

        logger.info("VoiceAgent initialized")

    def is_ready(self) -> bool:
        """Check if all components are ready

        Returns:
            True if agent is ready for conversations
        """
        return (
            self.memory is not None
            and self.llm is not None
            and self.settings is not None
        )

    def is_listening(self) -> bool:
        """Check if agent is in listening mode

        Returns:
            True if agent is listening for input
        """
        return self.session_active and self.audio_available

    def is_session_active(self) -> bool:
        """Check if conversation session is active

        Returns:
            True if session is active
        """
        return self.session_active

    def start_conversation_session(self) -> None:
        """Start a new conversation session"""
        self.session_active = True
        self.session_start_time = time.time()
        self.notifications.clear()
        logger.info("Conversation session started")

    async def process_voice_input(self, spoken_text: str) -> str:
        """Process voice input through the complete pipeline

        Args:
            spoken_text: Transcribed speech text

        Returns:
            Agent's response text

        Raises:
            Exception: If STT system is not available
        """
        if not self.stt_available:
            raise Exception("Speech recognition system is not available")

        # Store transcription
        self.last_transcription = spoken_text

        # Add user message to memory
        self.memory.add_message("user", spoken_text)

        # Get conversation context
        context = self.memory.get_context_for_llm()

        # Generate response using LLM with tools
        response = await self.llm.generate_with_tools(
            spoken_text,
            self.tools,
            context=context[:-1] if context else None,  # Exclude system prompt
        )

        # Add assistant response to memory
        self.memory.add_message("assistant", response)

        # Generate audio output (mocked for now)
        await self._generate_audio_output(response)

        return response

    async def process_text_input(self, text: str) -> str:
        """Process text input (fallback when voice not available)

        Args:
            text: Text message to process

        Returns:
            Agent's response text
        """
        # Add user message to memory
        self.memory.add_message("user", text)

        # Check for tool usage (simple keyword detection)
        await self._check_for_tools(text)

        # Get conversation context
        context_messages = self.memory.get_recent_messages(limit=10)

        # Generate response using LLM with tools
        response = await self.llm.generate_with_tools(
            text,
            self.tools,
            context=context_messages[:-1] if context_messages else None,
        )

        # Add assistant response to memory
        self.memory.add_message("assistant", response)

        return response

    async def _generate_audio_output(self, text: str) -> None:
        """Generate audio output for response

        Args:
            text: Text to convert to speech
        """
        # For now, just store that audio was generated
        self.last_audio_output = f"Audio for: {text[:50]}..."
        logger.debug(f"Generated audio output: {len(text)} characters")

    async def _check_for_tools(self, text: str) -> List[Dict[str, Any]]:
        """Check if text requires tool usage

        Args:
            text: Input text to analyze

        Returns:
            List of tools that should be used
        """
        tools_used = []
        text_lower = text.lower()

        # Simple keyword-based tool detection
        if any(word in text_lower for word in ["time", "clock", "what time"]):
            tools_used.append(
                {
                    "name": "time_tool",
                    "description": "Get current time",
                    "result": datetime.now().strftime("%I:%M %p"),
                }
            )

        if any(word in text_lower for word in ["weather", "temperature", "forecast"]):
            tools_used.append(
                {
                    "name": "weather_tool",
                    "description": "Get weather information",
                    "result": "Weather information would be retrieved here",
                }
            )

        self.last_tools_used = tools_used
        return tools_used

    async def check_session_timeout(self) -> bool:
        """Check if session has timed out

        Returns:
            True if session timed out
        """
        if not self.session_active or self.session_start_time is None:
            return False

        elapsed = time.time() - self.session_start_time
        if elapsed > self.settings.session_timeout:
            self.session_active = False
            self.notifications.append("Session timed out due to inactivity")
            logger.info("Session timed out")
            return True

        return False

    # Getter methods for test verification
    def get_last_transcription(self) -> Optional[str]:
        """Get the last speech transcription"""
        return self.last_transcription

    def get_last_audio_output(self) -> Optional[str]:
        """Get the last audio output information"""
        return self.last_audio_output

    def get_memory(self) -> ConversationMemory:
        """Get the conversation memory instance"""
        return self.memory

    def get_notifications(self) -> List[str]:
        """Get current notifications"""
        return self.notifications.copy()

    def get_last_tools_used(self) -> List[Dict[str, Any]]:
        """Get tools used in last interaction"""
        return self.last_tools_used.copy()

    async def start_conversation(
        self, push_to_talk: bool = False, verbose: bool = False
    ) -> None:
        """Start interactive conversation mode

        Args:
            push_to_talk: Use push-to-talk mode
            verbose: Enable verbose output
        """
        self.start_conversation_session()

        logger.info("Starting conversation mode")
        if verbose:
            logger.info(f"Push-to-talk: {push_to_talk}")

        # This would be the main conversation loop
        # For now, just maintain session state
        while self.session_active:
            # Check for timeout
            if await self.check_session_timeout():
                break

            # In real implementation, this would:
            # 1. Listen for audio input
            # 2. Process through STT
            # 3. Generate response
            # 4. Play through TTS
            # 5. Handle interruptions

            await asyncio.sleep(0.1)  # Prevent tight loop

    async def process_question(
        self, question: str, speak_response: bool = False, save_to_history: bool = True
    ) -> str:
        """Process a single question

        Args:
            question: Question text
            speak_response: Whether to speak the response
            save_to_history: Whether to save to conversation history

        Returns:
            Agent's response
        """
        # Generate response
        response = await self.llm.generate_response(question)

        if response is None:
            response = "I'm sorry, I couldn't process that question."

        # Save to history if requested
        if save_to_history:
            self.memory.add_message("user", question)
            self.memory.add_message("assistant", response)

        # Generate speech if requested
        if speak_response:
            await self._generate_audio_output(response)

        return response
