"""OpenAI Whisper API for speech-to-text"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from openai import AsyncOpenAI
    from openai._types import NOT_GIVEN, NotGiven

    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None  # type: ignore
    NotGiven = None  # type: ignore
    NOT_GIVEN = None  # type: ignore
    HAS_OPENAI = False

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class OpenAIWhisperSTT:
    """OpenAI Whisper API speech-to-text processor"""

    def __init__(self, settings: Settings):
        """Initialize OpenAI Whisper STT

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.client = None

        if not HAS_OPENAI:
            logger.warning("OpenAI package not available - STT disabled")
            return

        if not settings.openai_api_key:
            logger.warning("No OpenAI API key provided - STT disabled")
            return

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        logger.info("OpenAI Whisper STT initialized")

    async def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """Transcribe audio data using OpenAI Whisper API

        Args:
            audio_data: Audio data as numpy array (float32, 16kHz expected)

        Returns:
            Transcribed text or None if failed
        """
        if not self.client or not HAS_OPENAI:
            return None

        if len(audio_data) == 0:
            return None

        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)

                # Save audio to file (Whisper API expects file input)
                sf.write(tmp_path, audio_data, self.settings.sample_rate, format="WAV")

                # Transcribe using OpenAI Whisper API
                with open(tmp_path, "rb") as audio_file:
                    # Handle optional language parameter
                    language_param: NotGiven | str = NOT_GIVEN
                    if self.settings.whisper_language:
                        language_param = self.settings.whisper_language

                    response = await self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language_param,
                    )

                # Clean up temp file
                tmp_path.unlink()

                # Check if response exists and has text attribute
                if not response or not hasattr(response, "text"):
                    return None

                text = response.text.strip() if response.text else None

                # Filter out common noise/silence transcriptions
                if text and text.lower() in ["you", "thank you", "thanks", "", " "]:
                    return None

                return text

        except Exception as e:
            logger.error(f"OpenAI Whisper transcription failed: {e}")
            return None

    async def close(self) -> None:
        """Clean up resources"""
        if self.client:
            await self.client.close()
            self.client = None
