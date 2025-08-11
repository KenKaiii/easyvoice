"""Speech-to-Text using OpenAI Whisper with timeout handling"""

import asyncio
import concurrent.futures
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import whisper

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class WhisperSTT:
    """OpenAI Whisper Speech-to-Text processor with timeout handling"""

    def __init__(self, settings: Settings):
        """Initialize Whisper STT

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.model: Optional[whisper.Whisper] = None
        self.model_loaded = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Preload model if not in test mode
        if not settings.is_development():
            asyncio.create_task(self.load_model())

    async def load_model(self) -> None:
        """Load Whisper model asynchronously"""
        if self.model_loaded:
            return

        try:
            logger.info(f"Loading Whisper model: {self.settings.whisper_model}")

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self.executor, self._load_model_sync
            )

            self.model_loaded = True
            logger.info("Whisper model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def _load_model_sync(self) -> whisper.Whisper:
        """Load Whisper model synchronously (for thread pool)"""
        # Set device preference
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return whisper.load_model(self.settings.whisper_model, device=device)

    async def transcribe_audio_data(
        self, audio_data: np.ndarray, language: Optional[str] = None
    ) -> Optional[str]:
        """Transcribe numpy audio data to text

        Args:
            audio_data: Audio data as numpy array (float32, sample_rate Hz)
            language: Optional language code (e.g., 'en', 'es')

        Returns:
            Transcribed text or None if failed
        """
        if not self.model_loaded:
            await self.load_model()

        if self.model is None:
            logger.error("Whisper model not loaded")
            return None

        try:
            # Prepare transcription options
            options = self.settings.get_whisper_kwargs()
            if language:
                options["language"] = language

            # Run transcription in thread pool with timeout
            loop = asyncio.get_event_loop()

            transcription_task = loop.run_in_executor(
                self.executor, self._transcribe_sync, audio_data, options
            )

            # Apply timeout
            result = await asyncio.wait_for(
                transcription_task, timeout=self.settings.stt_timeout
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"STT timeout after {self.settings.stt_timeout}s")
            return None
        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            return None

    def _transcribe_sync(
        self, audio_data: np.ndarray, options: Dict[str, Any]
    ) -> Optional[str]:
        """Synchronous transcription (for thread pool)

        Args:
            audio_data: Audio data as numpy array
            options: Whisper transcription options

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Ensure audio is in correct format for Whisper
            # Whisper expects float32 audio normalized to [-1, 1]
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio if needed
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val

            # Transcribe using Whisper
            if self.model is None:
                return None
            result = self.model.transcribe(audio_data, **options)

            text = str(result.get("text", "")).strip()

            if text:
                logger.info(f"STT result: '{text}'")
                return text
            else:
                logger.warning("STT returned empty result")
                return None

        except Exception as e:
            logger.error(f"Synchronous STT failed: {e}")
            return None

    async def transcribe_file(self, audio_file_path: str) -> Optional[str]:
        """Transcribe audio file to text

        Args:
            audio_file_path: Path to audio file

        Returns:
            Transcribed text or None if failed
        """
        try:
            # Load audio file
            audio_data, sample_rate = sf.read(audio_file_path)

            # Resample if needed (Whisper works best at 16kHz)
            if sample_rate != 16000:
                import librosa

                audio_data = librosa.resample(
                    audio_data, orig_sr=sample_rate, target_sr=16000
                )

            return await self.transcribe_audio_data(audio_data)

        except Exception as e:
            logger.error(f"Failed to transcribe file {audio_file_path}: {e}")
            return None

    async def save_and_transcribe(self, audio_data: np.ndarray) -> Optional[str]:
        """Save audio data to temporary file and transcribe

        This method is useful when you need to save audio for debugging
        or when working with audio data that needs to be preprocessed.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text or None if failed
        """
        temp_file = None
        try:
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_file = f.name

            # Save audio data
            sf.write(temp_file, audio_data, self.settings.sample_rate, format="WAV")

            # Transcribe the file
            result = await self.transcribe_file(temp_file)

            return result

        except Exception as e:
            logger.error(f"Save and transcribe failed: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_file and Path(temp_file).exists():
                Path(temp_file).unlink()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        if not self.model_loaded or self.model is None:
            return {"status": "not_loaded"}

        return {
            "status": "loaded",
            "model_name": self.settings.whisper_model,
            "device": next(self.model.parameters()).device.type,
            "is_multilingual": self.model.is_multilingual,
            "languages": (
                list(self.model.tokenizer.language_tokens.keys())
                if hasattr(self.model.tokenizer, "language_tokens")
                else []
            ),
        }

    async def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        # Clear model from memory
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False

            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Testing and utility functions
async def test_speech_recognition(
    settings: Settings, duration: float = 3.0, verbose: bool = False
) -> Optional[str]:
    """Test speech recognition pipeline

    Args:
        settings: EasyVoice settings
        duration: Recording duration in seconds
        verbose: Show detailed output

    Returns:
        Transcribed text or None if failed
    """
    try:
        from easyvoice.audio.input import AudioInput

        # Initialize components
        audio_input = AudioInput(settings)
        stt = WhisperSTT(settings)

        if verbose:
            print(f"Recording for {duration} seconds...")

        # Record audio
        audio_data = await audio_input.record_for_duration(duration)

        if len(audio_data) == 0:
            if verbose:
                print("No audio recorded")
            return None

        if verbose:
            print("Processing speech recognition...")

        # Transcribe audio
        text = await stt.transcribe_audio_data(audio_data)

        # Clean up
        await stt.close()

        return text

    except Exception as e:
        if verbose:
            print(f"Speech recognition test failed: {e}")
        return None


async def benchmark_stt_performance(
    settings: Settings, test_duration: float = 5.0
) -> Dict[str, Any]:
    """Benchmark STT performance

    Args:
        settings: EasyVoice settings
        test_duration: Audio duration for testing

    Returns:
        Performance metrics
    """
    try:
        from easyvoice.audio.input import AudioInput

        # Record test audio
        audio_input = AudioInput(settings)
        audio_data = await audio_input.record_for_duration(test_duration)

        if len(audio_data) == 0:
            return {"error": "No audio recorded"}

        # Initialize STT
        stt = WhisperSTT(settings)

        # Measure model loading time
        load_start = time.time()
        await stt.load_model()
        load_time = time.time() - load_start

        # Measure transcription time
        transcribe_start = time.time()
        result = await stt.transcribe_audio_data(audio_data)
        transcribe_time = time.time() - transcribe_start

        # Calculate real-time factor
        real_time_factor = transcribe_time / test_duration

        # Get model info
        model_info = stt.get_model_info()

        # Clean up
        await stt.close()

        return {
            "success": result is not None,
            "transcription": result,
            "audio_duration": test_duration,
            "load_time": load_time,
            "transcribe_time": transcribe_time,
            "real_time_factor": real_time_factor,
            "model_info": model_info,
            "audio_samples": len(audio_data),
        }

    except Exception as e:
        return {"error": str(e)}
