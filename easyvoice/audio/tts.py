"""Text-to-Speech using KittenTTS with timeout handling"""

import asyncio
import concurrent.futures
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import sounddevice as sd  # type: ignore[import-untyped]
import soundfile as sf  # type: ignore[import-untyped]

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class KittenTTS:
    """KittenTTS Text-to-Speech processor with timeout handling"""

    def __init__(self, settings: Settings):
        """Initialize KittenTTS

        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.model = None
        self.model_loaded = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Preload model if not in test mode
        if not settings.is_development():
            asyncio.create_task(self.load_model())

    async def load_model(self) -> None:
        """Load KittenTTS model asynchronously"""
        if self.model_loaded:
            return

        try:
            logger.info(f"Loading KittenTTS model: {self.settings.tts_model}")

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            model_result = await loop.run_in_executor(
                self.executor, self._load_model_sync
            )
            self.model = model_result

            self.model_loaded = True
            logger.info("KittenTTS model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load KittenTTS model: {e}")
            raise

    def _load_model_sync(self) -> Any:
        """Load KittenTTS model synchronously (for thread pool)"""
        try:
            # Import KittenTTS here to avoid import errors if not installed
            from kittentts import KittenTTS as KittenTTSModel  # type: ignore

            # Let KittenTTS handle model download automatically
            if self.settings.tts_model == "auto":
                logger.info("Loading KittenTTS with auto model download")
                return KittenTTSModel()  # No model path = auto download
            else:
                logger.info(f"Loading KittenTTS model: {self.settings.tts_model}")
                return KittenTTSModel(self.settings.tts_model)

        except ImportError:
            logger.error("KittenTTS not installed. Install with: pip install kittentts")
            raise
        except Exception as e:
            logger.error(f"Failed to load KittenTTS model: {e}")
            raise

    async def synthesize_text(
        self, text: str, voice: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """Synthesize text to audio

        Args:
            text: Text to synthesize
            voice: Voice ID (0-7), uses settings default if None

        Returns:
            Audio data as numpy array (float32, 24kHz) or None if failed
        """
        if not self.model_loaded:
            await self.load_model()

        if self.model is None:
            logger.error("KittenTTS model not loaded")
            return None

        if not text.strip():  # type: ignore[unreachable]
            logger.warning("Empty text provided for synthesis")
            return None

        try:
            # Use specified voice or settings default
            voice_id = voice if voice is not None else self.settings.tts_voice

            # Run synthesis in thread pool with timeout
            loop = asyncio.get_event_loop()

            synthesis_task = loop.run_in_executor(
                self.executor, self._synthesize_sync, text, voice_id
            )

            # Apply timeout
            audio_data = await asyncio.wait_for(
                synthesis_task, timeout=self.settings.tts_timeout
            )

            return audio_data

        except asyncio.TimeoutError:
            logger.error(f"TTS timeout after {self.settings.tts_timeout}s")
            return None
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None

    def _synthesize_sync(self, text: str, voice_id: int) -> Optional[np.ndarray]:
        """Synchronous synthesis (for thread pool)

        Args:
            text: Text to synthesize
            voice_id: Voice ID (0-7)

        Returns:
            Audio data as numpy array or None if failed
        """
        try:
            if self.model is None:
                logger.error("KittenTTS model not loaded")
                return None

            # Preprocess text to avoid ONNX Runtime issues
            processed_text = self._preprocess_text_for_onnx(text)

            # Map voice ID to KittenTTS voice names
            voice_map: Dict[int, str] = {  # type: ignore[unreachable]
                0: "expr-voice-2-m",
                1: "expr-voice-2-f",
                2: "expr-voice-3-m",
                3: "expr-voice-3-f",
                4: "expr-voice-4-m",
                5: "expr-voice-4-f",
                6: "expr-voice-5-m",
                7: "expr-voice-5-f",
            }

            voice_name = voice_map.get(voice_id, "expr-voice-2-m")
            logger.debug(f"Using KittenTTS voice {voice_id} -> {voice_name}")

            # Generate audio data with error handling for ONNX issues
            try:
                audio_data = self.model.generate(processed_text, voice=voice_name)
            except Exception as onnx_error:
                # Handle ONNX Runtime Expand node errors
                if "Expand node" in str(onnx_error) or "invalid expand shape" in str(onnx_error):
                    logger.warning(f"ONNX Expand error, trying text chunking: {onnx_error}")
                    audio_data = self._synthesize_with_chunking(processed_text, voice_name)
                else:
                    raise onnx_error

            if audio_data is None or len(audio_data) == 0:
                logger.warning("KittenTTS returned empty audio")
                return None

            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Apply speed adjustment if needed
            if abs(self.settings.tts_speed - 1.0) > 0.001:
                audio_data = self._adjust_speed(audio_data, self.settings.tts_speed)

            logger.info(
                f"KittenTTS generated {len(audio_data)} samples "
                f"for text: '{text[:50]}...'"
            )
            return audio_data

        except Exception as e:
            logger.error(f"KittenTTS synthesis failed: {e}")
            return None

    def _preprocess_text_for_onnx(self, text: str) -> str:
        """Preprocess text to avoid ONNX Runtime issues
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Processed text that's more compatible with ONNX
        """
        # Limit text length to avoid dynamic shape issues
        max_length = 200
        if len(text) > max_length:
            # Split at sentence boundaries if possible
            sentences = text.split('. ')
            if len(sentences) > 1 and len(sentences[0]) <= max_length:
                text = sentences[0] + '.'
            else:
                text = text[:max_length].rstrip() + '.'
        
        # Remove problematic characters that might cause shape issues
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Ensure text isn't empty after preprocessing
        if not text.strip():
            text = "Hello."
            
        return text
    
    def _synthesize_with_chunking(self, text: str, voice_name: str) -> Optional[np.ndarray]:
        """Synthesize text using chunking to avoid ONNX errors
        
        Args:
            text: Text to synthesize
            voice_name: Voice name to use
            
        Returns:
            Combined audio data or None if failed
        """
        try:
            # Split text into smaller chunks
            max_chunk_size = 50
            words = text.split()
            
            if len(words) <= max_chunk_size:
                # Text is already small, try direct synthesis with minimal text
                minimal_text = " ".join(words[:10]) if len(words) > 10 else text
                return self.model.generate(minimal_text, voice=voice_name)
            
            # Process in chunks
            audio_chunks = []
            for i in range(0, len(words), max_chunk_size):
                chunk_words = words[i:i + max_chunk_size]
                chunk_text = " ".join(chunk_words)
                
                try:
                    chunk_audio = self.model.generate(chunk_text, voice=voice_name)
                    if chunk_audio is not None and len(chunk_audio) > 0:
                        audio_chunks.append(chunk_audio)
                except Exception as chunk_error:
                    logger.warning(f"Chunk synthesis failed, skipping: {chunk_error}")
                    continue
            
            if not audio_chunks:
                logger.error("All chunks failed to synthesize")
                return None
                
            # Concatenate audio chunks
            combined_audio = np.concatenate(audio_chunks)
            logger.info(f"Successfully synthesized {len(audio_chunks)} chunks")
            return combined_audio
            
        except Exception as e:
            logger.error(f"Chunked synthesis failed: {e}")
            return None

    def _adjust_speed(self, audio_data: np.ndarray, speed_factor: float) -> np.ndarray:
        """Adjust audio playback speed (simplified without librosa)

        Args:
            audio_data: Input audio data
            speed_factor: Speed multiplier (1.0 = normal, 2.0 = 2x faster)

        Returns:
            Speed-adjusted audio data (simplified - just return original for now)
        """
        # Simple speed adjustment by skipping samples (basic but works)
        if abs(speed_factor - 1.0) < 0.001:
            return audio_data
            
        try:
            # Basic speed adjustment by resampling every Nth sample
            if speed_factor > 1.0:
                # Faster: skip samples
                step = int(speed_factor)
                return audio_data[::step]
            else:
                # Slower: duplicate samples (simple approach)
                repeat = int(1.0 / speed_factor)
                return np.repeat(audio_data, repeat)
        except Exception as e:
            logger.warning(f"Speed adjustment failed: {e}")
            return audio_data

    async def play_audio(self, audio_data: np.ndarray, wait: bool = True) -> bool:
        """Play audio data through speakers

        Args:
            audio_data: Audio data to play (float32, 24kHz)
            wait: Whether to wait for playback to complete

        Returns:
            True if playback started successfully, False otherwise
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                logger.warning("No audio data to play")
                return False

            # KittenTTS outputs at 24kHz - use it directly for best quality
            sample_rate = 24000

            # Play audio
            sd.play(audio_data, samplerate=sample_rate)

            if wait:
                sd.wait()  # Wait for playback to complete

            logger.info("Audio playback completed")
            return True

        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False

    async def synthesize_and_play(
        self, text: str, voice: Optional[int] = None, wait: bool = True
    ) -> bool:
        """Synthesize text and play audio

        Args:
            text: Text to synthesize and play
            voice: Voice ID (0-7), uses settings default if None
            wait: Whether to wait for playback to complete

        Returns:
            True if synthesis and playback successful, False otherwise
        """
        try:
            # Synthesize text to audio
            audio_data = await self.synthesize_text(text, voice)

            if audio_data is None:
                return False

            # Play the audio
            return await self.play_audio(audio_data, wait)

        except Exception as e:
            logger.error(f"Synthesize and play failed: {e}")
            return False

    async def save_audio(self, audio_data: np.ndarray, output_path: str) -> bool:
        """Save audio data to file

        Args:
            audio_data: Audio data to save (float32, 24kHz)
            output_path: Path to save audio file

        Returns:
            True if save successful, False otherwise
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                logger.warning("No audio data to save")
                return False

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save audio (KittenTTS outputs at 24kHz)
            sf.write(output_path, audio_data, 24000, format="WAV")

            logger.info(f"Audio saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save audio to {output_path}: {e}")
            return False

    def get_available_voices(self) -> Dict[int, str]:
        """Get information about available voices

        Returns:
            Dictionary mapping voice IDs to descriptions
        """
        # KittenTTS has 8 voices (4 male, 4 female)
        return {
            0: "Male Voice 1",
            1: "Male Voice 2",
            2: "Male Voice 3",
            3: "Male Voice 4",
            4: "Female Voice 1",
            5: "Female Voice 2",
            6: "Female Voice 3",
            7: "Female Voice 4",
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model

        Returns:
            Dictionary with model information
        """
        if not self.model_loaded or self.model is None:
            return {"status": "not_loaded"}

        return {  # type: ignore[unreachable]
            "status": "loaded",
            "model_name": self.settings.tts_model,
            "sample_rate": 24000,  # KittenTTS output sample rate
            "available_voices": self.get_available_voices(),
            "current_voice": self.settings.tts_voice,
            "speed": self.settings.tts_speed,
        }

    async def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

        # Clear model from memory
        if self.model is not None:
            del self.model  # type: ignore[unreachable]
            self.model = None
            self.model_loaded = False


# Testing and utility functions
async def test_text_to_speech(
    settings: Settings,
    text: str = "Hello, this is a test of the text to speech system.",
    verbose: bool = False,
) -> bool:
    """Test text-to-speech pipeline

    Args:
        settings: EasyVoice settings
        text: Text to synthesize
        verbose: Show detailed output

    Returns:
        True if TTS test successful, False otherwise
    """
    try:
        tts = KittenTTS(settings)

        if verbose:
            print(f"Synthesizing: '{text}'")

        # Test synthesis
        audio_data = await tts.synthesize_text(text)

        if audio_data is None:
            if verbose:
                print("TTS synthesis failed")
            return False

        if verbose:
            print(f"Generated {len(audio_data)} audio samples")
            print("Playing audio...")

        # Test playback
        success = await tts.play_audio(audio_data, wait=True)

        # Clean up
        await tts.close()

        if verbose and success:
            print("TTS test completed successfully")

        return success

    except Exception as e:
        if verbose:
            print(f"TTS test failed: {e}")
        return False


async def _test_single_voice(
    tts: "KittenTTS", voice_id: int, voice_name: str, text: str, verbose: bool
) -> bool:
    """Test a single voice"""
    try:
        if verbose:
            print(f"Testing {voice_name} (ID: {voice_id})")

        audio_data = await tts.synthesize_text(text, voice=voice_id)
        success = audio_data is not None

        if verbose:
            status = "✓" if success else "✗"
            print(f"  {status} {voice_name}")

        return success
    except Exception as e:
        if verbose:
            print(f"  ✗ {voice_name}: {e}")
        return False


async def test_all_voices(
    settings: Settings, text: str = "Testing voice", verbose: bool = False
) -> Dict[int, bool]:
    """Test all available TTS voices

    Args:
        settings: EasyVoice settings
        text: Text to test with each voice
        verbose: Show detailed output

    Returns:
        Dictionary mapping voice IDs to success status
    """
    try:
        tts = KittenTTS(settings)
        voices = tts.get_available_voices()

        results = {}
        for voice_id, voice_name in voices.items():
            results[voice_id] = await _test_single_voice(
                tts, voice_id, voice_name, text, verbose
            )

        await tts.close()
        return results

    except Exception as e:
        if verbose:
            print(f"Voice testing failed: {e}")
        return {}


async def benchmark_tts_performance(
    settings: Settings,
    test_text: str = "This is a performance test of the text to speech system.",
) -> Dict[str, Any]:
    """Benchmark TTS performance

    Args:
        settings: EasyVoice settings
        test_text: Text to use for benchmarking

    Returns:
        Performance metrics
    """
    try:
        tts = KittenTTS(settings)

        # Measure model loading time
        load_start = time.time()
        await tts.load_model()
        load_time = time.time() - load_start

        # Measure synthesis time
        synthesis_times = []

        for _ in range(3):  # Test 3 times for average
            synthesis_start = time.time()
            audio_data = await tts.synthesize_text(test_text)
            synthesis_time = time.time() - synthesis_start

            if audio_data is not None:
                synthesis_times.append(synthesis_time)

        # Calculate metrics
        avg_synthesis_time = (
            sum(synthesis_times) / len(synthesis_times) if synthesis_times else 0
        )

        # Estimate audio duration (assuming 24kHz output)
        if audio_data is not None:
            audio_duration = len(audio_data) / 24000
            real_time_factor = (
                avg_synthesis_time / audio_duration if audio_duration > 0 else 0
            )
        else:
            audio_duration = 0
            real_time_factor = 0

        # Get model info
        model_info = tts.get_model_info()

        # Clean up
        await tts.close()

        return {
            "success": len(synthesis_times) > 0,
            "text_length": len(test_text),
            "load_time": load_time,
            "avg_synthesis_time": avg_synthesis_time,
            "audio_duration": audio_duration,
            "real_time_factor": real_time_factor,
            "model_info": model_info,
            "test_runs": len(synthesis_times),
        }

    except Exception as e:
        return {"error": str(e)}
