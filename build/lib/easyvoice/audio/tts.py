"""Text-to-Speech using KittenTTS with timeout handling"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
import concurrent.futures

import numpy as np
import sounddevice as sd
import soundfile as sf

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
            self.model = await loop.run_in_executor(
                self.executor,
                self._load_model_sync
            )
            
            self.model_loaded = True
            logger.info("KittenTTS model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load KittenTTS model: {e}")
            raise
    
    def _load_model_sync(self):
        """Load KittenTTS model synchronously (for thread pool)"""
        try:
            # Import KittenTTS here to avoid import errors if not installed
            from kittentts import KittenTTS as KittenTTSModel
            
            return KittenTTSModel(self.settings.tts_model)
            
        except ImportError:
            logger.error("KittenTTS not installed. Install with: pip install kittentts")
            raise
        except Exception as e:
            logger.error(f"Failed to load KittenTTS model: {e}")
            raise
    
    async def synthesize_text(self, text: str, voice: Optional[int] = None) -> Optional[np.ndarray]:
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
        
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return None
        
        try:
            # Use specified voice or settings default
            voice_id = voice if voice is not None else self.settings.tts_voice
            
            # Run synthesis in thread pool with timeout
            loop = asyncio.get_event_loop()
            
            synthesis_task = loop.run_in_executor(
                self.executor,
                self._synthesize_sync,
                text,
                voice_id
            )
            
            # Apply timeout
            audio_data = await asyncio.wait_for(
                synthesis_task,
                timeout=self.settings.tts_timeout
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
            # Generate audio using KittenTTS
            # KittenTTS returns audio at 24kHz sample rate
            audio_data = self.model.generate(text, voice=voice_id)
            
            if audio_data is None or len(audio_data) == 0:
                logger.warning("TTS returned empty audio")
                return None
            
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Apply speed adjustment if needed
            if self.settings.tts_speed != 1.0:
                audio_data = self._adjust_speed(audio_data, self.settings.tts_speed)
            
            logger.info(f"TTS generated {len(audio_data)} samples for text: '{text[:50]}...'")
            return audio_data
            
        except Exception as e:
            logger.error(f"Synchronous TTS failed: {e}")
            return None
    
    def _adjust_speed(self, audio_data: np.ndarray, speed_factor: float) -> np.ndarray:
        """Adjust audio playback speed
        
        Args:
            audio_data: Input audio data
            speed_factor: Speed multiplier (1.0 = normal, 2.0 = 2x faster)
            
        Returns:
            Speed-adjusted audio data
        """
        try:
            import librosa
            return librosa.effects.time_stretch(audio_data, rate=speed_factor)
        except ImportError:
            logger.warning("librosa not available, speed adjustment disabled")
            return audio_data
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
            
            # KittenTTS outputs at 24kHz, but we might need to resample
            tts_sample_rate = 24000
            
            # Resample if needed to match our output settings
            if tts_sample_rate != self.settings.sample_rate:
                try:
                    import librosa
                    audio_data = librosa.resample(
                        audio_data,
                        orig_sr=tts_sample_rate,
                        target_sr=self.settings.sample_rate
                    )
                    sample_rate = self.settings.sample_rate
                except ImportError:
                    logger.warning("librosa not available, using original sample rate")
                    sample_rate = tts_sample_rate
            else:
                sample_rate = tts_sample_rate
            
            # Play audio
            sd.play(audio_data, samplerate=sample_rate)
            
            if wait:
                sd.wait()  # Wait for playback to complete
            
            logger.info("Audio playback completed")
            return True
            
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
            return False
    
    async def synthesize_and_play(self, text: str, 
                                voice: Optional[int] = None,
                                wait: bool = True) -> bool:
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
            7: "Female Voice 4"
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        if not self.model_loaded or self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": self.settings.tts_model,
            "sample_rate": 24000,  # KittenTTS output sample rate
            "available_voices": self.get_available_voices(),
            "current_voice": self.settings.tts_voice,
            "speed": self.settings.tts_speed
        }
    
    async def close(self) -> None:
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Clear model from memory
        if self.model is not None:
            del self.model
            self.model = None
            self.model_loaded = False


# Testing and utility functions
async def test_text_to_speech(settings: Settings,
                            text: str = "Hello, this is a test of the text to speech system.",
                            verbose: bool = False) -> bool:
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


async def test_all_voices(settings: Settings,
                        text: str = "Testing voice",
                        verbose: bool = False) -> Dict[int, bool]:
    """Test all available TTS voices
    
    Args:
        settings: EasyVoice settings
        text: Text to test with each voice
        verbose: Show detailed output
        
    Returns:
        Dictionary mapping voice IDs to success status
    """
    results = {}
    
    try:
        tts = KittenTTS(settings)
        voices = tts.get_available_voices()
        
        for voice_id, voice_name in voices.items():
            if verbose:
                print(f"Testing {voice_name} (ID: {voice_id})")
            
            try:
                # Test synthesis only (don't play to avoid noise)
                audio_data = await tts.synthesize_text(text, voice=voice_id)
                results[voice_id] = audio_data is not None
                
                if verbose:
                    status = "✓" if results[voice_id] else "✗"
                    print(f"  {status} {voice_name}")
                    
            except Exception as e:
                results[voice_id] = False
                if verbose:
                    print(f"  ✗ {voice_name}: {e}")
        
        # Clean up
        await tts.close()
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"Voice testing failed: {e}")
        return {}


async def benchmark_tts_performance(settings: Settings,
                                  test_text: str = "This is a performance test of the text to speech system.") -> Dict[str, Any]:
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
        
        for i in range(3):  # Test 3 times for average
            synthesis_start = time.time()
            audio_data = await tts.synthesize_text(test_text)
            synthesis_time = time.time() - synthesis_start
            
            if audio_data is not None:
                synthesis_times.append(synthesis_time)
        
        # Calculate metrics
        avg_synthesis_time = sum(synthesis_times) / len(synthesis_times) if synthesis_times else 0
        
        # Estimate audio duration (assuming 24kHz output)
        if audio_data is not None:
            audio_duration = len(audio_data) / 24000
            real_time_factor = avg_synthesis_time / audio_duration if audio_duration > 0 else 0
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
            "test_runs": len(synthesis_times)
        }
        
    except Exception as e:
        return {"error": str(e)}