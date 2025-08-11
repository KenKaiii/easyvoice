"""Audio input handling with microphone capture and voice activity detection"""

import asyncio
import logging
import time
from typing import Optional, Tuple, List
import threading

import numpy as np
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    sd = None
    HAS_SOUNDDEVICE = False

from easyvoice.config.settings import Settings

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Simple energy-based voice activity detector"""
    
    def __init__(self, threshold: float = 0.01, min_speech_duration: float = 0.5):
        """Initialize VAD
        
        Args:
            threshold: Energy threshold for voice detection
            min_speech_duration: Minimum duration to consider as speech
        """
        self.threshold = threshold
        self.min_speech_duration = min_speech_duration
        self.speech_start_time: Optional[float] = None
        self.is_speaking = False
        
    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk and return True if voice is detected
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            True if voice activity detected, False otherwise
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        current_time = time.time()
        
        if energy > self.threshold:
            if not self.is_speaking:
                self.speech_start_time = current_time
                self.is_speaking = True
            return True
        else:
            if self.is_speaking:
                # Check if we've had enough speech duration
                if (self.speech_start_time and 
                    current_time - self.speech_start_time >= self.min_speech_duration):
                    self.is_speaking = False
                    return True  # End of speech detected
            self.is_speaking = False
            return False


class AudioInput:
    """Handle microphone input and audio capture"""
    
    def __init__(self, settings: Settings):
        """Initialize audio input
        
        Args:
            settings: EasyVoice configuration settings
        """
        self.settings = settings
        self.vad = VoiceActivityDetector(
            threshold=settings.vad_threshold,
            min_speech_duration=0.5
        )
        
        # Audio stream state
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False
        self.audio_buffer: List[np.ndarray] = []
        self.buffer_lock = threading.Lock()
        
        # Validate audio device
        self._validate_audio_device()
    
    def _validate_audio_device(self) -> None:
        """Validate that audio input device is available"""
        if not HAS_SOUNDDEVICE:
            raise RuntimeError("sounddevice not available - install with: pip install sounddevice")
            
        try:
            devices = sd.query_devices()
            default_input = sd.default.device[0]
            
            if default_input is None:
                raise RuntimeError("No default audio input device found")
            
            device_info = sd.query_devices(default_input, 'input')
            logger.info(f"Using audio input device: {device_info['name']}")
            
            # Check if sample rate is supported
            try:
                sd.check_input_settings(
                    device=default_input,
                    channels=self.settings.channels,
                    samplerate=self.settings.sample_rate
                )
            except sd.PortAudioError as e:
                logger.warning(f"Audio settings validation warning: {e}")
                
        except Exception as e:
            raise RuntimeError(f"Audio device validation failed: {e}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback function for audio stream
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Convert to mono if needed
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        # Store in buffer
        with self.buffer_lock:
            self.audio_buffer.append(audio_data.copy())
    
    async def start_recording(self) -> None:
        """Start audio recording stream"""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        try:
            self.stream = sd.InputStream(
                channels=self.settings.channels,
                samplerate=self.settings.sample_rate,
                blocksize=self.settings.chunk_size,
                callback=self._audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            self.is_recording = True
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            raise
    
    async def stop_recording(self) -> None:
        """Stop audio recording stream"""
        if not self.is_recording:
            return
        
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            self.is_recording = False
            logger.info("Audio recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}")
    
    def get_audio_data(self) -> np.ndarray:
        """Get accumulated audio data and clear buffer
        
        Returns:
            Concatenated audio data as numpy array
        """
        with self.buffer_lock:
            if not self.audio_buffer:
                return np.array([])
            
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            self.audio_buffer.clear()
            
            return audio_data
    
    async def record_for_duration(self, duration: float) -> np.ndarray:
        """Record audio for a specific duration
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio data as numpy array
        """
        await self.start_recording()
        
        try:
            # Wait for specified duration
            await asyncio.sleep(duration)
            
            # Get recorded data
            audio_data = self.get_audio_data()
            return audio_data
            
        finally:
            await self.stop_recording()
    
    async def record_until_silence(self, 
                                 max_duration: float = 30.0,
                                 silence_duration: float = 1.0) -> np.ndarray:
        """Record audio until silence is detected
        
        Args:
            max_duration: Maximum recording duration in seconds
            silence_duration: Duration of silence to stop recording
            
        Returns:
            Recorded audio data as numpy array
        """
        await self.start_recording()
        
        try:
            start_time = time.time()
            last_speech_time = start_time
            
            while True:
                # Check for timeout
                if time.time() - start_time > max_duration:
                    logger.warning(f"Recording timeout after {max_duration}s")
                    break
                
                # Get recent audio data for VAD
                with self.buffer_lock:
                    if self.audio_buffer:
                        recent_chunk = self.audio_buffer[-1]
                        
                        # Check for voice activity
                        if self.vad.process_chunk(recent_chunk):
                            last_speech_time = time.time()
                
                # Check for silence duration
                if time.time() - last_speech_time > silence_duration:
                    logger.info("Silence detected, stopping recording")
                    break
                
                # Small delay to prevent tight loop
                await asyncio.sleep(0.1)
            
            # Get all recorded data
            audio_data = self.get_audio_data()
            return audio_data
            
        finally:
            await self.stop_recording()
    
    async def wait_for_speech(self, timeout: float = 30.0) -> bool:
        """Wait for speech to be detected
        
        Args:
            timeout: Maximum wait time in seconds
            
        Returns:
            True if speech detected, False if timeout
        """
        await self.start_recording()
        
        try:
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                with self.buffer_lock:
                    if self.audio_buffer:
                        recent_chunk = self.audio_buffer[-1]
                        
                        if self.vad.process_chunk(recent_chunk):
                            return True
                
                await asyncio.sleep(0.1)
            
            return False
            
        finally:
            await self.stop_recording()


# Testing functions
def test_microphone(duration: float = 2.0, verbose: bool = False) -> bool:
    """Test microphone input
    
    Args:
        duration: Test duration in seconds
        verbose: Show detailed output
        
    Returns:
        True if microphone is working, False otherwise
    """
    try:
        # Test basic recording
        test_data = sd.rec(
            int(duration * 16000),  # frames
            samplerate=16000,
            channels=1,
            dtype=np.float32
        )
        
        sd.wait()  # Wait for recording to complete
        
        # Check if we got data
        if len(test_data) == 0:
            if verbose:
                print("No audio data recorded")
            return False
        
        # Check for non-zero audio (basic sanity check)
        max_amplitude = np.max(np.abs(test_data))
        
        if verbose:
            print(f"Recorded {len(test_data)} samples")
            print(f"Max amplitude: {max_amplitude:.4f}")
        
        # Very basic test - just check we got some data
        return max_amplitude > 0.001  # Very low threshold
        
    except Exception as e:
        if verbose:
            print(f"Microphone test failed: {e}")
        return False


async def test_voice_activity_detection(settings: Settings, 
                                      duration: float = 5.0,
                                      verbose: bool = False) -> bool:
    """Test voice activity detection
    
    Args:
        settings: EasyVoice settings
        duration: Test duration in seconds
        verbose: Show detailed output
        
    Returns:
        True if VAD is working, False otherwise
    """
    try:
        audio_input = AudioInput(settings)
        
        if verbose:
            print(f"Testing VAD for {duration} seconds...")
        
        # Record and test VAD
        audio_data = await audio_input.record_for_duration(duration)
        
        if len(audio_data) == 0:
            if verbose:
                print("No audio data for VAD test")
            return False
        
        # Simple test - check that VAD processes without error
        vad = VoiceActivityDetector()
        
        # Process in chunks
        chunk_size = settings.chunk_size
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            if len(chunk) == chunk_size:
                vad.process_chunk(chunk)
        
        if verbose:
            print("VAD test completed successfully")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"VAD test failed: {e}")
        return False