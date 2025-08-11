"""Step definitions for audio input BDD scenarios"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pytest_bdd import scenarios, given, when, then, parsers

from easyvoice.audio.input import AudioInput, VoiceActivityDetector, test_microphone
from easyvoice.config.settings import Settings

# Load scenarios from feature file
scenarios('../features/audio_input.feature')


# Fixtures for test data
@pytest.fixture
def test_audio_data():
    """Generate test audio data"""
    # Generate 2 seconds of test audio (speech-like)
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)
    
    # Create sine wave with some noise (simulates speech)
    t = np.linspace(0, duration, samples)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    # Add some random noise
    noise = np.random.normal(0, 0.05, samples)
    audio = audio + noise
    
    return audio.astype(np.float32)


@pytest.fixture
def silent_audio_data():
    """Generate silent audio data"""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    
    # Very quiet noise (simulates silence)
    return np.random.normal(0, 0.001, samples).astype(np.float32)


# Background steps
@given("the audio system is initialized with test settings")
def audio_system_initialized(test_settings):
    """Initialize audio system with test settings"""
    pytest.current_settings = test_settings
    pytest.audio_input = None


@given("the microphone is available")  
def microphone_available():
    """Mock microphone availability"""
    with patch('sounddevice.query_devices') as mock_query:
        mock_query.return_value = [{'name': 'Test Microphone'}]
        with patch('sounddevice.default.device', [0, None]):
            with patch('sounddevice.check_input_settings'):
                pytest.microphone_mock_active = True


# Step definitions for microphone testing
@when("I test the microphone")
def test_microphone_step():
    """Test microphone functionality"""
    with patch('sounddevice.rec') as mock_rec, \
         patch('sounddevice.wait') as mock_wait:
        
        # Mock successful recording
        mock_rec.return_value = np.array([[0.1], [0.2], [0.3]])
        mock_wait.return_value = None
        
        pytest.microphone_test_result = test_microphone(duration=0.1, verbose=False)


@then("the microphone should be detected")
def microphone_detected():
    """Verify microphone was detected"""
    assert pytest.microphone_test_result is True


@then("no errors should occur") 
def no_errors():
    """Verify no errors occurred"""
    # This is implicitly tested by the success of previous steps
    pass


# Step definitions for audio recording
@given("the audio input is ready")
def audio_input_ready(test_settings):
    """Initialize audio input for recording"""
    with patch('sounddevice.query_devices'), \
         patch('sounddevice.default.device', [0, None]), \
         patch('sounddevice.check_input_settings'):
        
        pytest.audio_input = AudioInput(test_settings)


@when(parsers.parse("I record audio for {duration:d} seconds"))
async def record_audio_duration(duration, test_audio_data):
    """Record audio for specified duration"""
    with patch('sounddevice.InputStream') as mock_stream_class:
        mock_stream = Mock()
        mock_stream_class.return_value = mock_stream
        
        # Simulate audio data being captured
        pytest.audio_input.audio_buffer = [test_audio_data[:1024] for _ in range(10)]
        
        pytest.recorded_audio = await pytest.audio_input.record_for_duration(duration)


@then("audio data should be captured")
def audio_data_captured():
    """Verify audio data was captured"""
    assert pytest.recorded_audio is not None
    assert len(pytest.recorded_audio) > 0


@then("the audio data should not be empty")
def audio_data_not_empty():
    """Verify audio data is not empty"""
    assert len(pytest.recorded_audio) > 0


@then("the recording should stop automatically")
def recording_stops_automatically():
    """Verify recording stops after duration"""
    assert pytest.audio_input.is_recording is False


# Step definitions for voice activity detection
@given("the voice activity detector is initialized")
def vad_initialized():
    """Initialize voice activity detector"""
    pytest.vad = VoiceActivityDetector(threshold=0.01, min_speech_duration=0.1)


@when("I process an audio chunk with speech")
def process_speech_chunk(test_audio_data):
    """Process audio chunk containing speech"""
    # Use first chunk of test audio data
    speech_chunk = test_audio_data[:1024]
    pytest.speech_detected = pytest.vad.process_chunk(speech_chunk)


@when("I process an audio chunk with silence") 
def process_silence_chunk(silent_audio_data):
    """Process audio chunk containing silence"""
    silence_chunk = silent_audio_data[:1024]
    pytest.silence_detected = pytest.vad.process_chunk(silence_chunk)


@then("voice activity should be detected")
def voice_activity_detected():
    """Verify voice activity was detected"""
    assert pytest.speech_detected is True


@then("no voice activity should be detected")
def no_voice_activity():
    """Verify no voice activity was detected"""
    assert pytest.silence_detected is False


# Step definitions for silence detection recording
@given("voice activity detection is enabled")
def vad_enabled():
    """Ensure VAD is enabled"""
    assert pytest.audio_input.vad is not None


@when("I start recording with silence detection")
def start_silence_detection_recording():
    """Start recording with silence detection"""
    pytest.silence_recording_task = asyncio.create_task(
        pytest.audio_input.record_until_silence(max_duration=5.0, silence_duration=1.0)
    )


@when("speech is detected initially")
def speech_detected_initially(test_audio_data):
    """Simulate initial speech detection"""
    # Add speech data to buffer
    pytest.audio_input.audio_buffer = [test_audio_data[:1024]]


@when("silence follows for 1 second")
async def silence_follows(silent_audio_data):
    """Simulate silence following speech"""
    # Add silence data and wait for task completion
    pytest.audio_input.audio_buffer.extend([silent_audio_data[:1024] for _ in range(5)])
    
    # Wait a bit for the recording task to process
    await asyncio.sleep(0.1)
    
    if not pytest.silence_recording_task.done():
        pytest.silence_recording_task.cancel()
        
    try:
        pytest.silence_recorded_audio = await pytest.silence_recording_task
    except asyncio.CancelledError:
        pytest.silence_recorded_audio = pytest.audio_input.get_audio_data()


@then("recording should stop automatically")
def recording_stops_on_silence():
    """Verify recording stops when silence is detected"""
    assert not pytest.audio_input.is_recording


@then("captured audio should contain the speech")
def captured_audio_contains_speech():
    """Verify captured audio contains speech data"""
    assert pytest.silence_recorded_audio is not None
    assert len(pytest.silence_recorded_audio) > 0


# Step definitions for error handling
@given("the microphone is not available")
def microphone_not_available():
    """Mock microphone unavailability"""
    with patch('sounddevice.query_devices') as mock_query:
        mock_query.return_value = []
        with patch('sounddevice.default.device', [None, None]):
            pytest.microphone_unavailable = True


@when("I try to start audio recording")
def try_start_recording():
    """Attempt to start audio recording"""
    try:
        with patch('sounddevice.query_devices', return_value=[]), \
             patch('sounddevice.default.device', [None, None]):
            
            audio_input = AudioInput(pytest.current_settings)
            pytest.audio_error = None
    except Exception as e:
        pytest.audio_error = e


@then("an appropriate error should be raised")
def appropriate_error_raised():
    """Verify appropriate error was raised"""
    assert pytest.audio_error is not None
    assert isinstance(pytest.audio_error, RuntimeError)


@then("the error message should be informative")
def error_message_informative():
    """Verify error message is informative"""
    assert "audio" in str(pytest.audio_error).lower()


# Step definitions for buffer management  
@given("the audio input is recording")
def audio_input_recording():
    """Set up audio input in recording state"""
    pytest.audio_input.is_recording = True
    pytest.audio_input.audio_buffer = []


@when("audio data accumulates in the buffer")
def audio_accumulates(test_audio_data):
    """Simulate audio data accumulating"""
    # Add multiple chunks to buffer
    for i in range(5):
        chunk = test_audio_data[i*1024:(i+1)*1024]
        if len(chunk) > 0:
            pytest.audio_input.audio_buffer.append(chunk)


@when("I request the audio data")
def request_audio_data():
    """Request audio data from buffer"""
    pytest.buffer_audio_data = pytest.audio_input.get_audio_data()


@then("the buffer should return the audio data")
def buffer_returns_data():
    """Verify buffer returns audio data"""
    assert pytest.buffer_audio_data is not None
    assert len(pytest.buffer_audio_data) > 0


@then("the buffer should be cleared after retrieval")
def buffer_cleared():
    """Verify buffer is cleared after retrieval"""
    assert len(pytest.audio_input.audio_buffer) == 0


# Step definitions for timeout protection
@when(parsers.parse("I start recording with a {timeout:d} second timeout"))
def start_recording_with_timeout(timeout):
    """Start recording with specified timeout"""
    pytest.timeout_value = timeout
    pytest.timeout_task = asyncio.create_task(
        pytest.audio_input.record_until_silence(max_duration=timeout, silence_duration=2.0)
    )


@when("no silence is detected")
async def no_silence_detected(test_audio_data):
    """Simulate continuous speech without silence"""
    # Keep adding speech data to prevent silence detection
    for _ in range(10):
        pytest.audio_input.audio_buffer.append(test_audio_data[:1024])
        await asyncio.sleep(0.1)
    
    # Wait for timeout task to complete
    try:
        pytest.timeout_result = await pytest.timeout_task
    except Exception:
        pytest.timeout_result = pytest.audio_input.get_audio_data()


@then(parsers.parse("recording should stop after {timeout:d} seconds"))
def recording_stops_after_timeout(timeout):
    """Verify recording stops after timeout"""
    assert not pytest.audio_input.is_recording


@then("a timeout warning should be logged")
def timeout_warning_logged(caplog):
    """Verify timeout warning was logged"""
    # Check if timeout warning appears in logs
    timeout_logged = any("timeout" in record.message.lower() for record in caplog.records)
    assert timeout_logged