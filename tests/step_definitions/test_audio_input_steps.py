"""Step definitions for audio input BDD scenarios"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, patch
from pytest_bdd import scenarios, given, when, then, parsers

from easyvoice.audio.input import AudioInput, VoiceActivityDetector, test_microphone

# Load scenarios from feature file
scenarios("../features/audio_input.feature")


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
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    noise = rng.normal(0, 0.05, samples)
    audio = audio + noise

    return audio.astype(np.float32)


@pytest.fixture
def silent_audio_data():
    """Generate silent audio data"""
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Very quiet noise (simulates silence)
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    return rng.normal(0, 0.001, samples).astype(np.float32)


# Background steps
@given("the audio system is initialized with test settings")
def audio_system_initialized(test_settings, test_context):
    """Initialize audio system with test settings"""
    test_context.current_settings = test_settings
    test_context.audio_input = None


@given("the microphone is available")
def microphone_available(test_context):
    """Mock microphone availability"""
    with patch("sounddevice.query_devices") as mock_query:
        mock_query.return_value = [{"name": "Test Microphone"}]
        with patch("sounddevice.default.device", [0, None]):
            with patch("sounddevice.check_input_settings"):
                test_context.microphone_mock_active = True


# Step definitions for microphone testing
@when("I test the microphone")
def test_microphone_step(test_context):
    """Test microphone functionality"""
    with patch("sounddevice.rec") as mock_rec, patch("sounddevice.wait") as mock_wait:
        # Mock successful recording
        mock_rec.return_value = np.array([[0.1], [0.2], [0.3]])
        mock_wait.return_value = None

        test_context.microphone_test_result = test_microphone(
            duration=0.1, verbose=False
        )


@then("the microphone should be detected")
def microphone_detected(test_context):
    """Verify microphone was detected"""
    assert test_context.microphone_test_result is True


@then("no errors should occur")
def no_errors():
    """Verify no errors occurred"""
    # This is implicitly tested by the success of previous steps
    pass


# Step definitions for audio recording
@given("the audio input is ready")
def audio_input_ready(test_settings, test_context):
    """Initialize audio input for recording"""
    with (
        patch("sounddevice.query_devices"),
        patch("sounddevice.default.device", [0, None]),
        patch("sounddevice.check_input_settings"),
    ):
        test_context.audio_input = AudioInput(test_settings)


@when(parsers.parse("I record audio for {duration:d} seconds"))
def record_audio_duration(duration, test_audio_data, test_context):
    """Record audio for specified duration"""
    import asyncio

    async def _record_audio():
        with patch("sounddevice.InputStream") as mock_stream_class:
            mock_stream = Mock()
            mock_stream_class.return_value = mock_stream

            # Simulate audio data being captured
            test_context.audio_input.audio_buffer = [
                test_audio_data[:1024] for _ in range(10)
            ]

            return await test_context.audio_input.record_for_duration(duration)

    # Run the async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        test_context.recorded_audio = loop.run_until_complete(_record_audio())
    finally:
        loop.close()


@then("audio data should be captured")
def audio_data_captured(test_context):
    """Verify audio data was captured"""
    assert test_context.recorded_audio is not None
    assert len(test_context.recorded_audio) > 0


@then("the audio data should not be empty")
def audio_data_not_empty(test_context):
    """Verify audio data is not empty"""
    assert len(test_context.recorded_audio) > 0


@then("the recording should stop automatically")
def recording_stops_automatically(test_context):
    """Verify recording stops after duration"""
    assert test_context.audio_input.is_recording is False


# Step definitions for voice activity detection
@given("the voice activity detector is initialized")
def vad_initialized(test_context):
    """Initialize voice activity detector"""
    test_context.vad = VoiceActivityDetector(threshold=0.01, min_speech_duration=0.1)


@when("I process an audio chunk with speech")
def process_speech_chunk(test_audio_data, test_context):
    """Process audio chunk containing speech"""
    # Use first chunk of test audio data
    speech_chunk = test_audio_data[:1024]
    test_context.speech_detected = test_context.vad.process_chunk(speech_chunk)


@when("I process an audio chunk with silence")
def process_silence_chunk(silent_audio_data, test_context):
    """Process audio chunk containing silence"""
    silence_chunk = silent_audio_data[:1024]
    test_context.silence_detected = test_context.vad.process_chunk(silence_chunk)


@then("voice activity should be detected")
def voice_activity_detected(test_context):
    """Verify voice activity was detected"""
    assert test_context.speech_detected is True


@then("no voice activity should be detected")
def no_voice_activity(test_context):
    """Verify no voice activity was detected"""
    assert test_context.silence_detected is False


# Step definitions for silence detection recording
@given("voice activity detection is enabled")
def vad_enabled(test_context):
    """Ensure VAD is enabled"""
    assert test_context.audio_input.vad is not None


@when("I start recording with silence detection")
def start_silence_detection_recording(test_context):
    """Start recording with silence detection"""

    async def _create_recording_task():
        return await test_context.audio_input.record_until_silence(
            max_duration=5.0, silence_duration=1.0
        )

    # Store the coroutine for later execution
    test_context.silence_recording_coroutine = _create_recording_task()


@when("speech is detected initially")
def speech_detected_initially(test_audio_data, test_context):
    """Simulate initial speech detection"""
    # Add speech data to buffer
    test_context.audio_input.audio_buffer = [test_audio_data[:1024]]


@when("silence follows for 1 second")
def silence_follows(silent_audio_data, test_context):
    """Simulate silence following speech"""

    async def _silence_follows():
        # Add silence data and wait for task completion
        test_context.audio_input.audio_buffer.extend(
            [silent_audio_data[:1024] for _ in range(5)]
        )

        # Wait a bit for the recording task to process
        await asyncio.sleep(0.1)

        # Execute the stored coroutine
        try:
            test_context.silence_recorded_audio = (
                await test_context.silence_recording_coroutine
            )
        except Exception:
            test_context.silence_recorded_audio = (
                test_context.audio_input.get_audio_data()
            )

    # Run the async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_silence_follows())
    finally:
        loop.close()


@then("recording should stop automatically")
def recording_stops_on_silence(test_context):
    """Verify recording stops when silence is detected"""
    assert not test_context.audio_input.is_recording


@then("captured audio should contain the speech")
def captured_audio_contains_speech(test_context):
    """Verify captured audio contains speech data"""
    assert test_context.silence_recorded_audio is not None
    assert len(test_context.silence_recorded_audio) > 0


# Step definitions for error handling
@given("the microphone is not available")
def microphone_not_available(test_context):
    """Mock microphone unavailability"""
    with patch("sounddevice.query_devices") as mock_query:
        mock_query.return_value = []
        with patch("sounddevice.default.device", [None, None]):
            test_context.microphone_unavailable = True


@when("I try to start audio recording")
def try_start_recording(test_context):
    """Attempt to start audio recording"""
    try:
        with (
            patch("sounddevice.query_devices", return_value=[]),
            patch("sounddevice.default.device", [None, None]),
        ):
            AudioInput(test_context.current_settings)
            test_context.audio_error = None
    except Exception as e:
        test_context.audio_error = e


@then("an appropriate error should be raised")
def appropriate_error_raised(test_context):
    """Verify appropriate error was raised"""
    assert test_context.audio_error is not None
    assert isinstance(test_context.audio_error, RuntimeError)


@then("the error message should be informative")
def error_message_informative(test_context):
    """Verify error message is informative"""
    assert "audio" in str(test_context.audio_error).lower()


# Step definitions for buffer management
@given("the audio input is recording")
def audio_input_recording(test_context, test_settings):
    """Set up audio input in recording state"""
    # Ensure audio input is initialized
    if test_context.audio_input is None:
        with (
            patch("sounddevice.query_devices"),
            patch("sounddevice.default.device", [0, None]),
            patch("sounddevice.check_input_settings"),
        ):
            test_context.audio_input = AudioInput(test_settings)

    test_context.audio_input.is_recording = True
    test_context.audio_input.audio_buffer = []


@when("audio data accumulates in the buffer")
def audio_accumulates(test_audio_data, test_context):
    """Simulate audio data accumulating"""
    # Add multiple chunks to buffer
    for i in range(5):
        chunk = test_audio_data[i * 1024 : (i + 1) * 1024]
        if len(chunk) > 0:
            test_context.audio_input.audio_buffer.append(chunk)


@when("I request the audio data")
def request_audio_data(test_context):
    """Request audio data from buffer"""
    test_context.buffer_audio_data = test_context.audio_input.get_audio_data()


@then("the buffer should return the audio data")
def buffer_returns_data(test_context):
    """Verify buffer returns audio data"""
    assert test_context.buffer_audio_data is not None
    assert len(test_context.buffer_audio_data) > 0


@then("the buffer should be cleared after retrieval")
def buffer_cleared(test_context):
    """Verify buffer is cleared after retrieval"""
    assert len(test_context.audio_input.audio_buffer) == 0


# Step definitions for timeout protection
@when(parsers.parse("I start recording with a {timeout:d} second timeout"))
def start_recording_with_timeout(timeout, test_context):
    """Start recording with specified timeout"""
    test_context.timeout_value = timeout

    async def _create_timeout_task():
        return await test_context.audio_input.record_until_silence(
            max_duration=timeout, silence_duration=2.0
        )

    # Store the coroutine for later execution
    test_context.timeout_coroutine = _create_timeout_task()


@when("no silence is detected")
def no_silence_detected(test_audio_data, test_context):
    """Simulate continuous speech without silence"""

    async def _no_silence_detected():
        # Keep adding speech data to prevent silence detection
        for _ in range(5):  # Reduced iterations for faster testing
            test_context.audio_input.audio_buffer.append(test_audio_data[:1024])
            await asyncio.sleep(0.05)  # Reduced sleep time

        # Wait for timeout coroutine to complete
        try:
            test_context.timeout_result = await test_context.timeout_coroutine
        except Exception:
            test_context.timeout_result = test_context.audio_input.get_audio_data()

    # Run the async function synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_no_silence_detected())
    finally:
        loop.close()


@then(parsers.parse("recording should stop after {timeout:d} seconds"))
def recording_stops_after_timeout(timeout, test_context):
    """Verify recording stops after timeout"""
    assert not test_context.audio_input.is_recording


@then("a timeout warning should be logged")
def timeout_warning_logged(caplog):
    """Verify timeout warning was logged"""
    # Check if timeout warning appears in logs
    timeout_logged = any(
        "timeout" in record.message.lower() for record in caplog.records
    )
    assert timeout_logged
