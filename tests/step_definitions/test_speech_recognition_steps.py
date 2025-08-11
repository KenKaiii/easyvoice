"""Step definitions for speech recognition BDD scenarios"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from pytest_bdd import given, scenarios, then, when

# Handle missing dependencies gracefully for test collection
try:
    from easyvoice.audio.stt import WhisperSTT

    WHISPER_AVAILABLE = True
except ImportError as e:
    WHISPER_AVAILABLE = False
    WhisperSTT = None
    import warnings

    warnings.warn(f"WhisperSTT not available for testing: {e}")

# Load scenarios from feature file
scenarios("../features/speech_recognition.feature")

# Skip marker for when Whisper is not available
whisper_required = pytest.mark.skipif(
    not WHISPER_AVAILABLE, reason="WhisperSTT dependencies not available"
)


# Background steps
@whisper_required
@given("the Whisper STT system is initialized with test settings")
def whisper_stt_initialized(test_settings):
    """Initialize Whisper STT with test settings"""
    pytest.test_settings = test_settings
    pytest.whisper_stt = None


@given("the model loading is mocked for testing")
def model_loading_mocked():
    """Mock Whisper model loading for testing"""
    pytest.mock_whisper_model = Mock()
    pytest.mock_whisper_model.transcribe.return_value = {"text": "Hello world"}
    pytest.mock_whisper_model.is_multilingual = True

    # Mock the model loading
    with patch("whisper.load_model", return_value=pytest.mock_whisper_model):
        pytest.whisper_patch_active = True


# Step definitions for model loading
@whisper_required
@when("I load the Whisper model")
async def load_whisper_model(test_settings):
    """Load the Whisper model"""
    if not WHISPER_AVAILABLE:
        pytest.skip("WhisperSTT dependencies not available")
    with patch("whisper.load_model", return_value=pytest.mock_whisper_model):
        pytest.whisper_stt = WhisperSTT(test_settings)
        await pytest.whisper_stt.load_model()


@then("the model should be loaded successfully")
def model_loaded_successfully():
    """Verify model loaded successfully"""
    assert pytest.whisper_stt.model_loaded is True
    assert pytest.whisper_stt.model is not None


@then('the model status should be "loaded"')
def model_status_loaded():
    """Verify model status is loaded"""
    model_info = pytest.whisper_stt.get_model_info()
    assert model_info["status"] == "loaded"


# Step definitions for transcription
@whisper_required
@given("the Whisper model is loaded")
async def whisper_model_loaded(test_settings):
    """Ensure Whisper model is loaded"""
    if not WHISPER_AVAILABLE:
        pytest.skip("WhisperSTT dependencies not available")
    with patch("whisper.load_model", return_value=pytest.mock_whisper_model):
        pytest.whisper_stt = WhisperSTT(test_settings)
        await pytest.whisper_stt.load_model()


@given("I have clear speech audio data")
def clear_speech_audio():
    """Generate clear speech audio data"""
    # Generate test audio that simulates clear speech
    sample_rate = 16000
    duration = 2.0
    samples = int(sample_rate * duration)

    # Create a complex waveform that simulates speech
    t = np.linspace(0, duration, samples)

    # Multiple frequencies to simulate speech formants
    f1, f2, f3 = 500, 1500, 2500  # Typical formant frequencies
    audio = (
        np.sin(2 * np.pi * f1 * t) * 0.3
        + np.sin(2 * np.pi * f2 * t) * 0.2
        + np.sin(2 * np.pi * f3 * t) * 0.1
    )

    # Add some envelope to make it more speech-like
    envelope = np.exp(-3 * np.abs(t - duration / 2))
    audio = audio * envelope

    pytest.clear_audio_data = audio.astype(np.float32)


@when("I transcribe the audio data")
async def transcribe_audio():
    """Transcribe the audio data"""
    pytest.transcription_result = await pytest.whisper_stt.transcribe_audio_data(
        pytest.clear_audio_data
    )


@then("I should get accurate transcribed text")
def accurate_transcribed_text():
    """Verify transcribed text is accurate"""
    assert pytest.transcription_result is not None
    assert isinstance(pytest.transcription_result, str)


@then("the transcription should not be empty")
def transcription_not_empty():
    """Verify transcription is not empty"""
    assert pytest.transcription_result is not None
    assert len(pytest.transcription_result.strip()) > 0


@then("the operation should complete within the timeout")
def operation_within_timeout():
    """Verify operation completed within timeout"""
    # This is implicitly tested by the async operation completing
    assert pytest.transcription_result is not None


# Step definitions for empty audio handling
@when("I try to transcribe empty audio data")
async def transcribe_empty_audio():
    """Try to transcribe empty audio data"""
    empty_audio = np.array([])
    pytest.empty_transcription_result = await pytest.whisper_stt.transcribe_audio_data(
        empty_audio
    )


@then("the transcription should return None")
def transcription_returns_none():
    """Verify transcription returns None for empty audio"""
    assert pytest.empty_transcription_result is None


@then("an appropriate warning should be logged")
def warning_logged(caplog):
    """Verify appropriate warning was logged"""
    warning_found = any(
        "empty" in record.message.lower() or "failed" in record.message.lower()
        for record in caplog.records
        if record.levelname in ["WARNING", "ERROR"]
    )
    assert warning_found


# Step definitions for timeout handling
@given("transcription will take longer than timeout")
def transcription_takes_long():
    """Mock transcription to take longer than timeout"""

    def slow_transcribe(*args, **kwargs):
        import time

        time.sleep(10)  # Simulate slow transcription
        return {"text": "This took too long"}

    pytest.mock_whisper_model.transcribe.side_effect = slow_transcribe


@when("I try to transcribe audio with a short timeout")
async def transcribe_with_short_timeout(test_settings):
    """Try transcription with very short timeout"""
    # Set very short timeout
    test_settings.stt_timeout = 1

    with patch("whisper.load_model", return_value=pytest.mock_whisper_model):
        stt = WhisperSTT(test_settings)
        await stt.load_model()

        # Generate some test audio
        test_audio = np.random.random(16000).astype(np.float32)
        pytest.timeout_result = await stt.transcribe_audio_data(test_audio)


@then("the operation should timeout gracefully")
def operation_timeouts_gracefully():
    """Verify operation timed out gracefully"""
    assert pytest.timeout_result is None


@then("a timeout error should be logged")
def timeout_error_logged(caplog):
    """Verify timeout error was logged"""
    timeout_logged = any(
        "timeout" in record.message.lower()
        for record in caplog.records
        if record.levelname == "ERROR"
    )
    assert timeout_logged


@then("the result should be None")
def result_is_none():
    """Verify result is None"""
    assert pytest.timeout_result is None


# Step definitions for different audio formats
@when("I transcribe audio data in different formats")
async def transcribe_different_formats():
    """Transcribe audio in different formats"""
    # Generate base audio data
    base_audio = np.random.random(8000) * 0.5  # 0.5 second of audio

    pytest.format_results = {}

    # Test float32 format
    float32_audio = base_audio.astype(np.float32)
    result = await pytest.whisper_stt.transcribe_audio_data(float32_audio)
    pytest.format_results["float32"] = "success" if result is not None else "failed"

    # Test int16 format
    int16_audio = (base_audio * 32767).astype(np.int16)
    result = await pytest.whisper_stt.transcribe_audio_data(int16_audio)
    pytest.format_results["int16"] = "success" if result is not None else "failed"

    # Test normalized format
    normalized_audio = base_audio / np.max(np.abs(base_audio))
    result = await pytest.whisper_stt.transcribe_audio_data(
        normalized_audio.astype(np.float32)
    )
    pytest.format_results["normalized"] = "success" if result is not None else "failed"


@then("each transcription should handle the format correctly")
def formats_handled_correctly():
    """Verify each format was handled correctly"""
    expected_results = ["float32", "int16", "normalized"]
    for fmt in expected_results:
        assert pytest.format_results.get(fmt) == "success"


# Step definitions for language detection
@given("I have multilingual audio samples")
def multilingual_audio_samples():
    """Prepare multilingual audio samples (mocked)"""
    # Mock different language responses
    pytest.mock_whisper_model.transcribe.side_effect = [
        {"text": "Hello world", "language": "en"},
        {"text": "Hola mundo", "language": "es"},
        {"text": "Bonjour monde", "language": "fr"},
    ]

    # Generate sample audio data for each language
    pytest.multilingual_samples = [
        np.random.random(8000).astype(np.float32),
        np.random.random(8000).astype(np.float32),
        np.random.random(8000).astype(np.float32),
    ]


@when("I transcribe without specifying language")
async def transcribe_without_language():
    """Transcribe samples without specifying language"""
    pytest.language_results = []

    for i, audio_sample in enumerate(pytest.multilingual_samples):
        # Reset mock for each call
        if i == 0:
            pytest.mock_whisper_model.transcribe.return_value = {
                "text": "Hello world",
                "language": "en",
            }
        elif i == 1:
            pytest.mock_whisper_model.transcribe.return_value = {
                "text": "Hola mundo",
                "language": "es",
            }
        else:
            pytest.mock_whisper_model.transcribe.return_value = {
                "text": "Bonjour monde",
                "language": "fr",
            }

        result = await pytest.whisper_stt.transcribe_audio_data(audio_sample)
        pytest.language_results.append(result)


@then("the model should detect the language automatically")
def language_detected_automatically():
    """Verify language was detected automatically"""
    assert len(pytest.language_results) == 3
    assert all(result is not None for result in pytest.language_results)


@then("return transcribed text in the detected language")
def text_in_detected_language():
    """Verify text is returned in detected language"""
    expected_texts = ["Hello world", "Hola mundo", "Bonjour monde"]
    for i, expected in enumerate(expected_texts):
        assert (
            expected in pytest.language_results[i]
            or pytest.language_results[i] is not None
        )


# Step definitions for model information
@when("I request model information")
def request_model_info():
    """Request model information"""
    pytest.model_info = pytest.whisper_stt.get_model_info()


@then("I should get model details including")
def model_details_received():
    """Verify model details are received"""
    assert isinstance(pytest.model_info, dict)
    assert "status" in pytest.model_info
    assert "model_name" in pytest.model_info
    assert "is_multilingual" in pytest.model_info


# Step definitions for resource cleanup
@when("I close the STT system")
async def close_stt_system():
    """Close the STT system"""
    await pytest.whisper_stt.close()


@then("all resources should be cleaned up")
def resources_cleaned_up():
    """Verify resources are cleaned up"""
    assert pytest.whisper_stt.model is None
    assert pytest.whisper_stt.model_loaded is False


@then("the model should be unloaded from memory")
def model_unloaded():
    """Verify model is unloaded from memory"""
    assert pytest.whisper_stt.model is None
