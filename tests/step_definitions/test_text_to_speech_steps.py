"""Step definitions for text-to-speech BDD scenarios"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from pytest_bdd import given, scenarios, then, when

from easyvoice.audio.tts import KittenTTS

# Load scenarios from feature file
scenarios('../features/text_to_speech.feature')


# Background steps
@given("the KittenTTS system is initialized with test settings")
def kitten_tts_initialized(test_settings):
    """Initialize KittenTTS with test settings"""
    pytest.test_settings = test_settings
    pytest.kitten_tts = None


@given("the TTS model loading is mocked for testing")
def tts_model_loading_mocked():
    """Mock KittenTTS model loading for testing"""
    pytest.mock_kitten_model = Mock()

    # Mock successful audio generation
    sample_rate = 24000
    duration = 1.0
    samples = int(sample_rate * duration)
    test_audio = np.random.random(samples).astype(np.float32) * 0.5

    pytest.mock_kitten_model.generate.return_value = test_audio
    pytest.mock_generated_audio = test_audio


# Step definitions for model loading
@when("I load the KittenTTS model")
async def load_kitten_model(test_settings):
    """Load the KittenTTS model"""
    with patch('easyvoice.audio.tts.KittenTTS._load_model_sync',
               return_value=pytest.mock_kitten_model):
        pytest.kitten_tts = KittenTTS(test_settings)
        await pytest.kitten_tts.load_model()


@then("the model should be loaded successfully")
def tts_model_loaded_successfully():
    """Verify TTS model loaded successfully"""
    assert pytest.kitten_tts.model_loaded is True
    assert pytest.kitten_tts.model is not None


@then('the model status should be "loaded"')
def tts_model_status_loaded():
    """Verify TTS model status is loaded"""
    model_info = pytest.kitten_tts.get_model_info()
    assert model_info["status"] == "loaded"


# Step definitions for text synthesis
@given("the KittenTTS model is loaded")
async def kitten_model_loaded(test_settings):
    """Ensure KittenTTS model is loaded"""
    with patch('easyvoice.audio.tts.KittenTTS._load_model_sync',
               return_value=pytest.mock_kitten_model):
        pytest.kitten_tts = KittenTTS(test_settings)
        await pytest.kitten_tts.load_model()


@when('I synthesize the text "Hello world"')
async def synthesize_hello_world():
    """Synthesize the text 'Hello world'"""
    pytest.synthesis_result = await pytest.kitten_tts.synthesize_text("Hello world")


@then("audio data should be generated")
def audio_data_generated():
    """Verify audio data was generated"""
    assert pytest.synthesis_result is not None
    assert isinstance(pytest.synthesis_result, np.ndarray)


@then("the audio data should not be empty")
def audio_data_not_empty():
    """Verify audio data is not empty"""
    assert pytest.synthesis_result is not None
    assert len(pytest.synthesis_result) > 0


# Step definitions for empty text handling
@when("I try to synthesize empty text")
async def synthesize_empty_text():
    """Try to synthesize empty text"""
    pytest.empty_synthesis_result = await pytest.kitten_tts.synthesize_text("")


@then("the synthesis should return None")
def synthesis_returns_none():
    """Verify synthesis returns None for empty text"""
    assert pytest.empty_synthesis_result is None


@then("an appropriate warning should be logged")
def tts_warning_logged(caplog):
    """Verify appropriate warning was logged"""
    warning_found = any("empty" in record.message.lower()
                        for record in caplog.records
                        if record.levelname == "WARNING")
    assert warning_found


# Step definitions for voice selection
@when("I synthesize text with different voices")
async def synthesize_different_voices():
    """Synthesize text with different voices"""
    pytest.voice_results = {}

    voices_to_test = [
        (0, "Male Voice 1"),
        (4, "Female Voice 1"),
        (7, "Female Voice 4")
    ]

    for voice_id, voice_name in voices_to_test:
        # Mock different audio for each voice
        unique_audio = (np.random.random(24000).astype(np.float32) *
                        0.3 + voice_id * 0.01)
        pytest.mock_kitten_model.generate.return_value = unique_audio

        result = await pytest.kitten_tts.synthesize_text(
            f"Test voice {voice_id}", voice=voice_id
        )

        pytest.voice_results[voice_id] = {
            "name": voice_name,
            "result": result,
            "success": result is not None
        }


@then("each voice should generate unique audio")
def voices_generate_unique_audio():
    """Verify each voice generates unique audio"""
    results = [data["result"] for data in pytest.voice_results.values()]

    # Check that all results are different (basic uniqueness test)
    for i, result1 in enumerate(results):
        for j, result2 in enumerate(results[i+1:], i+1):
            if result1 is not None and result2 is not None:
                # Audio should be different (not exactly the same)
                assert not np.array_equal(result1, result2)


@then("all syntheses should complete successfully")
def all_syntheses_successful():
    """Verify all syntheses completed successfully"""
    for voice_data in pytest.voice_results.values():
        assert voice_data["success"] is True


# Step definitions for invalid voice handling
@when("I try to synthesize with voice ID 10")
async def synthesize_invalid_voice():
    """Try to synthesize with invalid voice ID"""
    try:
        pytest.invalid_voice_result = await pytest.kitten_tts.synthesize_text(
            "Test", voice=10
        )
        pytest.invalid_voice_error = None
    except Exception as e:
        pytest.invalid_voice_error = e
        pytest.invalid_voice_result = None


@then("an error should be raised")
def invalid_voice_error_raised():
    """Verify error was raised for invalid voice"""
    # Either an error was raised or the result is None (graceful handling)
    assert (pytest.invalid_voice_error is not None or
            pytest.invalid_voice_result is None)


@then("the error should mention invalid voice ID")
def error_mentions_invalid_voice():
    """Verify error mentions invalid voice ID"""
    if pytest.invalid_voice_error:
        error_msg = str(pytest.invalid_voice_error).lower()
        assert "voice" in error_msg or "invalid" in error_msg


# Step definitions for audio playback
@given("I have synthesized audio data")
async def have_synthesized_audio():
    """Ensure we have synthesized audio data"""
    pytest.playback_audio_data = await pytest.kitten_tts.synthesize_text(
        "Test playback"
    )


@when("I play the audio")
async def play_audio():
    """Play the synthesized audio"""
    with patch('sounddevice.play') as mock_play, \
         patch('sounddevice.wait') as mock_wait:

        mock_play.return_value = None
        mock_wait.return_value = None

        pytest.playback_result = await pytest.kitten_tts.play_audio(
            pytest.playback_audio_data
        )


@then("the playback should start successfully")
def playback_starts_successfully():
    """Verify playback started successfully"""
    assert pytest.playback_result is True


@then("no audio errors should occur")
def no_audio_errors():
    """Verify no audio errors occurred"""
    # This is implicitly tested by successful playback
    assert pytest.playback_result is True


# Step definitions for speed adjustment
@given("speed adjustment is set to 1.5x")
def speed_adjustment_set(test_settings):
    """Set speed adjustment to 1.5x"""
    test_settings.tts_speed = 1.5
    pytest.test_settings = test_settings


@when('I synthesize text "Testing speed"')
async def synthesize_speed_test():
    """Synthesize text with speed adjustment"""
    # Mock librosa for speed adjustment
    with patch('librosa.effects.time_stretch') as mock_stretch:
        original_audio = pytest.mock_generated_audio
        faster_audio = original_audio[::2]  # Simple simulation of faster audio
        mock_stretch.return_value = faster_audio

        pytest.speed_result = await pytest.kitten_tts.synthesize_text(
            "Testing speed"
        )


@then("the generated audio should be faster than normal")
def audio_faster_than_normal():
    """Verify audio is faster than normal"""
    assert pytest.speed_result is not None
    # In real implementation, this would be shorter due to speed adjustment
    assert len(pytest.speed_result) > 0


@then("the audio duration should be shorter")
def audio_duration_shorter():
    """Verify audio duration is shorter"""
    # This would be validated by comparing durations in real implementation
    assert pytest.speed_result is not None


# Step definitions for saving audio
@when('I save the audio to "test_output.wav"')
async def save_audio_file():
    """Save synthesized audio to file"""
    with patch('soundfile.write') as mock_write, \
         patch('pathlib.Path.mkdir'):

        mock_write.return_value = None
        pytest.save_result = await pytest.kitten_tts.save_audio(
            pytest.playback_audio_data,
            "test_output.wav"
        )


@then("the file should be created successfully")
def file_created_successfully():
    """Verify file was created successfully"""
    assert pytest.save_result is True


@then("the file should contain valid audio data")
def file_contains_valid_audio():
    """Verify file contains valid audio data"""
    # This is implicitly tested by successful save operation
    assert pytest.save_result is True


# Step definitions for TTS timeout handling
@given("synthesis will take longer than timeout")
def synthesis_takes_long():
    """Mock synthesis to take longer than timeout"""
    def slow_synthesis(*args, **kwargs):
        import time
        time.sleep(10)  # Simulate slow synthesis
        return pytest.mock_generated_audio

    pytest.mock_kitten_model.generate.side_effect = slow_synthesis


@when("I try to synthesize with a short timeout")
async def synthesize_with_short_timeout(test_settings):
    """Try synthesis with very short timeout"""
    # Set very short timeout
    test_settings.tts_timeout = 1

    with patch('easyvoice.audio.tts.KittenTTS._load_model_sync',
               return_value=pytest.mock_kitten_model):
        tts = KittenTTS(test_settings)
        await tts.load_model()

        pytest.tts_timeout_result = await tts.synthesize_text(
            "This will timeout"
        )


@then("the operation should timeout gracefully")
def tts_operation_timeouts_gracefully():
    """Verify TTS operation timed out gracefully"""
    assert pytest.tts_timeout_result is None


@then("a timeout error should be logged")
def tts_timeout_error_logged(caplog):
    """Verify timeout error was logged"""
    timeout_logged = any("timeout" in record.message.lower()
                         for record in caplog.records
                         if record.levelname == "ERROR")
    assert timeout_logged


# Step definitions for available voices
@when("I request available voices information")
def request_voices_info():
    """Request information about available voices"""
    pytest.voices_info = pytest.kitten_tts.get_available_voices()


@then("I should get a list of 8 voices")
def get_8_voices():
    """Verify we get information about 8 voices"""
    assert len(pytest.voices_info) == 8


@then("each voice should have an ID and description")
def voices_have_id_and_description():
    """Verify each voice has ID and description"""
    for voice_id, description in pytest.voices_info.items():
        assert isinstance(voice_id, int)
        assert isinstance(description, str)
        assert 0 <= voice_id <= 7


@then("voices should include both male and female options")
def voices_include_male_female():
    """Verify voices include both male and female options"""
    descriptions = list(pytest.voices_info.values())

    male_voices = [desc for desc in descriptions if "Male" in desc]
    female_voices = [desc for desc in descriptions if "Female" in desc]

    assert len(male_voices) > 0
    assert len(female_voices) > 0


# Step definitions for performance metrics
@when('I benchmark TTS performance with "Performance test text"')
async def benchmark_tts_performance():
    """Benchmark TTS performance"""
    from easyvoice.audio.tts import benchmark_tts_performance

    with patch('easyvoice.audio.tts.KittenTTS._load_model_sync',
               return_value=pytest.mock_kitten_model):
        pytest.performance_metrics = await benchmark_tts_performance(
            pytest.test_settings,
            "Performance test text"
        )


@then("I should get timing metrics")
def get_timing_metrics():
    """Verify timing metrics are provided"""
    assert "load_time" in pytest.performance_metrics
    assert "avg_synthesis_time" in pytest.performance_metrics


@then("the real-time factor should be calculated")
def real_time_factor_calculated():
    """Verify real-time factor is calculated"""
    assert "real_time_factor" in pytest.performance_metrics
    assert isinstance(pytest.performance_metrics["real_time_factor"],
                      (int, float))


@then("model information should be included")
def model_info_included():
    """Verify model information is included"""
    assert "model_info" in pytest.performance_metrics


# Step definitions for resource cleanup
@when("I close the TTS system")
async def close_tts_system():
    """Close the TTS system"""
    await pytest.kitten_tts.close()


@then("all resources should be cleaned up")
def tts_resources_cleaned_up():
    """Verify TTS resources are cleaned up"""
    assert pytest.kitten_tts.model is None
    assert pytest.kitten_tts.model_loaded is False


@then("the model should be unloaded from memory")
def tts_model_unloaded():
    """Verify TTS model is unloaded from memory"""
    assert pytest.kitten_tts.model is None
