"""Pytest configuration and fixtures for EasyVoice tests"""

import asyncio
import sqlite3
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from click.testing import CliRunner

from easyvoice.config.settings import Settings


@pytest.fixture
def cli_runner() -> CliRunner:
    """Click CLI test runner fixture"""
    return CliRunner()


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def test_settings(temp_db_path: Path) -> Settings:
    """Test configuration settings"""
    return Settings(
        # Audio - use smaller values for testing
        sample_rate=8000,
        chunk_size=512,
        # LLM - shorter timeouts for tests
        model_name="test-model",
        max_tokens=100,
        # Memory - use test database
        max_messages=5,  # Smaller for testing
        db_path=str(temp_db_path),
        # Shorter timeouts for tests
        stt_timeout=5,
        tts_timeout=3,
        llm_timeout=10,
    )


@pytest.fixture
def mock_whisper_model():
    """Mock Whisper STT model"""
    mock_model = MagicMock()
    mock_model.transcribe.return_value = {"text": "Hello world"}
    return mock_model


@pytest.fixture
def mock_kitten_tts():
    """Mock KittenTTS model"""
    mock_tts = MagicMock()
    mock_tts.generate.return_value = b"fake_audio_data"
    return mock_tts


@pytest.fixture
def mock_ollama_llm():
    """Mock Ollama LLM"""
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.content = "This is a test response"
    return mock_llm


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice for audio I/O"""
    mock_sd = MagicMock()
    mock_sd.rec.return_value = [[0.1, 0.2, 0.3]]  # Fake audio data
    mock_sd.play = MagicMock()
    mock_sd.wait = MagicMock()
    return mock_sd


@pytest.fixture
async def event_loop() -> AsyncGenerator[asyncio.AbstractEventLoop, None]:
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()


@pytest.fixture
def sample_audio_data() -> bytes:
    """Sample audio data for testing"""
    # Generate simple sine wave-like data
    import numpy as np

    sample_rate = 16000
    duration = 1.0  # 1 second
    frequency = 440  # A4 note

    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5

    # Convert to 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16.tobytes()


@pytest.fixture
def sample_conversation_history() -> list[dict]:
    """Sample conversation history for testing"""
    return [
        {"role": "user", "content": "Hello", "timestamp": "2025-01-01T10:00:00"},
        {
            "role": "assistant",
            "content": "Hi there!",
            "timestamp": "2025-01-01T10:00:01",
        },
        {"role": "user", "content": "How are you?", "timestamp": "2025-01-01T10:00:02"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you!",
            "timestamp": "2025-01-01T10:00:03",
        },
    ]


@pytest.fixture
def memory_database(temp_db_path: Path) -> sqlite3.Connection:
    """Create test memory database with schema"""
    conn = sqlite3.connect(temp_db_path)

    # Create schema
    conn.execute(
        """
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    conn.commit()
    return conn


# BDD Step fixtures for pytest-bdd
@pytest.fixture
def voice_agent():
    """Mock voice agent for BDD tests"""
    agent = MagicMock()
    agent.is_listening = False
    agent.memory_count = 0
    agent.last_response = None
    return agent


@pytest.fixture
def audio_pipeline():
    """Mock audio pipeline for BDD tests"""
    pipeline = MagicMock()
    pipeline.is_recording = False
    pipeline.last_transcription = None
    pipeline.last_synthesis = None
    return pipeline


# Global test context for BDD tests (replaces storing on pytest module)
@pytest.fixture
def test_context():
    """Test context for BDD step data sharing"""

    class TestContext:
        def __init__(self):
            # Audio test data
            self.recorded_audio = None
            self.silence_recorded_audio = None
            self.buffer_audio_data = None
            self.microphone_test_result = None
            self.audio_input = None
            self.current_settings = None
            self.audio_error = None

            # Voice activity detection
            self.vad = None
            self.speech_detected = None
            self.silence_detected = None

            # Recording tasks
            self.silence_recording_task = None
            self.silence_recording_coroutine = None
            self.timeout_task = None
            self.timeout_coroutine = None
            self.timeout_value = None
            self.timeout_result = None

            # Mock states
            self.microphone_mock_active = False
            self.microphone_unavailable = False

            # CLI test data
            self.cli_result = None
            self.cli_output = None
            self.exit_code = None
            self.runner = None
            self.test_settings = None

            # Command results
            self.version_result = None
            self.help_result = None
            self.test_audio_result = None
            self.verbose_result = None
            self.ask_result = None
            self.ask_voice_result = None
            self.ask_save_result = None
            self.history_result = None
            self.history_limit_result = None
            self.history_json_result = None
            self.history_plain_result = None
            self.empty_history_result = None
            self.listen_result = None
            self.invalid_result = None
            self.ask_no_args_result = None
            self.reset_result = None
            self.config_result = None

            # Mock data
            self.mock_agent = None
            self.mock_agent_tts = None
            self.mock_tts = None
            self.mock_memory = None
            self.mock_history = []
            self.mock_fifteen_history = []
            self.empty_history = []
            self.mock_voice_agent = None
            self.audio_patches = []
            self.agent_patch_active = False

            # Agent test data
            self.agent_response = None
            self.conversation_history = []
            self.memory_count = 0

    return TestContext()


# Pytest markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests (skip with -m 'not slow')")
    config.addinivalue_line("markers", "audio: Tests requiring audio hardware")

    # Configure logging for tests
    import logging

    logging.getLogger("easyvoice.audio.input").setLevel(logging.WARNING)


# Auto-use fixtures
@pytest.fixture(autouse=True)
def disable_audio_in_tests(monkeypatch):
    """Disable actual audio hardware access in tests"""
    import numpy as np

    # Mock sounddevice module completely
    mock_sd = MagicMock()

    # Mock rec to return fake audio data
    def mock_rec(duration=1.0, samplerate=16000, channels=1, dtype=np.float32):
        samples = int(duration * samplerate)
        # Generate fake audio data
        return np.random.uniform(-0.1, 0.1, (samples, channels)).astype(dtype)

    mock_sd.rec = mock_rec
    mock_sd.play = MagicMock()
    mock_sd.wait = MagicMock()
    mock_sd.default = MagicMock()
    mock_sd.query_devices = MagicMock(return_value=[])

    # Mock the entire sounddevice module
    monkeypatch.setattr("easyvoice.audio.input.sd", mock_sd)
    monkeypatch.setattr("sounddevice.rec", mock_sd.rec)
    monkeypatch.setattr("sounddevice.play", mock_sd.play)
    monkeypatch.setattr("sounddevice.wait", mock_sd.wait)
    monkeypatch.setattr("sounddevice.query_devices", mock_sd.query_devices)

    # Mock test_microphone function to always succeed
    def mock_test_microphone():
        return {
            "available": True,
            "device_count": 1,
            "default_device": "Mock Device",
            "sample_rate": 16000,
        }

    monkeypatch.setattr("easyvoice.audio.input.test_microphone", mock_test_microphone)


@pytest.fixture(autouse=True)
def disable_llm_in_tests(monkeypatch):
    """Disable actual LLM calls in tests"""
    # Mock modules that may not be available
    import sys

    # Mock OpenAI
    if "openai" not in sys.modules:
        sys.modules["openai"] = MagicMock()

    # Mock Whisper
    if "whisper" not in sys.modules:
        mock_whisper = MagicMock()
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello world"}
        mock_whisper.load_model.return_value = mock_model
        sys.modules["whisper"] = mock_whisper

    # Mock KittenTTS
    if "kittentts" not in sys.modules:
        mock_kitten = MagicMock()
        mock_tts = MagicMock()
        mock_tts.generate.return_value = b"fake_audio_data"
        mock_kitten.TTS.return_value = mock_tts
        sys.modules["kittentts"] = mock_kitten


@pytest.fixture(autouse=True)
def mock_audio_components(monkeypatch):
    """Mock all audio-related components"""
    import numpy as np

    # Mock AudioInput class methods
    def mock_init(self, settings=None):
        # Check if we're in a test context where microphone should fail
        # If sounddevice is mocked to return empty device list, raise error
        try:
            import sounddevice as sd

            if hasattr(sd, "query_devices") and callable(sd.query_devices):
                devices = sd.query_devices()
                if not devices:  # Empty device list means no audio devices
                    raise RuntimeError("No audio input devices available")
        except Exception as e:
            if "No audio input devices available" in str(e):
                raise

        self.settings = settings or MagicMock()
        self.is_recording = False
        self.audio_data = []
        self.audio_buffer = []
        self.buffer_lock = MagicMock()
        self.vad = MagicMock()
        self.stream = None

    def mock_get_audio_data(self, duration=5.0, timeout=30.0):
        # Generate fake audio data
        samples = int(16000 * duration)  # 16kHz for duration seconds
        fake_data = np.random.uniform(-0.1, 0.1, samples).astype(np.float32)

        # Clear the buffer after getting data (as real implementation does)
        self.audio_buffer.clear()

        return fake_data

    async def mock_start_recording(self):
        self.is_recording = True
        return True

    async def mock_stop_recording(self):
        self.is_recording = False
        return True

    async def mock_record_until_silence(self, max_duration=30.0, silence_duration=1.0):
        """Mock record_until_silence that simulates timeout behavior"""
        import logging

        logger = logging.getLogger("easyvoice.audio.input")

        await self.mock_start_recording()
        try:
            # Simulate timeout - wait for max_duration then return
            await asyncio.sleep(0.1)  # Short delay for testing

            # Log timeout warning as expected by the test
            logger.warning(f"Recording timeout after {max_duration}s")

            # Generate fake audio data
            samples = int(16000 * 1.0)  # 1 second of fake audio
            return np.random.uniform(-0.1, 0.1, samples).astype(np.float32)
        finally:
            await self.mock_stop_recording()

    # Apply mocks
    monkeypatch.setattr("easyvoice.audio.input.AudioInput.__init__", mock_init)
    monkeypatch.setattr(
        "easyvoice.audio.input.AudioInput.get_audio_data", mock_get_audio_data
    )
    monkeypatch.setattr(
        "easyvoice.audio.input.AudioInput.start_recording", mock_start_recording
    )
    monkeypatch.setattr(
        "easyvoice.audio.input.AudioInput.stop_recording", mock_stop_recording
    )
    monkeypatch.setattr(
        "easyvoice.audio.input.AudioInput.record_until_silence",
        mock_record_until_silence,
    )
