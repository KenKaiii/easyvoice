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
    conn.execute("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

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


# Auto-use fixtures
@pytest.fixture(autouse=True)
def disable_audio_in_tests(monkeypatch):
    """Disable actual audio hardware access in tests"""
    # Mock sounddevice to prevent actual audio access
    mock_sd = MagicMock()
    monkeypatch.setattr("sounddevice.rec", mock_sd.rec)
    monkeypatch.setattr("sounddevice.play", mock_sd.play)
    monkeypatch.setattr("sounddevice.wait", mock_sd.wait)


@pytest.fixture(autouse=True)
def disable_llm_in_tests(monkeypatch):
    """Disable actual LLM calls in tests"""
    # This will be implemented when we create the LLM module
    pass
