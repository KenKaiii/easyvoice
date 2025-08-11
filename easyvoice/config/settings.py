"""Configuration settings for EasyVoice CLI"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Handle tomllib/tomli for different Python versions
if sys.version_info >= (3, 11):
    import tomllib  # type: ignore[import-not-found]
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None

# Handle tomli_w for writing TOML files
try:
    import tomli_w
except ImportError:
    tomli_w = None


@dataclass
class Settings:
    """EasyVoice configuration settings

    All settings can be overridden via environment variables using the
    EASYVOICE_ prefix (e.g., EASYVOICE_MODEL_NAME=llama3.2).
    """

    # Audio Configuration
    sample_rate: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_SAMPLE_RATE", "16000"))
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_CHUNK_SIZE", "1024"))
    )
    channels: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_CHANNELS", "1"))
    )  # Mono

    # Voice Activity Detection
    vad_threshold: float = field(
        default_factory=lambda: float(os.getenv("EASYVOICE_VAD_THRESHOLD", "0.5"))
    )
    silence_duration: float = field(
        default_factory=lambda: float(os.getenv("EASYVOICE_SILENCE_DURATION", "1.0"))
    )
    push_to_talk: bool = field(
        default_factory=lambda: os.getenv("EASYVOICE_PUSH_TO_TALK", "true").lower() == "true"
    )

    # Speech-to-Text (Whisper)
    whisper_model: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_WHISPER_MODEL", "base")
    )
    whisper_language: Optional[str] = field(
        default_factory=lambda: os.getenv("EASYVOICE_WHISPER_LANGUAGE")
    )

    # Text-to-Speech (KittenTTS)
    tts_model: str = field(
        default_factory=lambda: os.getenv(
            "EASYVOICE_TTS_MODEL",
            "auto"  # Let KittenTTS auto-download model
        )
    )
    tts_voice: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_TTS_VOICE", "0"))
    )  # Voice 0-7
    tts_speed: float = field(
        default_factory=lambda: float(os.getenv("EASYVOICE_TTS_SPEED", "1.0"))
    )

    # LLM Configuration
    llm_provider: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_LLM_PROVIDER", "openai")
    )
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_OPENAI_MODEL", "gpt-5-nano")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_MODEL_NAME", "llama3.2")
    )
    ollama_host: str = field(
        default_factory=lambda: os.getenv(
            "EASYVOICE_OLLAMA_HOST", "http://localhost:11434"
        )
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_MAX_TOKENS", "2000"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("EASYVOICE_TEMPERATURE", "0.7"))
    )

    # Memory Configuration
    max_messages: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_MAX_MESSAGES", "20"))
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_DB_PATH", "memory.db")
    )

    # Timeout Configuration (seconds)
    stt_timeout: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_STT_TIMEOUT", "30"))
    )
    tts_timeout: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_TTS_TIMEOUT", "15"))
    )
    llm_timeout: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_LLM_TIMEOUT", "45"))
    )
    session_timeout: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_SESSION_TIMEOUT", "300"))
    )  # 5 min

    # UI Configuration
    show_waveform: bool = field(
        default_factory=lambda: os.getenv("EASYVOICE_SHOW_WAVEFORM", "true").lower()
        == "true"
    )
    update_interval: float = field(
        default_factory=lambda: float(os.getenv("EASYVOICE_UPDATE_INTERVAL", "0.1"))
    )

    # Tool Configuration
    enable_tools: bool = field(
        default_factory=lambda: os.getenv("EASYVOICE_ENABLE_TOOLS", "true").lower()
        == "true"
    )
    max_tool_calls: int = field(
        default_factory=lambda: int(os.getenv("EASYVOICE_MAX_TOOL_CALLS", "5"))
    )

    # Debug Configuration
    debug: bool = field(
        default_factory=lambda: os.getenv("EASYVOICE_DEBUG", "false").lower() == "true"
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_LOG_LEVEL", "INFO")
    )
    save_audio: bool = field(
        default_factory=lambda: os.getenv("EASYVOICE_SAVE_AUDIO", "false").lower()
        == "true"
    )
    audio_save_dir: str = field(
        default_factory=lambda: os.getenv("EASYVOICE_AUDIO_SAVE_DIR", "./audio_logs")
    )

    def __post_init__(self) -> None:
        """Validate and process settings after initialization"""
        # Validate sample rate
        valid_sample_rates = [8000, 16000, 22050, 44100, 48000]
        if self.sample_rate not in valid_sample_rates:
            raise ValueError(
                f"Invalid sample_rate: {self.sample_rate}. "
                f"Must be one of {valid_sample_rates}"
            )

        # Validate chunk size (must be power of 2)
        if not (self.chunk_size & (self.chunk_size - 1)) == 0:
            raise ValueError(f"chunk_size must be a power of 2, got {self.chunk_size}")

        # Validate TTS voice selection
        if not 0 <= self.tts_voice <= 7:
            raise ValueError(f"tts_voice must be between 0-7, got {self.tts_voice}")

        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be between 0.0-2.0, got {self.temperature}"
            )

        # Validate timeouts
        for timeout_name in [
            "stt_timeout",
            "tts_timeout",
            "llm_timeout",
            "session_timeout",
        ]:
            timeout_value = getattr(self, timeout_name)
            if timeout_value <= 0:
                raise ValueError(
                    f"{timeout_name} must be positive, got {timeout_value}"
                )

        # Ensure database directory exists
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure audio save directory exists if saving audio
        if self.save_audio:
            audio_dir = Path(self.audio_save_dir)
            audio_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_file(cls, config_path: str) -> "Settings":
        """Load settings from a configuration file

        Args:
            config_path: Path to TOML, JSON, or YAML config file

        Returns:
            Settings instance with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is invalid
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        config_data = {}

        if config_file.suffix.lower() == ".toml":
            if tomllib is None:
                raise ImportError(
                    "TOML support not available. Install with: pip install tomli"
                )

            with open(config_file, "rb") as f:
                config_data = tomllib.load(f)
        elif config_file.suffix.lower() == ".json":
            import json

            with open(config_file, "r") as f:
                config_data = json.load(f)
        elif config_file.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml

                with open(config_file, "r") as f:
                    config_data = yaml.safe_load(f)
            except ImportError:
                raise ValueError("PyYAML is required to load YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")

        # Extract easyvoice section if it exists
        if "easyvoice" in config_data:
            config_data = config_data["easyvoice"]

        return cls(**config_data)

    def to_dict(self) -> dict:
        """Convert settings to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }

    def save_to_file(self, config_path: str) -> None:
        """Save current settings to a configuration file

        Args:
            config_path: Path where to save the config file
        """
        config_file = Path(config_path)
        config_data = {"easyvoice": self.to_dict()}

        config_file.parent.mkdir(parents=True, exist_ok=True)

        if config_file.suffix.lower() == ".toml":
            if tomli_w is None:
                raise ImportError(
                    "TOML writing support not available. "
                    "Install with: pip install tomli_w"
                )

            with open(config_file, "wb") as f:
                tomli_w.dump(config_data, f)
        elif config_file.suffix.lower() == ".json":
            import json

            with open(config_file, "w") as f:
                json.dump(config_data, f, indent=2)
        elif config_file.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml

                with open(config_file, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False)
            except ImportError:
                raise ValueError("PyYAML is required to save YAML config files")
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")

    def get_whisper_kwargs(self) -> dict:
        """Get Whisper-specific configuration as kwargs"""
        kwargs = {}
        if self.whisper_language:
            kwargs["language"] = self.whisper_language
        return kwargs

    def get_ollama_kwargs(self) -> dict:
        """Get Ollama-specific configuration as kwargs"""
        return {
            "base_url": self.ollama_host,
            "model": self.model_name,
            "temperature": self.temperature,
        }

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.debug or os.getenv("EASYVOICE_ENV") == "development"
