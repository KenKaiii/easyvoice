# 🎤 EasyVoice CLI

Lightweight voice agent CLI with real-time audio processing and tool calling capabilities.

**Created by Ken Kai** - AI Developer  
*Follow more AI projects: Ken Kai does AI*

> [!TIP]
> **Want to build your own apps with AI?** A free 7-day build challenge and 20+ step-by-step projects are in my free community: **[skool.com/raid](https://www.skool.com/raid)**

## ✨ Features

- **Interactive CLI**: No flags needed - just run `easyvoice`
- **20-Message Memory**: Sliding window conversation memory
- **Tool Integration**: Time, system info, weather tools
- **BDD Testing**: 80%+ test coverage with comprehensive scenarios
- **Timeout Protection**: All operations have configurable timeouts
- **Modular Design**: Clean separation of concerns
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **Graceful Degradation**: Works without external services

## 🚀 Quick Start

### One-Command Installation (Recommended)

**Linux/macOS:**
```bash
curl -sSL https://raw.githubusercontent.com/KenKaiii/easyvoice/main/install-easyvoice | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/KenKaiii/easyvoice/main/install-easyvoice.ps1 | iex
```

### Alternative: Platform-Specific Installers

**Linux/macOS:**
```bash
curl -sSL https://raw.githubusercontent.com/KenKaiii/easyvoice/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/KenKaiii/easyvoice/main/install.ps1 | iex
```

**Windows (Command Prompt):**
```cmd
curl -o install.bat https://raw.githubusercontent.com/easyvoice/easyvoice/main/install.bat && install.bat
```

### Option 2: Manual Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/easyvoice/easyvoice.git
   cd easyvoice
   ```

2. **Install with pip:**
   ```bash
   pip install --user .
   ```

3. **Optional - Install audio dependencies:**
   ```bash
   pip install --user ".[audio]"
   ```

4. **Run EasyVoice:**
   ```bash
   easyvoice
   ```

## 📋 Requirements

- **Python 3.10+**
- **pip** (usually comes with Python)

### Optional Dependencies

For voice processing features:
- `sounddevice` - Audio input/output
- `soundfile` - Audio file handling  
- `openai-whisper` - Speech-to-text
- `torch` - ML framework for audio processing

## 🎯 Usage

### Interactive Mode
```bash
easyvoice
```

This launches the interactive CLI with available commands:
- `chat` - Start text conversation
- `voice` - Start voice conversation (requires audio deps)
- `ask` - Ask a single question
- `history` - View conversation history
- `status` - Show system status
- `test` - Test audio system
- `config` - Show configuration
- `help` - Show available commands
- `quit` - Exit

### Alternative Execution
```bash
python -m easyvoice
```

## 🛠️ Development

### Setup Development Environment

1. **Clone and install in development mode:**
   ```bash
   git clone https://github.com/easyvoice/easyvoice.git
   cd easyvoice
   pip install -e ".[dev]"
   ```

2. **Run tests:**
   ```bash
   # All tests
   python run_basic_tests.py
   
   # BDD scenarios
   pytest tests/
   
   # Integration tests
   python test_complete_system.py
   
   # Interactive CLI test
   python test_cli_interactive.py
   ```

### Project Structure

```
easyvoice/
├── easyvoice/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   ├── interactive_cli.py   # CLI interface
│   ├── config/
│   │   └── settings.py      # Configuration management
│   ├── agent/
│   │   ├── core.py         # Main VoiceAgent
│   │   ├── memory.py       # Conversation memory
│   │   ├── llm_custom.py   # LLM interface
│   │   └── tools.py        # Tool system
│   ├── audio/
│   │   ├── capture.py      # Audio input
│   │   ├── stt.py          # Speech-to-text
│   │   └── tts.py          # Text-to-speech
│   └── ui/
│       └── indicators.py   # UI components
├── tests/
│   ├── features/           # BDD scenarios
│   └── step_definitions/   # BDD step implementations
├── install.sh              # Linux/macOS installer
├── install.bat             # Windows batch installer
├── install.ps1             # Windows PowerShell installer
└── pyproject.toml          # Project configuration
```

## 🔧 Configuration

EasyVoice uses environment variables for configuration:

```bash
# LLM Settings
export EASYVOICE_MODEL_NAME="llama3.2"
export EASYVOICE_OLLAMA_HOST="http://localhost:11434"
export EASYVOICE_LLM_TIMEOUT=30

# Audio Settings
export EASYVOICE_SAMPLE_RATE=16000
export EASYVOICE_STT_TIMEOUT=10
export EASYVOICE_TTS_TIMEOUT=10

# Memory Settings
export EASYVOICE_MAX_MESSAGES=20
export EASYVOICE_DB_PATH="conversations.db"

# Debug
export EASYVOICE_DEBUG=true
```

## 🧪 Testing Approach

EasyVoice follows **Behavior-Driven Development (BDD)**:

1. **Write failing tests first** (Red)
2. **Implement minimal code** to pass tests (Green)  
3. **Refactor** for quality (Refactor)

Test coverage includes:
- Unit tests for individual components
- BDD scenarios for user workflows
- Integration tests for complete pipeline
- Timeout and error handling
- Cross-platform compatibility

## 🔌 Architecture

### Simple > Clever Design

- **No langchain dependency** - Custom lightweight LLM interface
- **Graceful degradation** - Works without external services
- **Timeout protection** - All operations have configurable timeouts
- **Modular components** - Easy to test and maintain
- **Async throughout** - Non-blocking operations

### Tool System

Tools are registered with decorators and executed asynchronously:

```python
@tool(description="Get current time")
async def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S")
```

### Memory Management

20-message sliding window with SQLite persistence:
- Automatic cleanup of old messages
- Conversation context preservation
- Cross-session persistence

## 🌍 Cross-Platform Support

### Linux
- Tested on Ubuntu 20.04+
- Uses `~/.local/bin/easyvoice`

### macOS  
- Tested on macOS 11+
- Uses `~/Library/Python/3.x/bin/easyvoice`

### Windows
- Tested on Windows 10+
- Uses `%APPDATA%\Python\Scripts\easyvoice.exe`
- PowerShell and Command Prompt support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write BDD tests first
4. Implement the feature
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Homepage**: https://github.com/easyvoice/easyvoice
- **Issues**: https://github.com/easyvoice/easyvoice/issues
- **Discussions**: https://github.com/easyvoice/easyvoice/discussions

---

**Built with ❤️ using BDD methodology and "Simple > Clever" philosophy**