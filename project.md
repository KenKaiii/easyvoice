# EasyVoice CLI Project

**Goal**: Lightweight voice agent CLI with real-time audio processing and tool calling

## 🎯 Core Requirements

- **Lightweight**: Simple > Clever
- **Real-time**: Listen → Process → Respond cycle
- **Memory**: 20 message conversation history
- **Tools**: Function calling capabilities
- **Testing**: BDD approach, 80% coverage
- **Visual**: Clean interface with live DB indicators

## 🏗️ Architecture

```
CLI Entry → Audio Input → Speech Processing → LLM Agent → Response Output
              ↓              ↓                  ↓           ↑
         [Microphone]   [Whisper STT]    [Memory DB]  [KittenTTS]
```

## 📦 Project Structure

```
easyvoice/
├── pyproject.toml              # Dependencies & config
├── README.md                   # Usage instructions
├── easyvoice/
│   ├── __init__.py
│   ├── cli.py                  # Entry point & argument parsing
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── input.py            # Microphone capture
│   │   ├── stt.py              # Whisper speech-to-text
│   │   └── tts.py              # KittenTTS text-to-speech
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py             # Main conversation loop
│   │   ├── memory.py           # 20-message sliding window
│   │   ├── tools.py            # Available function calls
│   │   └── llm.py              # Ollama integration
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── display.py          # Status indicators & output
│   │   └── indicators.py       # Live DB activity visual
│   └── config/
│       ├── __init__.py
│       └── settings.py         # Configuration management
├── tests/
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_audio.py          # Audio module BDD tests
│   ├── test_agent.py          # Agent logic BDD tests
│   ├── test_ui.py             # UI component tests
│   └── features/              # BDD feature files
│       ├── conversation.feature
│       ├── memory.feature
│       └── tools.feature
└── scripts/
    ├── setup.sh               # Environment setup
    └── run_tests.sh          # Test execution with coverage
```

## 🛠️ Technology Stack

**Core**:
- Python 3.10+ (compatibility)
- Click (CLI framework)
- asyncio (async processing)

**Audio**:
- sounddevice (audio I/O)
- openai-whisper (STT)
- KittenTTS (TTS)

**Agent**:
- langchain-ollama (LLM)
- sqlite3 (lightweight memory)

**Testing**:
- pytest (testing framework)
- pytest-bdd (BDD scenarios)
- pytest-cov (coverage reporting)
- pytest-asyncio (async testing)
- pytest-timeout (test timeouts)

**UI**:
- rich (beautiful terminal output)
- pyfiglet (ASCII art headers)

**Dev Tools**:
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## 📋 Implementation Plan

### Phase 1: Core CLI (Week 1)
1. **Project setup** with pyproject.toml
2. **CLI entry point** with Click
3. **Basic audio input/output** pipeline
4. **Simple conversation loop**
5. **Unit tests** for core modules

### Phase 2: Agent Intelligence (Week 2)
1. **Whisper STT integration**
2. **KittenTTS integration**
3. **Ollama LLM connection**
4. **Memory system** (20 messages)
5. **BDD feature tests**

### Phase 3: Tools & UI (Week 3)
1. **Function calling** system
2. **Live status indicators**
3. **DB activity visualization**
4. **Error handling & timeouts**
5. **Integration tests**

### Phase 4: Polish & Deploy (Week 4)
1. **80% test coverage** verification
2. **Performance optimization**
3. **Documentation completion**
4. **User acceptance testing**

## 🔧 Key Features

### CLI Interface
```bash
# Start listening mode
easyvoice listen

# One-shot mode
easyvoice ask "What's the weather?"

# Show conversation history
easyvoice history

# Test audio pipeline
easyvoice test-audio
```

### Live Indicators
```
🎤 EasyVoice CLI v1.0
┌─ Status ─────────────────────────┐
│ 🟢 Ready      💾 DB: 15 msgs    │
│ 🎯 Listening  ⚡ Tools: 4 avail │
└──────────────────────────────────┘

👤 You: "What's the current time?"
🤖 Agent: "It's currently 2:30 PM EST"
    └─ 🔧 Used: system_time_tool
    └─ 💾 Saved to memory (16/20)
```

### Memory Management
- **Sliding window**: Keep latest 20 messages
- **SQLite storage**: Lightweight, fast queries
- **Context aware**: Include relevant history in prompts

### Tool System
```python
@tool
def get_weather(location: str) -> str:
    """Get current weather for location"""
    # Implementation with timeout
    
@tool  
def system_info() -> dict:
    """Get system information"""
    # Implementation with error handling
```

## 🧪 Testing Strategy

### BDD Scenarios
```gherkin
Feature: Voice Conversation
  Scenario: User asks a simple question
    Given the voice agent is listening
    When I say "Hello"
    Then the agent should respond with a greeting
    And the conversation should be saved to memory
```

### Coverage Requirements
- **80% minimum** across all modules
- **Timeout protection** on all external calls
- **Mock external dependencies** (LLM, TTS, STT)
- **Error case testing** for network/audio failures

### Test Commands
```bash
# Run all tests with coverage
pytest --cov=easyvoice --cov-report=html --timeout=30

# BDD scenarios only
pytest tests/features/ -v

# Quick smoke tests
pytest -m "not slow" --timeout=10
```

## ⚙️ Configuration

### settings.py
```python
@dataclass
class Settings:
    # Audio
    sample_rate: int = 16000
    chunk_size: int = 1024
    
    # LLM
    model_name: str = "llama3.2"
    max_tokens: int = 500
    
    # Memory
    max_messages: int = 20
    db_path: str = "memory.db"
    
    # Timeouts
    stt_timeout: int = 30
    tts_timeout: int = 15
    llm_timeout: int = 45
```

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone <repo>
cd easyvoice
./scripts/setup.sh

# Install dependencies
pip install -e .

# Test installation
easyvoice test-audio
```

### First Run
```bash
# Start voice agent
easyvoice listen

# Or one-shot mode
easyvoice ask "Hello, can you hear me?"
```

## ✅ Success Criteria

- [ ] **Lightweight**: < 100MB memory footprint
- [ ] **Responsive**: < 2s response time
- [ ] **Reliable**: Handles network/audio failures gracefully
- [ ] **Tested**: 80% coverage, all timeouts respected
- [ ] **Clean**: Clear visual feedback and status
- [ ] **Modular**: Easy to extend with new tools

---

**Simple, tested, working voice AI in CLI form** 🎯