# Voice Agent Tech Stack

## 🎯 Project Vision
A powerful desktop voice agent that can perform tasks through natural conversation, featuring:
- Real-time speech recognition & synthesis
- Beautiful terminal UI with modern design
- Plugin/command architecture for extensibility
- Multi-modal interaction (voice + text + visual)

## 🏗️ Core Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Voice Processing │───▶│   Agent Core    │
│                 │    │                  │    │                 │
│ • Microphone    │    │ • Speech-to-Text │    │ • Intent Parser │
│ • Hotkey        │    │ • Noise Gate     │    │ • Task Executor │
│ • Push-to-Talk  │    │ • VAD            │    │ • Memory        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                  │
                                  ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Output   │◀───│   TTS & UI       │◀───│   Capabilities  │
│                 │    │                  │    │                 │
│ • KittenTTS     │    │ • Textual UI     │    │ • File System   │
│ • System Audio  │    │ • Visual Feedback│    │ • Web Search    │
│ • Voice Effects │    │ • Logs/History   │    │ • Code Exec     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### **UI Framework**
- **[Textual](https://github.com/Textualize/textual)** - Modern Python TUI framework
  - CSS-like styling for beautiful interfaces
  - Component-based architecture
  - Async support for real-time updates
  - Web deployment capability

### **Speech Recognition (Input)**
- **[OpenAI Whisper](https://github.com/openai/whisper)** - Converts user speech → text
  - Local processing (privacy-first)
  - Turbo model recommended for real-time performance
  - Multi-language support
  - High accuracy, handles accents well

### **Text-to-Speech (Output)**
- **[KittenTTS](https://github.com/KittenML/KittenTTS)** - Converts agent text → speech
  - 25MB model size, CPU-optimized
  - Real-time synthesis (no GPU required)
  - 8 voice options (4M/4F)
  - 24kHz output quality

### **Audio Processing**
- **[sounddevice](https://github.com/spatialaudio/python-sounddevice)** - Low-latency audio I/O
- **SileroVAD** - Voice Activity Detection (recommended over webrtcvad)
- **[noisereduce](https://github.com/timsainb/noisereduce)** - Background noise reduction
  - ⚠️ Consider SileroVAD alternative for better real-time performance

### **Agent Intelligence**
- **[LangChain](https://github.com/langchain-ai/langchain)** - LLM orchestration & tools
- **[Ollama](https://github.com/ollama/ollama)** - Local LLM inference (Llama 3.2, etc.)
- **[ChromaDB](https://github.com/chroma-core/chroma)** - Vector memory/context storage

### **System Integration**
- **[pynput](https://github.com/moses-palmer/pynput)** - Global hotkeys & system control
- **[psutil](https://github.com/giampaolo/psutil)** - System monitoring & control
- **[rich](https://github.com/Textualize/rich)** - Beautiful terminal output (Textual dependency)

## 📦 Core Dependencies

```toml
[tool.poetry.dependencies]
python = "^3.10,<3.13"  # Verified compatibility range

# UI Framework
textual = "^0.85.0"
rich = "^13.0.0"

# Audio Processing  
sounddevice = "^0.4.6"
# webrtcvad = "^2.0.10"  # Consider SileroVAD instead
noisereduce = "^3.0.0"  # Note: has real-time performance issues

# Speech Recognition (Speech → Text)
openai-whisper = "^20250625"  # Latest 2025 release
torch = "^2.1.0"  # For Whisper

# Text-to-Speech (Text → Speech)
# kittentts = "0.1.0" - Install via wheel (see setup section)
soundfile = "^0.12.0"  # Required for KittenTTS audio output

# Agent & LLM
langchain-ollama = "^0.3.6"  # Latest Ollama integration
chromadb = "^1.0.16"  # Latest stable release

# System Integration
pynput = "^1.7.6"
psutil = "^5.9.0"

# Utilities
typer = "^0.12.0"  # CLI framework
pydantic = "^2.5.0"  # Data validation
asyncio = "*"
```

## 🏃 Getting Started

### 1. Project Setup (Python 3.10-3.11 Required)
```bash
# Use Python 3.10 or 3.11 for maximum compatibility
python3.10 -m venv voice_env  # or python3.11
source voice_env/bin/activate  # Linux/Mac
pip install poetry

# Install system dependencies first
# macOS: brew install ffmpeg espeak-ng ollama
# Ubuntu: apt install ffmpeg espeak-ng
```

### 2. Install Dependencies
```bash
# Core Python packages
poetry add textual rich sounddevice soundfile
poetry add openai-whisper torch langchain-ollama chromadb
poetry add pynput psutil typer pydantic

# KittenTTS (special wheel install)
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl

# Start Ollama service
ollama serve &
ollama pull llama3.2
```

### 3. Project Structure
```
easyvoice/
├── pyproject.toml
├── README.md
├── voice_agent/
│   ├── __init__.py
│   ├── main.py              # Entry point
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py           # Main Textual app
│   │   ├── components/      # UI components
│   │   └── styles.tcss      # Textual CSS
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── input.py         # Microphone & STT
│   │   ├── output.py        # TTS & audio output
│   │   └── processing.py    # VAD, noise reduction
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── core.py          # Main agent logic
│   │   ├── memory.py        # Context & history
│   │   ├── tools.py         # Available functions
│   │   └── llm.py           # LLM integration
│   └── config/
│       ├── __init__.py
│       ├── settings.py      # App configuration
│       └── voices.py        # Voice profiles
└── tests/
    ├── __init__.py
    ├── test_audio.py
    ├── test_agent.py
    └── test_ui.py
```

## 🎨 UI Design Philosophy

### Modern Terminal Aesthetics
- **Dark theme with accent colors**
- **Rounded corners and smooth animations**
- **Real-time waveform visualization**
- **Status indicators (listening, thinking, speaking)**

### Layout Zones
```
┌──────────────────────────────────────────────────────┐
│  🎤 EasyVoice Agent - Listening...            [⚙️]   │  Header
├──────────────────────────────────────────────────────┤
│                                                      │
│  ┌─ Conversation ─────────────────────────────────┐  │
│  │ 👤 You: "What's the weather like?"           │  │  Main
│  │ 🤖 Agent: "Let me check the weather for you" │  │  Area
│  │ 🌤️  Currently 72°F and sunny in SF           │  │
│  │                                               │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ System Status ──┐  ┌─ Audio Levels ────────────┐│
│  │ 🟢 Ready         │  │ ▓▓▓▓▓░░░░░ Input          ││  Info
│  │ 🎯 Mode: Voice   │  │ ▓▓▓░░░░░░░ Output         ││  Panel
│  │ 🧠 Model: Local  │  │                           ││
│  └──────────────────┘  └───────────────────────────┘│
├──────────────────────────────────────────────────────┤
│ [Space] Talk  [Tab] Type  [Ctrl+C] Quit  [F1] Help  │  Footer
└──────────────────────────────────────────────────────┘
```

## 🔧 Key Features

### Voice Pipeline Flow
```
User Speech → Whisper (STT) → Agent Processing → KittenTTS (TTS) → Audio Output
     ↑                                                                    ↓
  Microphone ←────────────── Complete Voice Conversation ──────────── Speakers
```

### Voice Interaction
- **Push-to-talk** (Space key) or **Voice Activity Detection**
- **Interrupt handling** - Stop agent mid-response
- **Voice profiles** - Different personalities/modes
- **Audio feedback** - Beeps, confirmations

### Intelligence Features
- **Context awareness** - Remembers conversation history
- **Task decomposition** - Breaks down complex requests
- **Tool integration** - File system, web search, calculations
- **Learning** - Adapts to user preferences

### System Integration
- **Global hotkeys** - Activate from any application
- **System tray** - Minimize to background
- **Notification support** - Visual alerts
- **Session persistence** - Resume conversations

## 🔮 Future Enhancements

### Advanced Features
- **Multi-modal input** - Voice + image + text
- **Plugin system** - Custom command extensions  
- **Cloud sync** - Conversation backup
- **Voice cloning** - Personalized TTS
- **Gesture control** - Hand/eye tracking

### Integration Options
- **Browser extension** - Web page interaction
- **IDE plugins** - Code assistance
- **Smart home** - IoT device control
- **Calendar/Email** - Productivity features

## 🚀 Development Roadmap

### Phase 1: Core Foundation (Week 1-2)
- [x] Basic project structure
- [ ] Audio input pipeline (Whisper STT)
- [ ] Audio output pipeline (KittenTTS)
- [ ] Simple Textual UI
- [ ] Voice conversation loop

### Phase 2: Agent Intelligence (Week 3-4)
- [ ] LangChain + Ollama integration
- [ ] Basic command processing
- [ ] Memory/context system (ChromaDB)
- [ ] Tool/function calling

### Phase 3: Polish & Features (Week 5-6)
- [ ] Beautiful UI design
- [ ] Voice profiles
- [ ] Global hotkeys
- [ ] Configuration system

### Phase 4: Advanced Features (Week 7+)
- [ ] Plugin architecture
- [ ] Advanced LLM integration
- [ ] Multi-modal capabilities
- [ ] Performance optimization

---

**Ready to build the future of voice interaction!** 🎯