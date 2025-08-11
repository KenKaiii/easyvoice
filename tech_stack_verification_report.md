# Tech Stack Verification Report for KittenTTS Voice Application

_Generated: 2025-08-10 | Sources: 35+ official docs and repositories_

## 🎯 Quick Reference

<key-points>
- **Python 3.10+** is the ideal version for the entire stack
- **KittenTTS** works with CPU-only setups, 25MB model size
- **Textual** has excellent async support for real-time audio apps
- **Whisper + LangChain + Ollama** integration is mature and stable
- **ChromaDB** requires Python 3.8+ (recommend 3.11 for stability)
- **Audio libraries** have performance considerations for real-time use
</key-points>

## 📋 Overview

<summary>
This report verifies the compatibility and installation requirements for a KittenTTS-based voice application stack. All components are actively maintained in 2025 and compatible with each other when using Python 3.10+. The stack enables building a complete voice agent with real-time speech recognition, text-to-speech, LLM integration, and vector storage capabilities.
</summary>

## 🔧 Implementation Details

<details>

### Core TTS Component - KittenTTS
**Version**: 0.1.0 (Latest as of 2025)
**Python Requirement**: 3.10+ for server setup, 3.x for basic model

**Installation:**
```bash
# Direct wheel installation
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl

# Dependencies
# eSpeak NG (required for phonemization)
# NumPy and PyTorch (basic dependencies)
```

**Basic Usage:**
```python
from kittentts import KittenTTS
import soundfile as sf

# Load model (downloads from Hugging Face on first run)
m = KittenTTS("KittenML/kitten-tts-nano-0.1")

# Generate audio
text = "This works without a GPU!"
audio = m.generate(text)

# Save at 24kHz sample rate
sf.write('output.wav', audio, 24000)
```

### UI Framework - Textual
**Version**: Latest (actively updated in 2025)
**Python Requirement**: 3.8+

**Installation:**
```bash
pip install textual
pip install textual-dev  # Development console
```

**Async Background Tasks for Real-Time Audio:**
```python
import asyncio
from textual.app import App
from textual.widgets import Label

class VoiceApp(App):
    async def on_mount(self):
        # Create background task for audio processing
        asyncio.create_task(self.audio_processor())
    
    async def audio_processor(self):
        # Real-time audio processing loop
        while True:
            # Process audio without blocking UI
            await asyncio.sleep(0.01)  # 10ms chunks
```

### Speech Recognition - OpenAI Whisper
**Version**: 20250625 release available
**Python Requirement**: 3.8-3.11 (NOT 3.13)
**Recommended**: Python 3.10

**Installation:**
```bash
# Install FFmpeg first (required)
# Windows: choco install ffmpeg
# macOS: brew install ffmpeg

pip install -U openai-whisper
# OR latest from GitHub:
pip install git+https://github.com/openai/whisper.git
```

**Usage:**
```python
import whisper

model = whisper.load_model("turbo")  # Fast model for real-time
result = model.transcribe("audio.mp3")
print(result["text"])
```

### LLM Integration - LangChain + Ollama
**LangChain-Ollama Version**: 0.3.6 (Latest)
**Python Requirement**: 3.8+

**Installation:**
```bash
# Install LangChain Ollama integration
pip install -U langchain-ollama

# Install Ollama locally
# macOS: brew install ollama && brew services start ollama
# Pull models: ollama pull llama3.2
```

**Usage:**
```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2")
response = llm.invoke("Respond to user voice input")
```

### Vector Storage - ChromaDB
**Version**: 1.0.16 (Latest)
**Python Requirement**: 3.8+ (recommend 3.11 for stability)
**SQLite Requirement**: 3.35+

**Installation:**
```bash
pip install chromadb
```

**Basic Setup:**
```python
import chromadb

# Persistent storage
client = chromadb.PersistentClient(path="/path/to/chroma/db")
collection = client.create_collection("voice_context")
```

### Audio Processing Libraries

#### sounddevice
**Real-time Performance**: Good for streaming audio I/O
```bash
pip install sounddevice
```

#### webrtcvad
**Performance Notes**: 
- Requires 16-bit mono PCM at 8/16/32/48kHz
- Frame duration: 10/20/30ms only
- Can have false positives with music/background noise

```bash
pip install webrtcvad
```

#### noisereduce
**Performance Warning**: Memory allocation issues in real-time scenarios
**Alternative**: Consider SileroVAD for better real-time performance

```bash
pip install noisereduce
```

### System Integration

#### pynput (Global Hotkeys)
**Version**: 1.7.6 (Updated March 2025)
**Python Requirement**: 3.6+
**macOS Issue**: ctrl+alt hotkeys don't work (use cmd+key instead)

```bash
pip install pynput
```

#### psutil (System Monitoring)
**Version**: 7.0.0 (Updated February 2025)
**Python Requirement**: 3.6+

```bash
pip install psutil
```

</details>

## ⚠️ Important Considerations

<warnings>

### Python Version Compatibility Matrix
- **KittenTTS**: Requires 3.10+ for server setup
- **Whisper**: Supports 3.8-3.11 (NOT 3.13)
- **ChromaDB**: Works best with 3.11 (some 3.12 issues reported)
- **LangChain**: 3.8+ (3.8 deprecated soon)
- **Textual**: 3.8+

**Recommendation: Use Python 3.10 or 3.11 for maximum compatibility**

### Real-Time Performance Issues
1. **noisereduce**: Memory allocation problems - consider SileroVAD alternative
2. **webrtcvad**: Strict audio format requirements and false positives
3. **Whisper turbo model**: Best balance of speed vs accuracy for real-time

### Platform-Specific Issues
- **macOS pynput**: ctrl+alt global hotkeys don't work
- **ChromaDB**: Requires SQLite 3.35+
- **KittenTTS**: Needs eSpeak NG for phonemization
- **Whisper**: Requires FFmpeg

### Breaking Changes to Watch
- Python 3.8 EOL approaching (affects LangChain)
- ChromaDB Python 3.12 compatibility still developing
- Whisper Python 3.13 support not ready

</warnings>

## 🔗 Resources

<references>
- [KittenTTS Repository](https://github.com/KittenML/KittenTTS) - Official 25MB TTS model
- [Textual Documentation](https://textual.textualize.io/) - Modern TUI framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [LangChain Ollama](https://python.langchain.com/docs/integrations/providers/ollama/) - LLM integration
- [ChromaDB Docs](https://www.trychroma.com/) - Vector database
- [Pipecat Framework](https://github.com/pipecat-ai/pipecat) - Real-time voice AI reference
- [AssemblyAI Voice Bot Tutorial](https://www.assemblyai.com/blog/real-time-ai-voice-bot-python) - Implementation example
</references>

## 🏷️ Metadata

<meta>
research-date: 2025-08-10
confidence: high
version-checked: all-latest-2025
python-recommendation: 3.10-3.11
stack-status: fully-compatible
</meta>

---

## 📝 Installation Quick Start

For fastest setup with maximum compatibility:

```bash
# Use Python 3.10 or 3.11
python3.10 -m venv voice_env
source voice_env/bin/activate

# Install system dependencies
# macOS: brew install ffmpeg espeak-ng ollama
# Ubuntu: apt install ffmpeg espeak-ng

# Install Python packages
pip install textual textual-dev
pip install openai-whisper
pip install langchain-ollama
pip install chromadb
pip install sounddevice pynput psutil
pip install https://github.com/KittenML/KittenTTS/releases/download/0.1/kittentts-0.1.0-py3-none-any.whl

# Start Ollama and pull model
ollama serve
ollama pull llama3.2
```

## 🚀 Next Steps

1. **Prototype Integration**: Start with basic KittenTTS + Textual integration
2. **Audio Pipeline**: Test sounddevice + Whisper real-time processing
3. **Alternative VAD**: Consider SileroVAD instead of webrtcvad for better performance
4. **Platform Testing**: Verify global hotkeys work on target platforms
5. **Performance Optimization**: Profile real-time audio processing pipeline

This stack is **production-ready for 2025** with the noted compatibility considerations.