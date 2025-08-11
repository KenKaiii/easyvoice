# 🚀 EasyVoice Installation Guide

## One-Command Installation (Recommended)

The easiest way to install EasyVoice is using our universal installer that automatically detects your operating system and runs the appropriate installation script.

### Linux/macOS
```bash
curl -sSL https://raw.githubusercontent.com/your-username/easyvoice/main/install-easyvoice | bash
```

### Windows (PowerShell)
```powershell
irm https://raw.githubusercontent.com/your-username/easyvoice/main/install-easyvoice.ps1 | iex
```

## What the Installer Does

1. **System Check**: Verifies Python 3.10+ is installed
2. **OS Detection**: Automatically detects your operating system
3. **Smart Installation**: Downloads and runs the appropriate platform-specific installer
4. **Verification**: Confirms installation was successful
5. **Next Steps**: Shows you how to get started

## Requirements

- **Python**: 3.10 or higher
- **pip**: Usually comes with Python
- **Internet connection**: For downloading dependencies

## Installation Options

The installer will try these methods in order:

1. **Regular pip install**: `pip install --user .`
2. **System packages fallback**: `pip install --user . --break-system-packages` 
3. **pipx alternative**: If you have pipx installed

## Audio Dependencies (Optional)

For voice features, install additional audio dependencies:

```bash
# After basic installation
pip install 'easyvoice[audio]'
```

This includes:
- `sounddevice` - Audio input/output
- `soundfile` - Audio file handling  
- `openai-whisper` - Speech-to-text
- `torch` - ML framework for Whisper
- `kittentts` - Text-to-speech

## Verification

After installation, verify it works:

```bash
easyvoice --version
easyvoice --help
```

## Configuration

### OpenAI API (Recommended)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Local Ollama (Alternative)
```bash
# Install Ollama first: https://ollama.ai
ollama pull llama2  # or your preferred model
```

## Troubleshooting

### Python Version Issues
```bash
# Check your Python version
python --version
python3 --version

# If < 3.10, install newer Python
```

### Permission Issues (Linux/macOS)
```bash
# Use --user flag (installer does this automatically)
pip install --user easyvoice
```

### Windows PATH Issues
If `easyvoice` command not found after installation:
1. Restart your terminal/command prompt
2. Add Python Scripts directory to PATH
3. Use `python -m easyvoice` as alternative

### Manual Installation
If the installer fails, you can install manually:

```bash
# Clone the repository
git clone https://github.com/your-username/easyvoice.git
cd easyvoice

# Install with pip
pip install --user .

# Or with pipx
pipx install .
```

## Getting Started

Once installed:

```bash
# Interactive mode (recommended for first time)
easyvoice

# Quick chat
easyvoice chat

# Ask a single question
easyvoice ask "What's the weather like?"

# Show all options
easyvoice --help
```

## Need Help?

- 📖 [Full Documentation](README.md)
- 🐛 [Report Issues](https://github.com/your-username/easyvoice/issues)
- 💬 [Discussions](https://github.com/your-username/easyvoice/discussions)

---

**Happy voice chatting! 🎤**