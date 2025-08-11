#!/bin/bash
# Cross-platform installation script for EasyVoice CLI
# Works on Linux, macOS, and Windows (via Git Bash/WSL)

set -e

echo "🎤 EasyVoice CLI Installation Script"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    print_status "Detected OS: $OS"
}

# Check Python version
check_python() {
    print_status "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python is not installed. Please install Python 3.10+ first."
        exit 1
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 10 ]]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
}

# Check pip
check_pip() {
    print_status "Checking pip installation..."
    
    if ! $PYTHON_CMD -m pip --version &> /dev/null; then
        print_error "pip is not installed. Please install pip first."
        exit 1
    fi
    
    print_success "pip is available"
}

# Install EasyVoice
install_easyvoice() {
    print_status "Installing EasyVoice CLI..."
    
    # Try user install first, then with break-system-packages if needed
    if $PYTHON_CMD -m pip install --user . 2>/dev/null; then
        print_success "EasyVoice CLI installed successfully!"
    elif $PYTHON_CMD -m pip install --user --break-system-packages .; then
        print_success "EasyVoice CLI installed successfully (with system packages override)!"
    elif command -v pipx &> /dev/null && pipx install .; then
        print_success "EasyVoice CLI installed successfully with pipx!"
    else
        print_error "All installation methods failed"
        print_status "Try manually: pip install --user --break-system-packages ."
        print_status "Or with pipx: pipx install ."
        exit 1
    fi
    
    # Optional: Install audio dependencies
    echo ""
    read -p "Do you want to install audio processing dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing audio dependencies (this may take a while)..."
        
        # Try the same installation method that worked for the base package
        $PYTHON_CMD -m pip install --user ".[audio]" 2>/dev/null
        
        if [[ $? -ne 0 ]]; then
            $PYTHON_CMD -m pip install --user --break-system-packages ".[audio]" 2>/dev/null
            
            if [[ $? -ne 0 ]] && command -v pipx &> /dev/null; then
                pipx install ".[audio]"
            fi
        fi
        
        if [[ $? -eq 0 ]]; then
            print_success "Audio dependencies installed!"
        else
            print_warning "Audio dependencies installation failed (you can install them later)"
        fi
    fi
}

# Check installation
verify_installation() {
    print_status "Verifying installation..."
    
    if command -v easyvoice &> /dev/null; then
        print_success "easyvoice command is available globally"
        
        # Test the command
        echo "quit" | easyvoice > /dev/null 2>&1
        if [[ $? -eq 0 ]]; then
            print_success "EasyVoice CLI is working correctly!"
        else
            print_warning "Command installed but may have issues"
        fi
    else
        print_warning "easyvoice command not found in PATH"
        
        # Try to find where it was installed
        PYTHON_USER_BASE=$($PYTHON_CMD -m site --user-base)
        USER_BIN="$PYTHON_USER_BASE/bin"
        
        if [[ -f "$USER_BIN/easyvoice" ]]; then
            print_warning "easyvoice is installed at: $USER_BIN/easyvoice"
            print_status "Add $USER_BIN to your PATH to use 'easyvoice' globally"
            
            # OS-specific PATH instructions
            case $OS in
                "linux")
                    print_status "Add this to your ~/.bashrc or ~/.zshrc:"
                    echo "export PATH=\"\$PATH:$USER_BIN\""
                    ;;
                "macos")
                    print_status "Add this to your ~/.bash_profile or ~/.zshrc:"
                    echo "export PATH=\"\$PATH:$USER_BIN\""
                    ;;
                "windows")
                    print_status "Add $USER_BIN to your Windows PATH environment variable"
                    ;;
            esac
        fi
    fi
}

# Post-installation instructions
show_usage() {
    print_success "Installation complete!"
    echo ""
    echo "🚀 Quick Start:"
    echo "  easyvoice          # Start interactive CLI"
    echo "  python3 -m easyvoice  # Alternative way to run"
    echo ""
    echo "📚 Available commands:"
    echo "  chat    - Start text conversation"
    echo "  ask     - Ask a single question"
    echo "  status  - Show system status"
    echo "  help    - Show all commands"
    echo ""
    echo "🔧 Optional audio setup:"
    echo "  pip install --user easyvoice[audio]  # Install audio dependencies"
    echo ""
    print_status "For more information, run: easyvoice help"
}

# Main installation flow
main() {
    detect_os
    check_python
    check_pip
    install_easyvoice
    verify_installation
    show_usage
}

# Run installation
main