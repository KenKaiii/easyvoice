# PowerShell installation script for EasyVoice CLI
# Works on Windows 10+ with PowerShell 5.1+

param(
    [switch]$Audio = $false,
    [switch]$Help = $false
)

if ($Help) {
    Write-Host "EasyVoice CLI PowerShell Installer" -ForegroundColor Cyan
    Write-Host "Usage: .\install.ps1 [-Audio] [-Help]" -ForegroundColor White
    Write-Host "  -Audio    Install audio processing dependencies" -ForegroundColor Green
    Write-Host "  -Help     Show this help message" -ForegroundColor Green
    exit 0
}

Write-Host "🎤 EasyVoice CLI Installation Script (PowerShell)" -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Cyan

function Write-Status {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

# Check Python installation
Write-Status "Checking Python installation..."
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python not found"
    }
    
    # Extract version number
    $versionMatch = [regex]::Match($pythonVersion, "Python (\d+)\.(\d+)")
    if ($versionMatch.Success) {
        $major = [int]$versionMatch.Groups[1].Value
        $minor = [int]$versionMatch.Groups[2].Value
        
        if ($major -eq 3 -and $minor -ge 10) {
            Write-Success "Python $($versionMatch.Groups[0].Value.Split(' ')[1]) found"
        } else {
            Write-Error "Python 3.10+ required, found $($versionMatch.Groups[0].Value.Split(' ')[1])"
            exit 1
        }
    } else {
        Write-Warning "Could not parse Python version: $pythonVersion"
    }
} catch {
    Write-Error "Python is not installed or not in PATH"
    Write-Host "Please install Python 3.10+ from https://python.org" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

# Check pip
Write-Status "Checking pip installation..."
try {
    python -m pip --version | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "pip not available"
    }
    Write-Success "pip is available"
} catch {
    Write-Error "pip is not available"
    Write-Host "Please ensure pip is installed with Python" -ForegroundColor White
    Read-Host "Press Enter to exit"
    exit 1
}

# Install EasyVoice
Write-Status "Installing EasyVoice CLI..."
try {
    python -m pip install --user . 
    if ($LASTEXITCODE -ne 0) {
        throw "Installation failed"
    }
    Write-Success "EasyVoice CLI installed successfully!"
} catch {
    Write-Error "Installation failed"
    Read-Host "Press Enter to exit"
    exit 1
}

# Install audio dependencies if requested or ask user
if ($Audio) {
    $installAudio = $true
} else {
    $response = Read-Host "Do you want to install audio processing dependencies? (y/N)"
    $installAudio = $response -match "^[Yy]"
}

if ($installAudio) {
    Write-Status "Installing audio dependencies (this may take a while)..."
    try {
        python -m pip install --user ".[audio]"
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Audio dependencies installed!"
        } else {
            Write-Warning "Audio dependencies installation failed (you can install them later)"
        }
    } catch {
        Write-Warning "Audio dependencies installation failed (you can install them later)"
    }
}

# Verify installation
Write-Status "Verifying installation..."
try {
    Get-Command easyvoice -ErrorAction Stop | Out-Null
    Write-Success "easyvoice command is available globally"
    
    # Test the command
    "quit" | easyvoice | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Success "EasyVoice CLI is working correctly!"
    } else {
        Write-Warning "Command installed but may have issues"
    }
} catch {
    Write-Warning "easyvoice command not found in PATH"
    
    # Try to find where it was installed
    try {
        $userBase = python -m site --user-base
        $userScripts = Join-Path $userBase "Scripts"
        
        if (Test-Path (Join-Path $userScripts "easyvoice.exe")) {
            Write-Warning "easyvoice is installed at: $(Join-Path $userScripts 'easyvoice.exe')"
            Write-Status "Add $userScripts to your Windows PATH to use 'easyvoice' globally"
            Write-Host "To add to PATH:" -ForegroundColor White
            Write-Host "  1. Press Win+R, type 'sysdm.cpl', press Enter" -ForegroundColor Gray
            Write-Host "  2. Click 'Environment Variables'" -ForegroundColor Gray
            Write-Host "  3. Under User variables, select PATH and click Edit" -ForegroundColor Gray
            Write-Host "  4. Click New and add: $userScripts" -ForegroundColor Gray
            Write-Host "  5. Click OK on all dialogs" -ForegroundColor Gray
        }
    } catch {
        Write-Warning "Could not determine installation location"
    }
}

Write-Host ""
Write-Success "Installation complete!"
Write-Host ""
Write-Host "🚀 Quick Start:" -ForegroundColor White
Write-Host "  easyvoice                 # Start interactive CLI" -ForegroundColor Gray
Write-Host "  python -m easyvoice       # Alternative way to run" -ForegroundColor Gray
Write-Host ""
Write-Host "📚 Available commands:" -ForegroundColor White
Write-Host "  chat    - Start text conversation" -ForegroundColor Gray
Write-Host "  ask     - Ask a single question" -ForegroundColor Gray
Write-Host "  status  - Show system status" -ForegroundColor Gray
Write-Host "  help    - Show all commands" -ForegroundColor Gray
Write-Host ""
Write-Host "🔧 Optional audio setup:" -ForegroundColor White
Write-Host "  pip install --user easyvoice[audio]  # Install audio dependencies" -ForegroundColor Gray
Write-Host ""
Write-Status "For more information, run: easyvoice help"

Read-Host "Press Enter to exit"